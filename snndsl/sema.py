"""Semantic analysis: name resolution, constant folding, shape inference, IR build.

This is the stage that turns a *description* into a *program*: every pipeline
statement becomes an IR node with a statically known shape, and every neuron
body is rewritten into Python source that the backends embed verbatim.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from . import ast as A
from .errors import DiagnosticBag, SemaError
from .ir import IRGraph, IRNode, NeuronSpec, TemporalSpec

MATH_FUNCS = {
    "min": min, "max": max, "abs": abs, "int": int, "float": float,
    "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "round": round,
    "pow": pow, "ceil": math.ceil, "floor": math.floor,
}

PY_BINOPS = {
    "+": "+", "-": "-", "*": "*", "/": "/", "%": "%",
    "**": "**", "^": "**",
    "==": "==", "!=": "!=", "<": "<", "<=": "<=", ">": ">", ">=": ">=",
    "&&": "and", "||": "or",
}

SURROGATES = {
    "fast_sigmoid": 4.0, "sigmoid": 4.0, "atan": 2.0, "triangle": 1.0,
    "super_spike": 4.0, "rectangular": 1.0, "straight_through": 1.0,
}


# ------------------------------------------------------------ expression eval
def eval_expr(e: A.Expr, env: Dict[str, Any], what: str = "expression") -> Any:
    if isinstance(e, A.Number):
        v = e.value
        return int(v) if float(v).is_integer() and abs(v) < 1e15 else float(v)
    if isinstance(e, A.String):
        return e.value
    if isinstance(e, A.Bool):
        return e.value
    if isinstance(e, A.TupleLit):
        return tuple(eval_expr(i, env) for i in e.items)
    if isinstance(e, A.Name):
        if e.id in env:
            return env[e.id]
        raise SemaError(f"unknown identifier {e.id!r} in {what}", line=e.line, col=e.col)
    if isinstance(e, A.UnaryOp):
        v = eval_expr(e.operand, env, what)
        if e.op == "-":
            return -v
        if e.op == "+":
            return v
        if e.op in ("!", "~"):
            return not v
        raise SemaError(f"unsupported unary operator {e.op!r}", line=e.line, col=e.col)
    if isinstance(e, A.BinOp):
        l = eval_expr(e.left, env, what)  # noqa: E741 - l/r pair in operator code
        r = eval_expr(e.right, env, what)
        op = e.op
        if op == "+":
            return l + r
        if op == "-":
            return l - r
        if op == "*":
            return l * r
        if op == "/":
            if r == 0:
                raise SemaError("division by zero in constant expression",
                                line=e.line, col=e.col)
            return l / r
        if op == "%":
            return l % r
        if op in ("**", "^"):
            return l ** r
        if op == "==":
            return l == r
        if op == "!=":
            return l != r
        if op == "<":
            return l < r
        if op == "<=":
            return l <= r
        if op == ">":
            return l > r
        if op == ">=":
            return l >= r
        if op == "&&":
            return bool(l) and bool(r)
        if op == "||":
            return bool(l) or bool(r)
        raise SemaError(f"unsupported operator {op!r}", line=e.line, col=e.col)
    if isinstance(e, A.Call):
        fn = MATH_FUNCS.get(e.func)
        if fn is None:
            raise SemaError(f"unknown function {e.func!r} in constant expression",
                            line=e.line, col=e.col,
                            hint=f"available: {', '.join(sorted(MATH_FUNCS))}")
        args = [eval_expr(a, env, what) for a in e.args]
        kwargs = {k: eval_expr(v, env, what) for k, v in e.kwargs.items()}
        return fn(*args, **kwargs)
    raise SemaError(f"cannot evaluate {type(e).__name__}", line=getattr(e, "line", 0))


def expr_to_src(e: A.Expr, scope: set) -> str:
    """Lower a DSL expression to Python source for a neuron update."""
    if isinstance(e, A.Number):
        v = e.value
        return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(float(v))
    if isinstance(e, A.Bool):
        return "True" if e.value else "False"
    if isinstance(e, A.String):
        return repr(e.value)
    if isinstance(e, A.Name):
        if e.id not in scope:
            raise SemaError(
                f"unbound symbol {e.id!r} in neuron update", line=e.line, col=e.col,
                hint=f"in scope: {', '.join(sorted(scope))}")
        return e.id
    if isinstance(e, A.UnaryOp):
        inner = expr_to_src(e.operand, scope)
        if e.op in ("!", "~"):
            return f"(not {inner})"
        return f"({e.op}{inner})"
    if isinstance(e, A.BinOp):
        l = expr_to_src(e.left, scope)  # noqa: E741 - l/r pair in operator code
        r = expr_to_src(e.right, scope)
        op = PY_BINOPS.get(e.op)
        if op is None:
            raise SemaError(f"unsupported operator {e.op!r}", line=e.line, col=e.col)
        # '^' lowers to '**' but Python binds it tighter than unary minus; parens cover it
        return f"({l} {op} {r})"
    if isinstance(e, A.Call):
        if e.func not in MATH_FUNCS and e.func not in ("where", "clamp", "sign"):
            raise SemaError(f"function {e.func!r} is not available in neuron updates",
                            line=e.line, col=e.col)
        args = ", ".join(expr_to_src(a, scope) for a in e.args)
        kws = ", ".join(f"{k}={expr_to_src(v, scope)}" for k, v in e.kwargs.items())
        sep = ", " if args and kws else ""
        return f"{e.func}({args}{sep}{kws})"
    raise SemaError(f"cannot lower {type(e).__name__} to source",
                    line=getattr(e, "line", 0))


def lower_spike(expr: A.Expr, scope: set, sur_kind: str, alpha: float) -> str:
    """Lower ``v > v_th`` into ``surrogate((v) - (v_th), alpha)``.

    A comparison has no gradient; the surrogate supplies one on the backward
    path while the forward value stays a hard 0/1 spike.
    """
    if isinstance(expr, A.BinOp) and expr.op in (">", ">=", "<", "<="):
        lhs = expr_to_src(expr.left, scope)
        rhs = expr_to_src(expr.right, scope)
        if expr.op in (">", ">="):
            diff = f"({lhs}) - ({rhs})"
        else:
            diff = f"({rhs}) - ({lhs})"
        return f"surrogate_{sur_kind}({diff}, {alpha})"
    # constant / already-boolean spike predicate: fall back to straight-through
    return f"surrogate_straight_through({expr_to_src(expr, scope)}, {alpha})"


def build_neuron_spec(decl: "A.NeuronDecl", env: Dict[str, Any],
                      temporal_reset: str = "subtract",
                      bag: Optional["DiagnosticBag"] = None) -> NeuronSpec:
    """Turn a parsed ``neuron`` block into a validated NeuronSpec.

    Shared by both front ends: the text parser hands in a parsed declaration,
    the Python embedding builds an equivalent one from strings. Keeping the
    lowering here means the two paths cannot produce different equations.
    """
    params = {k: float(eval_expr(v, env, f"neuron {decl.name} param {k}"))
              for k, v in decl.params.items()}
    states = {k: float(eval_expr(v, env, f"neuron {decl.name} state {k}"))
              for k, v in decl.states.items()}
    # surrogate is resolved first: the spike line lowers to a call to it
    sur_kind, sur_alpha = "fast_sigmoid", 4.0
    if decl.surrogate is not None:
        se = decl.surrogate
        if isinstance(se, A.Call):
            sur_kind = se.func
            if "alpha" in se.kwargs:
                sur_alpha = float(eval_expr(se.kwargs["alpha"], env, "surrogate"))
            elif se.args:
                sur_alpha = float(eval_expr(se.args[0], env, "surrogate"))
            elif sur_kind in SURROGATES:
                sur_alpha = SURROGATES[sur_kind]
        elif isinstance(se, A.Name):
            sur_kind = se.id
            sur_alpha = SURROGATES.get(sur_kind, 4.0)
        else:
            raise SemaError("surrogate must be a name or call", line=decl.line)
    if sur_kind not in SURROGATES:
        if bag is not None:
            bag.add("sema",
                    f"unknown surrogate {sur_kind!r}, falling back to fast_sigmoid",
                    line=decl.line, severity="warning")
        sur_kind = "fast_sigmoid"

    scope = set(params) | set(states) | {"i", "dt", "t"}
    body: List[Tuple[str, str, str]] = []
    for stmt in decl.body:
        try:
            if stmt.kind == "spike":
                src = lower_spike(stmt.expr, scope, sur_kind, sur_alpha)
            else:
                src = expr_to_src(stmt.expr, scope)
        except SemaError as err:
            err.diagnostic.message = f"in neuron {decl.name}: {err.diagnostic.message}"
            raise
        body.append((stmt.kind, stmt.target, src))
        scope.add(stmt.target)
    if not any(k == "spike" for k, _, _ in body):
        raise SemaError(f"neuron {decl.name!r} never declares a spike",
                        line=decl.line,
                        hint="add a line like 'spike s = v > v_th'")
    return NeuronSpec(
        name=decl.name, params=params, states=states, body=body,
        spike_var=decl.spike_var, surrogate=sur_kind,
        surrogate_alpha=sur_alpha, reset=decl.reset or temporal_reset)


# ------------------------------------------------------------------ analyzer
class Analyzer:
    def __init__(self, unit: A.Unit):
        self.unit = unit
        self.bag = DiagnosticBag()
        self.env: Dict[str, Any] = {}          # constants + program metadata
        self.graph = IRGraph(source=unit.source, filename=unit.filename)
        self.names: Dict[str, str] = {}        # DSL name -> current value id
        self.shape_of: Dict[str, Tuple[int, ...]] = {}
        self._seq: Dict[str, int] = {}

    # ------------------------------------------------------------- helpers
    def _fresh_id(self, base: str) -> str:
        k = self._seq.get(base, 0)
        self._seq[base] = k + 1
        return base if k == 0 else f"{base}${k}"

    def bind(self, dsl_name: str, node: IRNode) -> None:
        self.names[dsl_name] = node.id
        self.shape_of[dsl_name] = node.shape

    def resolve(self, dsl_name: str, line: int) -> IRNode:
        if dsl_name not in self.names:
            raise SemaError(
                f"unknown value {dsl_name!r} referenced in pipeline", line=line,
                hint=f"defined so far: {', '.join(self.names) or '(none)'}")
        return self.graph.node(self.names[dsl_name])

    # ---------------------------------------------------------------- main
    def run(self) -> IRGraph:
        self._collect_consts()
        self._declarations()
        self._temporal()
        self._neurons()
        self._encoders()
        self._pipeline()
        self._configs()
        if not self.graph.outputs:
            raise SemaError("pipeline defines no output; add an 'output' statement")
        self._check_is_spiking()
        self.graph.meta["dsl_names"] = dict(self.names)
        return self.graph

    def _check_is_spiking(self) -> None:
        """A .snn program that never fires a spike is not an SNN.

        Writing ``dense h`` and forgetting ``neuron h`` silently yields an
        analog MLP: it trains, it predicts, and it reports 0% sparsity. That is
        exactly the mistake a newcomer makes, so refuse it here instead of
        letting it look like a working spiking network.
        """
        if any(n.op == "neuron" for n in self.graph.nodes):
            return
        analog = [n for n in self.graph.nodes if n.op in ("dense", "conv")]
        if not analog:
            return
        first = analog[0]
        raise SemaError(
            f"program has no spiking neuron: '{first.name}' outputs an analog "
            "current, so this is a plain ANN",
            line=first.line,
            hint=f"add 'neuron {first.name} -> LIF' after it, "
                 "or 'output out <- ... with readout=membrane' if you really "
                 "want a non-spiking readout")

    def _collect_consts(self) -> None:
        for d in self.unit.all("const"):
            assert isinstance(d, A.ConstDecl)
            val = eval_expr(d.value, self.env, f"const {d.name}")
            self.env[d.name] = val
        if any(isinstance(d, A.ProgramDecl) for d in self.unit.decls):
            self.graph.name = next(d.name for d in self.unit.decls
                                   if isinstance(d, A.ProgramDecl))

    def _declarations(self) -> None:
        # surface declared neuron/encoder names early so pipeline errors are clearer
        self.graph.meta["declared_neurons"] = [d.name for d in self.unit.all("neuron")]
        self.graph.meta["declared_encoders"] = [d.name for d in self.unit.all("encoder")]

    def _temporal(self) -> None:
        decl = self.unit.first("temporal")
        ts = TemporalSpec()
        if decl is not None:
            assert isinstance(decl, A.TemporalDecl)
            vals = {}
            for k, v in decl.values.items():
                try:
                    vals[k] = eval_expr(v, self.env, "temporal")
                except SemaError:
                    # symbolic settings such as 'reset = subtract' are names, not values
                    if isinstance(v, A.Name):
                        vals[k] = v.id
                    else:
                        raise
            unknown = set(vals) - {"steps", "dt", "reset", "strategy"}
            if unknown:
                self.bag.add("sema", f"unknown temporal key(s): {', '.join(sorted(unknown))}",
                             line=decl.line, severity="warning")
            ts.steps = int(vals.get("steps", ts.steps))
            ts.dt = float(vals.get("dt", ts.dt))
            ts.reset = str(vals.get("reset", ts.reset))
            if ts.steps < 1:
                raise SemaError("temporal.steps must be >= 1", line=decl.line)
            if ts.dt <= 0:
                raise SemaError("temporal.dt must be > 0", line=decl.line)
        else:
            self.bag.add(
                "sema",
                f"no 'temporal:' block, defaulting to steps={ts.steps}, dt={ts.dt}",
                hint="state the time semantics explicitly: temporal: steps = 8",
                severity="warning")
        self.graph.temporal = ts

    def _neurons(self) -> None:
        for d in self.unit.all("neuron"):
            assert isinstance(d, A.NeuronDecl)
            self.graph.neurons[d.name] = build_neuron_spec(
                d, self.env, self.graph.temporal.reset, bag=self.bag)

    @staticmethod
    def _lower_spike(expr: A.Expr, scope: set, sur_kind: str, alpha: float) -> str:
        return lower_spike(expr, scope, sur_kind, alpha)

    def _encoders(self) -> None:
        for d in self.unit.all("encoder"):
            assert isinstance(d, A.EncoderDecl)
            params = {k: eval_expr(v, self.env, f"encoder {d.name}")
                      for k, v in d.params.items()}
            self.graph.encoders[d.name] = {"params": params}

    # ------------------------------------------------------------- pipeline
    def _spec(self, layer: A.LayerDecl) -> List[Any]:
        keys = sorted((k for k in layer.attrs if k.startswith("spec")),
                      key=lambda s: int(s[4:]))
        out = []
        for k in keys:
            out.append(self._value(layer.attrs[k], f"{layer.op} {layer.name}"))
        return out

    def _value(self, expr: A.Expr, what: str) -> Any:
        """Evaluate, falling back to a symbolic name (e.g. ``kind=max``)."""
        try:
            return eval_expr(expr, self.env, what)
        except SemaError:
            if isinstance(expr, A.Name):
                return expr.id
            raise

    def _attr(self, layer: A.LayerDecl, key: str, default: Any) -> Any:
        if key not in layer.attrs:
            return default
        return self._value(layer.attrs[key], f"{layer.op} {layer.name}")

    def _pipeline(self) -> None:
        decl = self.unit.first("pipeline")
        if decl is None:
            raise SemaError("unit has no 'pipeline:' block")
        assert isinstance(decl, A.PipelineDecl)
        for layer in decl.stmts:
            self._lower_layer(layer)

    def _lower_layer(self, layer: A.LayerDecl) -> None:
        op = layer.op
        spec = self._spec(layer)

        handler = {
            "input": self._op_input, "dense": self._op_dense, "conv": self._op_conv,
            "pool": self._op_pool, "flatten": self._op_flatten,
            "dropout": self._op_dropout, "neuron": self._op_neuron,
            "add": self._op_add, "concat": self._op_concat,
            "encode": self._op_encode, "decode": self._op_decode,
            "output": self._op_output,
        }.get(op)
        if handler is None:
            raise SemaError(f"unknown pipeline operation {op!r}", line=layer.line,
                            col=layer.col,
                            hint="supported: input, dense, conv, pool, flatten, "
                                 "dropout, neuron, add, concat, encode, decode, output")
        handler(layer, spec)

    def _emit(self, layer: A.LayerDecl, op: str, dsl_name: str, inputs: List[IRNode],
              shape: Tuple[int, ...], attrs: Dict[str, Any]) -> IRNode:
        node = IRNode(id=self._fresh_id(dsl_name or op), op=op,
                      inputs=[i.id for i in inputs], attrs=attrs,
                      shape=shape, name=dsl_name or op, line=layer.line)
        self.graph.add(node)
        if dsl_name:
            self.bind(dsl_name, node)
        return node

    def _op_input(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        if not spec:
            raise SemaError("input requires a shape, e.g. 'input x : 1, 28, 28'",
                            line=layer.line, col=layer.col)
        dims = tuple(int(s) for s in spec)
        if len(dims) == 1:
            shape = (dims[0],)
        elif len(dims) == 3:
            shape = dims
        elif len(dims) == 2:
            shape = (1, *dims)
        else:
            raise SemaError(f"input shape must be 1-D features or (C,H,W), got {dims}",
                            line=layer.line, col=layer.col)
        node = self._emit(layer, "input", layer.name, [], shape,
                          {"dims": list(shape)})
        self.graph.inputs.append(node.id)
        self.graph.meta.setdefault("input_name", layer.name)

    def _reject_loop_carried(self, layer: A.LayerDecl) -> None:
        """Self-reference means the user wants a recurrent synapse.

        Without this check the error is 'dense takes exactly one input', which
        tells them nothing about the real limitation.
        """
        for name in layer.inputs:
            if name == layer.name:
                raise SemaError(
                    f"'{layer.name}' reads its own output: recurrent (loop-carried) "
                    "connections are not supported",
                    line=layer.line, col=layer.col,
                    hint="state carried across time steps must live in a neuron "
                         "'state' variable, not in a graph edge; every layer here "
                         "is a feed-forward map applied once per time step")

    def _op_dense(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        if len(layer.inputs) != 1:
            self._reject_loop_carried(layer)
            raise SemaError("dense takes exactly one input: 'dense h : 256 <- x'",
                            line=layer.line, col=layer.col)
        src = self.resolve(layer.inputs[0], layer.line)
        if not spec:
            raise SemaError("dense requires a unit count", line=layer.line, col=layer.col)
        units = int(spec[0])
        in_features = 1
        for d in src.shape:
            in_features *= int(d)
        self._emit(layer, "dense", layer.name, [src], (units,), {
            "units": units, "in_features": in_features,
            "bias": bool(self._attr(layer, "bias", True)),
            "init": str(self._attr(layer, "init", "kaiming_normal")),
            "flatten_input": len(src.shape) != 1,
            "weight_scale": float(self._attr(layer, "scale", 1.0)),
        })

    def _op_conv(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        if len(layer.inputs) != 1:
            raise SemaError("conv takes exactly one input", line=layer.line,
                            col=layer.col)
        src = self.resolve(layer.inputs[0], layer.line)
        if len(src.shape) != 3:
            raise SemaError(f"conv expects a (C,H,W) input, got {list(src.shape)}",
                            line=layer.line, col=layer.col)
        if not spec:
            raise SemaError("conv requires an output channel count", line=layer.line)
        cout = int(spec[0])
        k = int(self._attr(layer, "k", self._attr(layer, "kernel", 3)))
        s = int(self._attr(layer, "s", self._attr(layer, "stride", 1)))
        p = int(self._attr(layer, "p", self._attr(layer, "padding", k // 2)))
        cin = int(src.shape[0])
        h_out = (int(src.shape[1]) + 2 * p - k) // s + 1
        w_out = (int(src.shape[2]) + 2 * p - k) // s + 1
        if h_out < 1 or w_out < 1:
            raise SemaError(f"conv kernel {k} with padding {p} collapses the map "
                            f"{list(src.shape)}", line=layer.line, col=layer.col)
        self._emit(layer, "conv", layer.name, [src], (cout, h_out, w_out), {
            "out_channels": cout, "in_channels": cin, "kernel": k, "stride": s,
            "padding": p, "bias": bool(self._attr(layer, "bias", True)),
            "groups": int(self._attr(layer, "groups", 1)),
            "init": str(self._attr(layer, "init", "kaiming_normal")),
        })

    def _op_pool(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        src = self.resolve(layer.inputs[0], layer.line)
        if len(src.shape) != 3:
            raise SemaError("pool expects a (C,H,W) input", line=layer.line)
        k = int(spec[0]) if spec else int(self._attr(layer, "k", 2))
        s = int(self._attr(layer, "s", k))
        kind = str(self._attr(layer, "kind", "avg"))
        h_out = (int(src.shape[1]) - k) // s + 1
        w_out = (int(src.shape[2]) - k) // s + 1
        self._emit(layer, "pool", layer.name, [src],
                   (int(src.shape[0]), h_out, w_out),
                   {"kernel": k, "stride": s, "kind": kind})

    def _op_flatten(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        src = self.resolve(layer.inputs[0], layer.line)
        n = 1
        for d in src.shape:
            n *= int(d)
        self._emit(layer, "flatten", layer.name, [src], (n,), {})

    def _op_dropout(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        src = self.resolve(layer.inputs[0], layer.line)
        p = float(spec[0]) if spec else float(self._attr(layer, "p", 0.1))
        self._emit(layer, "dropout", layer.name, [src], src.shape, {"p": p})

    def _op_neuron(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        if "target" not in layer.attrs:
            raise SemaError("neuron needs a model: 'neuron h -> LIF'",
                            line=layer.line, col=layer.col)
        model = layer.attrs["target"].id
        if model not in self.graph.neurons:
            raise SemaError(
                f"unknown neuron model {model!r}", line=layer.line, col=layer.col,
                hint=f"declared: {', '.join(self.graph.neurons) or '(none)'}")
        src = self.resolve(layer.name, layer.line)
        self._emit(layer, "neuron", layer.name, [src], src.shape, {"model": model})

    def _op_add(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        operands = layer.inputs or [s for s in spec if isinstance(s, str)]
        if len(operands) < 2:
            raise SemaError("add requires at least two inputs, e.g. 'add z : a, b'",
                            line=layer.line, col=layer.col)
        srcs = [self.resolve(n, layer.line) for n in operands]
        shape = srcs[0].shape
        for s in srcs[1:]:
            if tuple(s.shape) != tuple(shape):
                raise SemaError(f"add shape mismatch: {list(shape)} vs {list(s.shape)}",
                                line=layer.line, col=layer.col)
        self._emit(layer, "add", layer.name, srcs, shape, {})

    def _op_concat(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        operands = layer.inputs or [s for s in spec if isinstance(s, str)]
        if len(operands) < 2:
            raise SemaError("concat requires at least two inputs, e.g. "
                            "'concat z : a, b'", line=layer.line, col=layer.col)
        srcs = [self.resolve(n, layer.line) for n in operands]
        axis = int(self._attr(layer, "axis", 0))
        total = sum(int(s.shape[axis]) for s in srcs)
        shape = list(srcs[0].shape)
        shape[axis] = total
        self._emit(layer, "concat", layer.name, srcs, tuple(shape), {"axis": axis})

    def _op_encode(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        scheme = layer.attrs["target"].id if "target" in layer.attrs else "rate"
        src = self.resolve(layer.name, layer.line)
        window = int(self._attr(layer, "window", self.graph.temporal.steps))
        gain = float(self._attr(layer, "gain", 1.0))
        self._emit(layer, "encode", layer.name, [src], src.shape,
                   {"scheme": scheme, "window": window, "gain": gain})

    def _op_decode(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        scheme = layer.attrs["target"].id if "target" in layer.attrs else "rate"
        src = self.resolve(layer.name, layer.line)
        self._emit(layer, "decode", layer.name, [src], src.shape,
                   {"scheme": scheme, "window": int(self._attr(
                       layer, "window", self.graph.temporal.steps))})

    def _op_output(self, layer: A.LayerDecl, spec: List[Any]) -> None:
        if layer.inputs:
            src = self.resolve(layer.inputs[0], layer.line)
        else:
            src = self.resolve(layer.name, layer.line)
        node = self._emit(layer, "output", layer.name, [src], src.shape,
                          {"num_classes": int(src.shape[-1])})
        self.graph.outputs.append(node.id)

    # -------------------------------------------------------------- configs
    def _configs(self) -> None:
        for kind in ("train", "infer", "budget"):
            decl = self.unit.first(kind)
            if decl is None:
                continue
            assert isinstance(decl, A.ConfigDecl)
            vals: Dict[str, Any] = {}
            for k, v in decl.values.items():
                try:
                    vals[k] = eval_expr(v, self.env, f"{kind}.{k}")
                except SemaError:
                    # string-ish values such as 'adam' or 'ce' arrive as Name exprs
                    if isinstance(v, A.Name):
                        vals[k] = v.id
                    elif isinstance(v, A.Call):
                        vals[k] = {"fn": v.func,
                                   "args": [eval_expr(a, self.env) for a in v.args],
                                   "kwargs": {kk: eval_expr(vv, self.env)
                                              for kk, vv in v.kwargs.items()}}
                    else:
                        raise
            if kind == "train":
                self.graph.train = vals
            elif kind == "infer":
                ro = str(vals.get("readout", "spikes")).lower()
                if ro not in ("spikes", "membrane", "voltage", "rate", "current"):
                    self.bag.add("sema",
                                 f"unknown readout {ro!r}, expected "
                                 "spikes|membrane", severity="warning")
                elif ro in ("membrane", "voltage"):
                    vals["readout"] = "membrane"
                self.graph.infer = vals
            else:
                self.graph.budgets = vals


def analyze(unit: A.Unit) -> IRGraph:
    an = Analyzer(unit)
    graph = an.run()
    if an.bag.has_errors():
        an.bag.raise_if_errors(unit.source, unit.filename)
    graph.meta.setdefault("warnings", [d.message for d in an.bag.warnings])
    return graph
