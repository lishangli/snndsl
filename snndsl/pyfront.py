"""Python-embedded front end: build the same IR by calling methods.

The text DSL is declarative but flat -- ``pipeline:`` cannot express a loop, a
conditional, or a helper function. This front end keeps the same IR and the
same backends while letting ordinary Python control flow generate the graph::

    import snndsl as snn

    LIF = snn.LIF(tau=2.0, v_th=1.0)

    net = snn.Net("mnist", steps=8, readout="membrane")
    x = net.input(1, 28, 28)
    h = x
    for i in range(3):                       # three identical blocks
        h = net.neuron(net.dense(h, 256, scale=2.0), LIF)
    net.output(net.neuron(net.dense(h, 10), LIF))
    net.train(data="mnist", epochs=4, lr=1e-3, sched="cosine")

    res = net.compile()                      # same passes, same codegen

Neuron *equations* stay as strings, because they are mathematics and the text
parser already validates them; only the graph construction becomes Python.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Sequence, Tuple

from . import parser as snn_parser
from .errors import SNNError
from .ir import IRGraph, IRNode, NeuronSpec, TemporalSpec
from .sema import build_neuron_spec


class Tensor:
    """Handle to an IR node. Carries the shape so mistakes surface early."""

    __slots__ = ("net", "node")

    def __init__(self, net: "Net", node: IRNode):
        self.net = net
        self.node = node

    @property
    def id(self) -> str:
        return self.node.id

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.node.shape)

    def __repr__(self) -> str:
        return f"<Tensor {self.name} {list(self.shape)}>"


class Neuron:
    """A neuron model: equations in the DSL's own syntax, validated on use."""

    def __init__(self, name: str, params: Dict[str, float], state: Dict[str, float],
                 body: Sequence[str], surrogate: Optional[str] = None,
                 reset: Optional[str] = None):
        self.name = name
        self.params = dict(params)
        self.state = dict(state)
        self.body = list(body)
        self.surrogate = surrogate
        self.reset = reset
        self._spec: Optional[NeuronSpec] = None

    # ------------------------------------------------------------- lowering
    def to_source(self) -> str:
        lines = [f"neuron {self.name}:"]
        for k, v in self.params.items():
            lines.append(f"    param {k} = {v!r}")
        for k, v in self.state.items():
            lines.append(f"    state {k} = {v!r}")
        for ln in self.body:
            lines.append(f"    {ln}")
        if self.surrogate:
            lines.append(f"    surrogate = {self.surrogate}")
        if self.reset:
            lines.append(f"    reset = {self.reset}")
        return "\n".join(lines) + "\n"

    def spec(self, reset_default: str = "subtract") -> NeuronSpec:
        if self._spec is None:
            unit = snn_parser.parse(self.to_source(), f"<neuron {self.name}>")
            decl = unit.all("neuron")[0]
            self._spec = build_neuron_spec(decl, {}, reset_default)
        return self._spec

    def __repr__(self) -> str:
        return f"<Neuron {self.name} params={self.params}>"


def LIF(tau: float = 2.0, v_th: float = 1.0, surrogate: str = "fast_sigmoid",
        alpha: float = 4.0, name: str = "LIF") -> Neuron:
    """Leaky integrate-and-fire: the standard baseline."""
    return Neuron(
        name=name, params={"tau": tau, "v_th": v_th}, state={"v": 0.0},
        body=["v = v * (1 - dt / tau) + i",
              "spike s = v > v_th",
              "v = v - s * v_th"],
        surrogate=f"{surrogate}(alpha={alpha})")


def ALIF(tau: float = 2.0, tau_a: float = 8.0, v_th: float = 1.0,
         beta: float = 0.25, surrogate: str = "atan", alpha: float = 2.0,
         name: str = "ALIF") -> Neuron:
    """Adaptive-threshold LIF: threshold rises after each spike."""
    return Neuron(
        name=name,
        params={"tau": tau, "tau_a": tau_a, "v_th": v_th, "beta": beta},
        state={"v": 0.0, "a": 0.0},
        body=["v = v * (1 - dt / tau) + i",
              "a = a * (1 - dt / tau_a)",
              "spike s = v > v_th + beta * a",
              "v = v - s * v_th",
              "a = a + s"],
        surrogate=f"{surrogate}(alpha={alpha})")


def IF(v_th: float = 1.0, surrogate: str = "fast_sigmoid", alpha: float = 4.0,
       name: str = "IF") -> Neuron:
    """Non-leaky integrate-and-fire."""
    return Neuron(
        name=name, params={"v_th": v_th}, state={"v": 0.0},
        body=["v = v + i", "spike s = v > v_th", "v = v - s * v_th"],
        surrogate=f"{surrogate}(alpha={alpha})")


def custom(name: str, params: Dict[str, float], state: Dict[str, float],
           body: Sequence[str], surrogate: Optional[str] = None) -> Neuron:
    """Escape hatch: write the equations yourself, in the DSL's syntax."""
    return Neuron(name=name, params=params, state=state, body=body,
                  surrogate=surrogate)


class Net:
    """Accumulates IR nodes. Compile at the end; nothing runs before that."""

    def __init__(self, name: str = "snn", steps: int = 8, dt: float = 1.0,
                 reset: str = "subtract", readout: str = "membrane",
                 schedule: str = "auto"):
        self.graph = IRGraph(name=name, filename=f"<net {name}>")
        self.graph.temporal = TemporalSpec(steps=int(steps), dt=float(dt),
                                           reset=reset, strategy=schedule)
        self.graph.infer["readout"] = readout
        self._counts: Dict[str, int] = {}
        self._used_neurons: Dict[str, NeuronSpec] = {}
        self._closed = False

    # ------------------------------------------------------------- internals
    _PREFIX: ClassVar[Dict[str, str]] = {"input": "x", "encode": "e", "dense": "d", "conv": "c",
               "pool": "p", "flatten": "f", "dropout": "drop", "neuron": "n",
               "add": "add", "concat": "cat", "output": "out"}

    def _fresh(self, base: str) -> str:
        stem = self._PREFIX.get(base, base)
        n = self._counts.get(stem, 0)
        self._counts[stem] = n + 1
        return stem if n == 0 else f"{stem}{n + 1}"

    def _add(self, op: str, inputs: Sequence[Tensor], shape: Tuple[int, ...],
             attrs: Dict[str, Any], node_id: Optional[str] = None,
             name: Optional[str] = None) -> Tensor:
        if self._closed:
            raise SNNError("this Net is already compiled; start a new one")
        # an explicit name becomes the node id, so generated code reads well
        nid = node_id or name or self._fresh(op)
        if any(n.id == nid for n in self.graph.nodes):
            raise SNNError(f"duplicate node name {nid!r}",
                           hint="pass name= differently for each layer")
        node = IRNode(id=nid, op=op, inputs=[t.id for t in inputs],
                      attrs=dict(attrs), shape=tuple(int(d) for d in shape),
                      name=name or nid)
        self.graph.add(node)
        return Tensor(self, node)

    def _register(self, neuron: Neuron) -> str:
        """Allocate a unique spec name so two LIFs with different tau coexist."""
        spec = neuron.spec(self.graph.temporal.reset)
        key = (tuple(sorted(spec.params.items())), tuple(sorted(spec.states.items())),
               tuple(spec.body), spec.surrogate, spec.surrogate_alpha)
        for name, existing in self._used_neurons.items():
            if existing == key:
                return name
        base = spec.name
        name = base
        n = 2
        while name in self.graph.neurons:
            name = f"{base}{n}"
            n += 1
        spec.name = name
        self.graph.neurons[name] = spec
        self._used_neurons[name] = key
        return name

    # --------------------------------------------------------------- layers
    def input(self, *dims: int, name: Optional[str] = None) -> Tensor:
        """Declare the network input. ``input(1, 28, 28)`` or ``input(784)``."""
        shape = tuple(int(d) for d in (dims[0] if len(dims) == 1 and
                                       isinstance(dims[0], (tuple, list)) else dims))
        t = self._add("input", [], shape, {"dims": list(shape)}, name=name)
        self.graph.inputs.append(t.id)
        return t

    def encode(self, x: Tensor, scheme: str = "rate", gain: float = 1.0) -> Tensor:
        """Turn analog input into spikes/current. ``rate`` = constant current."""
        return self._add("encode", [x], x.shape,
                         {"scheme": scheme, "window": self.graph.temporal.steps,
                          "gain": float(gain)})

    def dense(self, x: Tensor, units: int, scale: float = 1.0, bias: bool = True,
              init: str = "kaiming_normal", name: Optional[str] = None) -> Tensor:
        in_features = 1
        for d in x.shape:
            in_features *= int(d)
        return self._add("dense", [x], (int(units),), {
            "units": int(units), "in_features": in_features,
            "bias": bool(bias), "init": init,
            "flatten_input": len(x.shape) != 1,
            "weight_scale": float(scale),
        }, name=name)

    def conv(self, x: Tensor, out_channels: int, kernel: int = 3,
             stride: int = 1, padding: int = 0, scale: float = 1.0,
             bias: bool = True, name: Optional[str] = None) -> Tensor:
        if len(x.shape) != 3:
            raise SNNError(f"conv expects (C,H,W), got {list(x.shape)}")
        k, s, p = int(kernel), int(stride), int(padding)
        h = (int(x.shape[1]) + 2 * p - k) // s + 1
        w = (int(x.shape[2]) + 2 * p - k) // s + 1
        if h < 1 or w < 1:
            raise SNNError(f"conv output would be empty: {list(x.shape)} with "
                           f"kernel={k}, stride={s}, padding={p}")
        return self._add("conv", [x], (int(out_channels), h, w), {
            "out_channels": int(out_channels), "in_channels": int(x.shape[0]),
            "kernel": k, "stride": s, "padding": p,
            "bias": bool(bias), "weight_scale": float(scale),
        }, name=name)

    def pool(self, x: Tensor, kind: str = "max", kernel: int = 2,
             stride: Optional[int] = None, name: Optional[str] = None) -> Tensor:
        if len(x.shape) != 3:
            raise SNNError(f"pool expects (C,H,W), got {list(x.shape)}")
        k = int(kernel)
        s = int(stride or k)
        h = (int(x.shape[1]) - k) // s + 1
        w = (int(x.shape[2]) - k) // s + 1
        if h < 1 or w < 1:
            raise SNNError(f"pool output would be empty: {list(x.shape)}")
        return self._add("pool", [x], (int(x.shape[0]), h, w), {
            "kind": kind, "kernel": k, "stride": s,
        }, name=name)

    def flatten(self, x: Tensor, name: Optional[str] = None) -> Tensor:
        total = 1
        for d in x.shape:
            total *= int(d)
        return self._add("flatten", [x], (total,), {}, name=name)

    def dropout(self, x: Tensor, p: float = 0.1, name: Optional[str] = None) -> Tensor:
        return self._add("dropout", [x], x.shape, {"p": float(p)}, name=name)

    def neuron(self, x: Tensor, model: Neuron, name: Optional[str] = None) -> Tensor:
        """Apply a neuron model elementwise, once per time step."""
        model_name = self._register(model)
        return self._add("neuron", [x], x.shape, {"model": model_name}, name=name)

    def add(self, *xs: Tensor, name: Optional[str] = None) -> Tensor:
        if len(xs) < 2:
            raise SNNError("add needs at least two tensors")
        for other in xs[1:]:
            if tuple(other.shape) != tuple(xs[0].shape):
                raise SNNError(f"add shape mismatch: {list(xs[0].shape)} vs "
                               f"{list(other.shape)}")
        return self._add("add", list(xs), xs[0].shape, {}, name=name)

    def concat(self, *xs: Tensor, axis: int = 1, name: Optional[str] = None) -> Tensor:
        if len(xs) < 2:
            raise SNNError("concat needs at least two tensors")
        shape = list(xs[0].shape)
        shape[axis] = sum(int(t.shape[axis]) for t in xs)
        return self._add("concat", list(xs), tuple(shape), {"axis": int(axis)},
                         name=name)

    def output(self, x: Tensor, readout: Optional[str] = None,
               name: Optional[str] = None) -> Tensor:
        attrs = {"num_classes": int(x.shape[-1]) if x.shape else 1}
        if readout:
            self.graph.infer["readout"] = readout
        t = self._add("output", [x], x.shape, attrs, name=name)
        self.graph.outputs.append(t.id)
        return t

    # ---------------------------------------------------------------- config
    def train(self, **kwargs) -> "Net":
        self.graph.train.update(kwargs)
        return self

    def infer(self, **kwargs) -> "Net":
        self.graph.infer.update(kwargs)
        return self

    def budget(self, **kwargs) -> "Net":
        self.graph.budgets.update(kwargs)
        return self

    # --------------------------------------------------------------- compile
    def compile(self, passes: Optional[Sequence[str]] = None,
                class_name: str = "SnnNet", quiet: bool = False):
        """Run the same optimisation pipeline and code generation as .snn."""
        from .pipeline import finalize
        if not self.graph.outputs:
            raise SNNError("network has no output; call net.output(...)")
        if not any(n.op == "neuron" for n in self.graph.nodes):
            raise SNNError(
                "network has no spiking neuron: this is a plain ANN",
                hint="wrap a layer: net.neuron(net.dense(x, 256), LIF)")
        self._closed = True
        return finalize(self.graph, passes=passes, class_name=class_name,
                        quiet=quiet, filename=self.graph.filename)

    # ----------------------------------------------------------------- run
    def run(self, out_dir: str = "build", **kwargs) -> Dict[str, Any]:
        """Compile, then train/measure exactly like ``snndsl run``."""
        import time as _time

        from .runner import run_program_from_result
        t0 = _time.time()
        result = self.compile()
        return run_program_from_result(result, out_dir=out_dir,
                                       compile_ms=(_time.time() - t0) * 1000.0,
                                       **kwargs)

    def describe(self) -> str:
        lines = [f"program {self.graph.name}  T={self.graph.temporal.steps} "
                 f"readout={self.graph.infer.get('readout')}"]
        for n in self.graph.nodes:
            ins = ", ".join(n.inputs) or "-"
            lines.append(f"  {n.id:<14} {n.op:<8} {list(n.shape)!s:<18} <- {ins}")
        return "\n".join(lines)


def net(name: str = "snn", **kwargs) -> Net:
    """Alias so ``with snn.net(...)`` style reads naturally as a factory."""
    return Net(name, **kwargs)
