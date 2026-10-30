"""Lower the SNN IR to MLIR text (builtin / func / memref / arith / scf).

Why this exists
---------------
The neuromorphic compiler ecosystem has consolidated on MLIR: SNAX builds
``snaxc`` on xdsl for RISC-V accelerator clusters, SNN-MLIR emits bare-metal C11
out of an MLIR dialect, and IREE / torch-mlir can load anything that parses. A
format only this project can read is a dead end; a format those tools can read
is an exit.

So this is deliberately the dumbest possible lowering: flat 1-D memrefs, nested
``scf.for``, scalar ``arith``. No tiling, no vectorisation, no buffer reuse
beyond what correctness needs. Optimising is the downstream compiler's job, and
downstream has better optimisers than a 7k-line project will ever have.

What it buys
------------
* The output parses with ``mlir-opt`` and with xdsl, and *executes* under the
  xdsl interpreter — see ``tests/test_mlir_export.py``, which runs it there and
  compares spikes against PyTorch. Two independent third parties, one file.
* SNAX's flow consumes ``linalg`` on memrefs and hands off to a verilated RTL
  model. A LIF time loop is not what ``snaxc`` was written for, so this module
  emits the shape of program that flow can consume — static shapes, no dynamic
  control flow, memrefs only — and leaves the mapping to snaxc.

What it deliberately does not do
--------------------------------
No conv, no pool, no residual merge. Those need index maps this project would
get subtly wrong, or a merge primitive NIR does not have either. They raise
:class:`MLIRUnsupported` with the reason, instead of emitting something that
parses and computes the wrong thing. A wrong file that loads is the exact
failure mode this project exists to catch.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..ir import IRGraph, IRNode

# Reuse the NIR path's numerical probe of the neuron body. Sharing it is the
# point: the two backends cannot disagree about what a neuron *is*, because
# there is one recogniser. A second, text-pattern-matching implementation here
# would be a second opinion that silently drifts.
from .nir_export import recognise_lif

SUPPORTED = ("input", "encode", "flatten", "dense", "neuron", "output")

# indentation levels, counted in two-space steps from the module body
_BODY = 2       # function body / the time loop itself
_LAYER = 3      # inside the time loop: a layer's outer loop
_INNER = 4      # inside that: scalar ops, or the reduction loop header
_INNER2 = 5     # inside the reduction loop


class MLIRUnsupported(ValueError):
    """The graph is valid SNN, but this lowering cannot express it."""


def _f32(x: float) -> str:
    return f"{float(x):.6e}"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c == "_" else "_" for c in name)


def _prod(shape: Tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


class _Emitter:
    """Two passes: declare buffers, then emit one shared time loop.

    The single shared loop is the whole point. A per-layer time loop is what
    you get by writing the lowering top to bottom, and it is wrong: layer 2
    would consume layer 1's spikes accumulated over all T steps instead of
    layer 1's spikes at step t. Correctness here is a structural property,
    not something to check numerically afterwards.
    """

    def __init__(self, graph: IRGraph, steps: Optional[int] = None):
        self.g = graph
        self.T = int(graph.temporal.steps if steps is None else steps)
        self.dt = float(graph.temporal.dt)
        self.pre: List[str] = []        # once, before the time loop
        self.step: List[str] = []       # every time step
        self.decls: List[str] = []      # constants and allocations
        self.n = 0
        self.consts: Dict[Tuple[str, str], str] = {}
        self.buf: Dict[str, str] = {}
        self.params: List[Tuple[str, str, Tuple[int, ...], str]] = []
        self.owner: Dict[str, str] = {}   # argument name -> IR node id
        self.lif: Dict[str, Dict[str, Any]] = {}

    # ---------------------------------------------------------------- helpers
    def fresh(self, hint: str = "v") -> str:
        self.n += 1
        return f"%{hint}{self.n}"

    def _decl(self, text: str) -> None:
        self.decls.append("  " * _BODY + text)

    def _pre(self, depth: int, text: str) -> None:
        self.pre.append("  " * depth + text)

    def _body(self, depth: int, text: str) -> None:
        self.step.append("  " * depth + text)

    def _const(self, kind: str, key: str, text: str) -> str:
        k = (kind, key)
        if k not in self.consts:
            name = self.fresh("c")
            self.consts[k] = name
            self._decl(f"{name} = arith.constant {text}")
        return self.consts[k]

    def c_f32(self, x: float) -> str:
        return self._const("f32", repr(float(x)), f"{_f32(x)} : f32")

    def c_index(self, x: int) -> str:
        return self._const("index", str(int(x)), f"{int(x)} : index")

    def alloc(self, size: int, hint: str, elem: str = "f32") -> str:
        name = self.fresh(hint)
        self._decl(f"{name} = memref.alloc() : memref<{size}x{elem}>")
        return name

    # -------------------------------------------------------------- planning
    def plan(self) -> None:
        seen = set()

        def param(name: str, kind: str, size: int, elem: str) -> None:
            if name in seen:
                raise MLIRUnsupported(f"duplicate argument name {name!r}")
            seen.add(name)
            self.params.append((name, kind, (size,), f"memref<{size}x{elem}>"))

        for node in self.g.nodes:
            if node.op not in SUPPORTED:
                raise MLIRUnsupported(
                    f"{node.op!r} has no MLIR lowering in this backend: conv and "
                    "pool need index maps, and a residual merge needs an add "
                    "primitive that NIR does not have either. Emitting a file "
                    "that parses but computes something else is worse than "
                    "refusing."
                )
            if node.op == "input":
                name = _safe(node.id) + "_in"
                param(name, "input", _prod(node.shape), "f32")
                self.owner[name] = node.id
            elif node.op == "dense":
                out, in_f = int(node.attrs["units"]), int(node.attrs["in_features"])
                param(f"w_{_safe(node.id)}", "weight", out * in_f, "f32")
                self.owner[f"w_{_safe(node.id)}"] = node.id
                if node.attrs.get("bias", True):
                    param(f"b_{_safe(node.id)}", "bias", out, "f32")
                    self.owner[f"b_{_safe(node.id)}"] = node.id
            elif node.op == "neuron":
                n = _prod(node.shape)
                param(f"v_{_safe(node.id)}", "state", n, "f32")
                param(f"s_{_safe(node.id)}", "spikes", self.T * n, "f32")
                self.owner[f"v_{_safe(node.id)}"] = node.id
                self.owner[f"s_{_safe(node.id)}"] = node.id

    def check(self) -> None:
        """Refuse up front; a half-emitted file is worse than no file."""
        for node in self.g.nodes:
            if node.op == "neuron":
                self.lif[node.id] = self._recognise(node)

    def _recognise(self, node: IRNode) -> Dict[str, Any]:
        lif = recognise_lif(self.g, node, dt=self.dt)
        if lif is None:
            raise MLIRUnsupported(
                f"neuron {node.attrs.get('model')!r} does not lower to the affine "
                "form (v <- c + a*v + b*i) that a hardware neuron primitive "
                "implements. A custom body trains in PyTorch but has no silicon "
                "counterpart, so it is refused here rather than approximated."
            )
        if lif["reset_mode"] not in ("subtract", "zero"):
            raise MLIRUnsupported(
                f"neuron {node.attrs.get('model')!r} resets via "
                f"{lif['reset_mode']!r}; this backend implements 'subtract' and "
                "'zero', the two resets hardware LIF blocks expose."
            )
        return lif

    # ------------------------------------------------------------------ emit
    def emit(self) -> str:
        self.plan()
        self.check()
        for name, _kind, _shape, _t in self.params:
            self.buf[name] = "%" + name
        for node in self.g.nodes:
            getattr(self, f"_decl_{node.op}")(node)
        for node in self.g.nodes:
            step = getattr(self, f"_step_{node.op}", None)
            if step is not None:
                step(node)

        c0, c1 = self.c_index(0), self.c_index(self.T)
        lines = list(self.decls) + list(self.pre)
        if self.step:
            lines.append("  " * _BODY + f"scf.for %t = {c0} to {c1} step "
                                        f"{self.c_index(1)} {{")
            lines += self.step
            lines.append("  " * _BODY + "}")

        sig = ", ".join(f"%{n}: {t}" for n, _k, _s, t in self.params)
        return (
            self._header()
            + f"module {{\n  func.func @{_safe(self.g.name)}({sig}) {{\n"
            + "\n".join(lines)
            + "\n    func.return\n  }\n}\n"
        )

    def _header(self) -> str:
        rows = [
            "// Lowered from a .snn program by snndsl (MLIR target).",
            "//",
            f"// module: {self.g.name}   steps: {self.T}   dt: {self.dt:g}",
            "// dialects: builtin, func, memref, arith, scf -- no custom dialect,",
            "// so mlir-opt and xdsl both read this without a plugin.",
            "//",
            "// arguments, in order (this order is the ABI; memrefs carry no names):",
        ]
        for name, kind, shape, _t in self.params:
            rows.append(f"//   %{name:<20} {kind:<7} {list(shape)}")
        rows += [
            "//",
            "// Every tensor is flat, so a rank change is a no-op. Spikes are 0.0/1.0",
            "// f32 at [t * units + unit] -- the next dense layer consumes them as",
            "// activations, so the graph has exactly one element type. The state",
            "// argument holds the potential after the last step.",
        ]
        return "\n".join(rows) + "\n"

    # ---- declarations ----------------------------------------------------
    def _decl_input(self, node: IRNode) -> None:
        self.buf[node.id] = self.buf[_safe(node.id) + "_in"]

    def _decl_output(self, node: IRNode) -> None:
        self.buf[node.id] = self.buf[node.inputs[0]]

    def _decl_flatten(self, node: IRNode) -> None:
        self.buf[node.id] = self.buf[node.inputs[0]]

    def _decl_encode(self, node: IRNode) -> None:
        src = self.buf[node.inputs[0]]
        n = _prod(node.shape)
        gain = float(node.attrs.get("gain", 1.0))
        out = self.alloc(n, "enc")
        c0, c1, cn = self.c_index(0), self.c_index(1), self.c_index(n)
        cg = self.c_f32(gain)
        # rate coding is a constant current: same every step, so compute once
        e, ex, eg = self.fresh("e"), self.fresh("ex"), self.fresh("eg")
        self._pre(_BODY, f"scf.for {e} = {c0} to {cn} step {c1} {{")
        self._pre(_BODY + 1, f"{ex} = memref.load {src}[{e}] : memref<{n}xf32>")
        self._pre(_BODY + 1, f"{eg} = arith.mulf {ex}, {cg} : f32")
        self._pre(_BODY + 1, f"memref.store {eg}, {out}[{e}] : memref<{n}xf32>")
        self._pre(_BODY, "}")
        self.buf[node.id] = out

    def _decl_dense(self, node: IRNode) -> None:
        self.buf[node.id] = self.alloc(int(node.attrs["units"]), "den")

    def _decl_neuron(self, node: IRNode) -> None:
        n = _prod(node.shape)
        # Two spike buffers on purpose:
        #   act -- this step's spikes, overwritten each step. The next layer
        #          reads this, because a layer consumes the spikes of step t,
        #          not the whole trace.
        #   s   -- the trace, [t * n + unit], an output argument. This is what
        #          a conformance run compares position by position; collapsing
        #          the two would make the numbers agree while hiding a
        #          misaligned step axis.
        self.buf[f"act_{_safe(node.id)}"] = self.alloc(n, "act")
        self.buf[node.id] = self.buf[f"act_{_safe(node.id)}"]
        v = self.buf[f"v_{_safe(node.id)}"]
        c0, c1, cn = self.c_index(0), self.c_index(1), self.c_index(n)
        z = self.fresh("z")
        self._pre(_BODY, f"scf.for {z} = {c0} to {cn} step {c1} {{")
        self._pre(_BODY + 1, f"memref.store {self.c_f32(0.0)}, {v}[{z}] : "
                             f"memref<{n}xf32>")
        self._pre(_BODY, "}")

    # ---- per-step ops ----------------------------------------------------
    def _step_dense(self, node: IRNode) -> None:
        src = self.buf[node.inputs[0]]
        n_in = int(node.attrs["in_features"])
        n_out = int(node.attrs["units"])
        w = self.buf[f"w_{_safe(node.id)}"]
        out = self.buf[node.id]
        c0, c1 = self.c_index(0), self.c_index(1)
        cn_in, cn_out = self.c_index(n_in), self.c_index(n_out)
        o, i = self.fresh("o"), self.fresh("i")
        acc, b0, a0 = self.fresh("acc"), self.fresh("b"), self.fresh("a")
        self._body(_LAYER, f"scf.for {o} = {c0} to {cn_out} step {c1} {{")
        if node.attrs.get("bias", True):
            b = self.buf[f"b_{_safe(node.id)}"]
            self._body(_INNER, f"{b0} = memref.load {b}[{o}] : memref<{n_out}xf32>")
            init = b0
        else:
            init = self.c_f32(0.0)
        x, wv, off, idx = (self.fresh("x"), self.fresh("w"),
                           self.fresh("off"), self.fresh("idx"))
        self._body(_INNER, f"{acc} = scf.for {i} = {c0} to {cn_in} step {c1} "
                           f"iter_args({a0} = {init}) -> (f32) {{")
        self._body(_INNER2, f"{x} = memref.load {src}[{i}] : memref<{n_in}xf32>")
        self._body(_INNER2, f"{off} = arith.muli {o}, {cn_in} : index")
        self._body(_INNER2, f"{idx} = arith.addi {off}, {i} : index")
        self._body(_INNER2, f"{wv} = memref.load {w}[{idx}] : "
                            f"memref<{n_out * n_in}xf32>")
        pr, sm = self.fresh("mul"), self.fresh("sum")
        self._body(_INNER2, f"{pr} = arith.mulf {x}, {wv} : f32")
        self._body(_INNER2, f"{sm} = arith.addf {a0}, {pr} : f32")
        self._body(_INNER2, f"scf.yield {sm} : f32")
        self._body(_INNER, "}")
        self._body(_INNER, f"memref.store {acc}, {out}[{o}] : memref<{n_out}xf32>")
        self._body(_LAYER, "}")

    def _step_neuron(self, node: IRNode) -> None:
        lif = self.lif[node.id]
        beta = 1.0 - self.dt / lif["tau"]
        gain = lif["r"] * self.dt / lif["tau"]
        leak = lif["v_leak"] * self.dt / lif["tau"]
        v_th = float(lif["v_threshold"])

        src = self.buf[node.inputs[0]]
        n = _prod(node.shape)
        v = self.buf[f"v_{_safe(node.id)}"]
        s = self.buf[f"s_{_safe(node.id)}"]
        act = self.buf[f"act_{_safe(node.id)}"]

        c0, c1 = self.c_index(0), self.c_index(1)
        cn = self.c_index(n)
        c_beta, c_gain, c_th = self.c_f32(beta), self.c_f32(gain), self.c_f32(v_th)
        c_leak, c_zero = self.c_f32(leak), self.c_f32(0.0)
        c_one = self.c_f32(1.0)

        u = self.fresh("u")
        nv, dec, ni, drv, cur = (self.fresh("nv"), self.fresh("dec"),
                                 self.fresh("ni"), self.fresh("drv"),
                                 self.fresh("cur"))
        self._body(_LAYER, f"scf.for {u} = {c0} to {cn} step {c1} {{")
        self._body(_INNER, f"{nv} = memref.load {v}[{u}] : memref<{n}xf32>")
        self._body(_INNER, f"{dec} = arith.mulf {nv}, {c_beta} : f32")
        self._body(_INNER, f"{ni} = memref.load {src}[{u}] : memref<{n}xf32>")
        self._body(_INNER, f"{drv} = arith.mulf {ni}, {c_gain} : f32")
        self._body(_INNER, f"{cur} = arith.addf {dec}, {drv} : f32")
        acc = cur
        if leak != 0.0:
            lk = self.fresh("lk")
            self._body(_INNER, f"{lk} = arith.addf {acc}, {c_leak} : f32")
            acc = lk
        sp, base, sid, inc = (self.fresh("sp"), self.fresh("base"),
                              self.fresh("sid"), self.fresh("inc"))
        self._body(_INNER, f"{sp} = arith.cmpf ogt, {acc}, {c_th} : f32")
        self._body(_INNER, f"{base} = arith.muli %t, {cn} : index")
        self._body(_INNER, f"{sid} = arith.addi {base}, {u} : index")
        self._body(_INNER, f"{inc} = arith.select {sp}, {c_one}, {c_zero} : f32")
        self._body(_INNER, f"memref.store {inc}, {s}[{sid}] : "
                           f"memref<{self.T * n}xf32>")
        self._body(_INNER, f"memref.store {inc}, {act}[{u}] : memref<{n}xf32>")
        if lif["reset_mode"] == "zero":
            rst = self.fresh("rst")
            self._body(_INNER, f"{rst} = arith.select {sp}, {c_zero}, {acc} : f32")
        else:
            sub, rst = self.fresh("sub"), self.fresh("rst")
            self._body(_INNER, f"{sub} = arith.subf {acc}, {c_th} : f32")
            self._body(_INNER, f"{rst} = arith.select {sp}, {sub}, {acc} : f32")
        self._body(_INNER, f"memref.store {rst}, {v}[{u}] : memref<{n}xf32>")
        self._body(_LAYER, "}")


def signature(graph: IRGraph, steps: Optional[int] = None) -> List[Dict[str, Any]]:
    """The generated function's argument list, in order.

    Callers need this to build buffers: nothing about a memref's meaning
    survives into MLIR, so the order *is* the ABI.
    """
    em = _Emitter(graph, steps)
    em.plan()
    return [{"name": name, "kind": kind, "node": em.owner.get(name, ""),
             "shape": list(shape)}
            for name, kind, shape, _t in em.params]


def lower(graph: IRGraph, steps: Optional[int] = None) -> str:
    """Return MLIR text for ``graph``. Raises :class:`MLIRUnsupported`."""
    return _Emitter(graph, steps).emit()


def export_mlir(graph: IRGraph, path: str, steps: Optional[int] = None) -> str:
    text = lower(graph, steps)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
