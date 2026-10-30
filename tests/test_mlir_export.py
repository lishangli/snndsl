"""The MLIR target, checked by two things that are not us.

Two separate claims, each needing its own witness:

* **the file is readable by the MLIR ecosystem** — xdsl parses it, and xdsl's
  verifier is what rejects a wrong operand type or a missing result. Nothing in
  this repo participates;
* **the file computes what PyTorch computes** — the xdsl interpreter runs it
  and every spike position is compared against a PyTorch forward pass.

A ``max|delta| = 0`` against this project's own reference interpreter would only
prove that the two halves of one mistake agree. Both witnesses above are
third-party runtimes, which is the only kind of agreement worth recording.
"""
from __future__ import annotations

import os
import re

import numpy as np
import pytest
from snndsl.backends import mlir_export
from snndsl.pipeline import compile_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")

TINY = """\
program tiny
const T = 4
temporal:
    steps = T
    dt = 1.0
    reset = subtract
neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v - s * v_th
    surrogate = fast_sigmoid(alpha=4.0)
encoder rate:
    gain = 1.0
pipeline:
    input x : 4
    encode x -> rate with gain=1.0
    dense h : 5 <- x with scale=1.0
    neuron h -> LIF
    dense y : 3 <- h with scale=1.0
    neuron y -> LIF
    output out <- y with readout=spikes
infer:
    readout = spikes
"""

DENSE_ONLY = ("quickstart", "mnist_mlp", "mnist_hw", "mnist_membrane")


def _tiny(tmp_path):
    p = tmp_path / "tiny.snn"
    p.write_text(TINY, encoding="utf-8")
    return compile_file(str(p)).graph


def _body_lines(text):
    """Code lines only, with their indent depth in two-space steps."""
    out = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("//"):
            continue
        out.append(((len(raw) - len(raw.lstrip())) // 2, raw.strip()))
    return out


# --------------------------------------------------------------- structure
def test_conv_is_refused_not_approximated():
    """conv needs index maps; a file that parses and computes the wrong thing
    is the exact failure this project exists to catch."""
    graph = compile_file(os.path.join(EXAMPLES, "cifar_hw_avg.snn")).graph
    with pytest.raises(mlir_export.MLIRUnsupported) as err:
        mlir_export.lower(graph)
    assert "conv" in str(err.value)


def test_one_shared_time_loop(tmp_path):
    """Two layers, one clock.

    A per-layer time loop is what falls out of writing the lowering top to
    bottom, and it is wrong in a way that no amount of numeric checking
    afterwards fixes: layer two would read layer one's spikes summed over all
    T steps instead of layer one's spikes at step t.
    """
    graph = _tiny(tmp_path)
    lines = _body_lines(mlir_export.lower(graph))
    loops = [(d, t) for d, t in lines if t.startswith("scf.for %t ")]
    assert len(loops) == 1, f"expected exactly one time loop, got {len(loops)}"
    t_depth = loops[0][0]
    inner = [d for d, t in lines if re.match(r"scf\.for %o\d+ =", t)]
    assert len(inner) == 2, "both dense layers must be inside the time loop"
    assert all(d > t_depth for d in inner)


def test_signature_is_the_abi(tmp_path):
    """Nothing about a memref's meaning survives into MLIR, so the argument
    order *is* the contract. If it drifts, every caller silently mis-binds."""
    graph = _tiny(tmp_path)
    text = mlir_export.lower(graph)
    sig = mlir_export.signature(graph)
    line = next(ln for ln in text.splitlines() if "func.func @" in ln)
    args = re.findall(r"%(\w+): memref<(\d+)(xf32)>", line)
    assert [a[0] for a in args] == [e["name"] for e in sig]
    assert [int(a[1]) for a in args] == [e["shape"][0] for e in sig]
    # every spike buffer is T * units, which is the trace length the checker
    # compares position by position
    T = graph.temporal.steps
    for e in sig:
        if e["kind"] == "spikes":
            units = int(np.prod(graph.by_id()[e["node"]].shape))
            assert e["shape"][0] == T * units


def test_states_and_spikes_are_both_outputs(tmp_path):
    """The membrane is observable too: a spike trace can match while the
    potential underneath has already diverged."""
    graph = _tiny(tmp_path)
    kinds = [e["kind"] for e in mlir_export.signature(graph)]
    assert kinds.count("state") == 2
    assert kinds.count("spikes") == 2



def _torch_reference(graph, weights, x):
    """Time-major, like the network.

    Layer-major is the natural way to write this and it is wrong past the first
    layer: layer two would be fed layer one's spikes from step T-1 on all T
    steps. The lowering gets this right by construction — one loop, layers
    inside it — so a layer-major reference would "disagree" with a correct file
    and, worse, agree with a wrong one.
    """
    import torch
    from snndsl.backends.nir_export import recognise_lif

    T, dt = graph.temporal.steps, graph.temporal.dt
    lifs = {n.id: recognise_lif(graph, n, dt=dt)
            for n in graph.nodes if n.op == "neuron"}
    with torch.no_grad():
        # rate coding is a constant current: it applies to the input once
        cur = torch.from_numpy(x)
        for node in graph.nodes:
            if node.op == "encode":
                cur = cur * float(node.attrs.get("gain", 1.0))
            elif node.op in ("dense", "neuron"):
                break
        cur0 = cur
        state = dict.fromkeys(lifs)
        traces = {k: [] for k in lifs}
        for _ in range(T):
            cur = cur0
            for node in graph.nodes:
                if node.op in ("encode", "flatten", "input", "output"):
                    continue
                if node.op == "dense":
                    cur = torch.nn.functional.linear(
                        cur,
                        torch.from_numpy(weights[f"w_{node.id}"]),
                        torch.from_numpy(weights[f"b_{node.id}"]),
                    )
                else:
                    lif = lifs[node.id]
                    beta = 1.0 - dt / lif["tau"]
                    gain = lif["r"] * dt / lif["tau"]
                    leak = lif["v_leak"] * dt / lif["tau"]
                    v_th = lif["v_threshold"]
                    v = state[node.id]
                    if v is None:
                        v = torch.zeros_like(cur)
                    v = v * beta + cur * gain + leak
                    s = (v > v_th).float()
                    state[node.id] = torch.where(s.bool(), v - v_th, v)
                    traces[node.id].append(s.clone())
                    cur = s
    return ({k: torch.stack(v).numpy().reshape(-1) for k, v in traces.items()},
            [float(torch.stack(v).mean()) for v in traces.values()])


# ------------------------------------------------------------- third party
def test_every_dense_example_parses_under_xdsl():
    """xdsl is the MLIR implementation SNAX builds its compiler on, so a file
    it rejects is a file that flow cannot consume either."""
    pytest.importorskip("xdsl", reason="the MLIR reader we are being judged by")
    from xdsl.context import Context
    from xdsl.dialects import arith, func, memref, scf
    from xdsl.dialects.builtin import Builtin
    from xdsl.parser import Parser

    ctx = Context()
    ctx.load_dialect(Builtin)
    for d in (arith.Arith, func.Func, scf.Scf, memref.MemRef):
        ctx.load_dialect(d)

    parsed = 0
    for name in DENSE_ONLY:
        path = os.path.join(EXAMPLES, f"{name}.snn")
        if not os.path.exists(path):
            continue
        text = mlir_export.lower(compile_file(path).graph)
        module = Parser(ctx, text).parse_module()
        assert len(list(module.walk())) > 10
        parsed += 1
    assert parsed >= 3, "examples moved; this test stopped checking anything"


def test_spikes_match_torch_position_by_position(tmp_path):
    """The load-bearing claim, with both ends third-party.

    PyTorch produces the reference trace; the xdsl interpreter executes our
    MLIR; this repo only carries the file between them.
    """
    pytest.importorskip("xdsl")
    pytest.importorskip("torch")
    from xdsl.context import Context
    from xdsl.dialects import arith, func, memref, scf
    from xdsl.dialects.builtin import Builtin
    from xdsl.dialects.func import FuncOp
    from xdsl.interpreter import Interpreter
    from xdsl.interpreters import arith as iarith
    from xdsl.interpreters import func as ifunc
    from xdsl.interpreters import memref as imemref
    from xdsl.interpreters import scf as iscf
    from xdsl.interpreters.shaped_array import ShapedArray
    from xdsl.interpreters.utils.ptr import TypedPtr, f32
    from xdsl.parser import Parser

    graph = _tiny(tmp_path)
    rng = np.random.default_rng(3)
    weights = {}
    for node in graph.nodes:
        if node.op == "dense":
            weights[f"w_{node.id}"] = rng.normal(
                0, 1.0, (node.attrs["units"], node.attrs["in_features"])
            ).astype(np.float32)
            weights[f"b_{node.id}"] = rng.normal(0, 0.1, (node.attrs["units"],)).astype(
                np.float32
            )
    x = rng.uniform(-1.0, 1.0, (1, 4)).astype(np.float32)

    want, rates = _torch_reference(graph, weights, x)

    # ---- our MLIR, executed by xdsl
    ctx = Context()
    ctx.load_dialect(Builtin)
    for d in (arith.Arith, func.Func, scf.Scf, memref.MemRef):
        ctx.load_dialect(d)
    module = Parser(ctx, mlir_export.lower(graph)).parse_module()
    interp = Interpreter(module)
    for f in (
        iarith.ArithFunctions,
        ifunc.FuncFunctions,
        iscf.ScfFunctions,
        imemref.MemRefFunctions,
    ):
        interp.register_implementations(f())
    fn = next(o for o in module.walk() if isinstance(o, FuncOp))

    holders, args = {}, []
    for entry in mlir_export.signature(graph):
        n = entry["shape"][0]
        if entry["kind"] == "weight":
            flat = weights[f"w_{entry['node']}"].reshape(-1)
        elif entry["kind"] == "bias":
            flat = weights[f"b_{entry['node']}"].reshape(-1)
        elif entry["kind"] == "input":
            flat = x.reshape(-1)
        else:
            flat = np.zeros(n, np.float32)
        sa = ShapedArray(TypedPtr.new(flat.tolist(), xtype=f32), [n])
        holders[entry["name"]] = sa
        args.append(sa)
    interp.call_op(fn, tuple(args))

    # A test that agrees on an all-zero trace proves nothing: both sides can be
    # wrong in the same direction. Pin the regime before comparing.
    assert all(0.1 < r < 0.9 for r in rates), f"degenerate spiking rates {rates}"

    for entry in mlir_export.signature(graph):
        if entry["kind"] != "spikes":
            continue
        got = np.array(
            [holders[entry["name"]].load((i,)) for i in range(entry["shape"][0])],
            np.float32,
        )
        ref = want[entry["node"]]
        assert got.size == ref.size
        assert int((got != ref).sum()) == 0, (
            f"{entry['node']}: {int((got != ref).sum())} of {got.size} spike "
            "positions differ between xdsl and PyTorch"
        )
