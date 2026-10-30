"""Compiler regression tests.

Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from snndsl import parser as snn_parser
from snndsl.errors import ParseError, SemaError, SNNError
from snndsl.passes import live_peak_bytes
from snndsl.pipeline import compile_source, load_module

HAS_TORCH = importlib.util.find_spec("torch") is not None

MINIMAL = """
program tiny

temporal:
    steps = 4
    dt = 1.0

neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v - s * v_th
    surrogate = fast_sigmoid(alpha=4.0)

pipeline:
    input  x  : 1, 8, 8
    dense  h  : 16 <- x
    neuron h  -> LIF
    dense  y  : 4  <- h
    neuron y  -> LIF
    output out <- y
"""


class TestFrontend(unittest.TestCase):
    def test_parse_program_name(self):
        unit = snn_parser.parse(MINIMAL)
        self.assertEqual(unit.first("program").name, "tiny")

    def test_const_folding(self):
        src = MINIMAL.replace("program tiny", "program tiny\nconst K = 2 * 3 + 1")
        src = src.replace("dense  h  : 16", "dense  h  : K + 9")
        g = compile_source(src).graph
        h = next(n for n in g.nodes if n.name == "h")
        self.assertEqual(h.shape, (16,))

    def test_error_has_line_number(self):
        bad = MINIMAL.replace("dense  y  : 4  <- h", "dense  y  : 4  <- nope")
        with self.assertRaises(SemaError) as ctx:
            compile_source(bad)
        self.assertIn("nope", str(ctx.exception))
        self.assertIsNotNone(ctx.exception.diagnostic.line)

    def test_parse_error_is_located(self):
        bad = "program p\ntemporal:\n  steps 4\n"
        with self.assertRaises(ParseError) as ctx:
            snn_parser.parse(bad)
        self.assertLessEqual(ctx.exception.diagnostic.line, 3)

    def test_unknown_neuron_model(self):
        bad = MINIMAL.replace("neuron h  -> LIF", "neuron h  -> XYZ")
        with self.assertRaises(SemaError) as ctx:
            compile_source(bad)
        self.assertIn("XYZ", str(ctx.exception))


class TestIr(unittest.TestCase):
    def setUp(self):
        self.g = compile_source(MINIMAL).graph

    def test_topological_shapes(self):
        shapes = {n.name: n.shape for n in self.g.nodes if n.op in ("dense", "neuron")}
        self.assertEqual(shapes["h"], (16,))
        self.assertEqual(shapes["y"], (4,))

    def test_rebinding_creates_distinct_values(self):
        # 'h' is bound twice (dense, then neuron) -> two IR values, one DSL name
        ids = [n.id for n in self.g.nodes if n.name == "h"]
        self.assertEqual(len(ids), 2)
        self.assertEqual(len(set(ids)), 2)

    def test_json_roundtrip(self):
        import json
        d = json.loads(self.g.to_json())
        self.assertEqual(len(d["nodes"]), len(self.g.nodes))
        self.assertEqual(d["stats"]["params"], self.g.stats()["params"])

    def test_params_count(self):
        # 64*16+16 + 16*4+4
        self.assertEqual(self.g.stats()["params"], 64 * 16 + 16 + 16 * 4 + 4)


class TestPasses(unittest.TestCase):
    def test_dce_drops_unused_branch(self):
        src = MINIMAL.replace(
            "    output out <- y",
            "    dense  dead : 32 <- h\n    output out <- y")
        g = compile_source(src).graph
        self.assertFalse(any(n.name == "dead" for n in g.nodes))

    def test_memplan_reuses_buffers(self):
        g = compile_source(MINIMAL).graph
        stats = live_peak_bytes(g)
        self.assertLessEqual(stats["planned_bytes"], stats["no_reuse_bytes"])
        self.assertGreaterEqual(stats["slots"], 1)

    def test_fuse_flatten_into_dense(self):
        src = MINIMAL.replace(
            "    dense  h  : 16 <- x",
            "    conv   c  : 4 <- x with k=3, p=1\n"
            "    flatten f <- c\n"
            "    dense  h  : 16 <- f")
        g = compile_source(src).graph
        self.assertFalse(any(n.op == "flatten" for n in g.nodes))

    def test_schedule_unrolls_small_t(self):
        src = MINIMAL.replace("steps = 4", "steps = 2")
        r = compile_source(src)
        self.assertEqual(r.graph.temporal.strategy, "unrolled")
        self.assertIn("time step 1", r.module_source)

    def test_direct_input_pass_opt_in(self):
        src = MINIMAL.replace("    dense  h  : 16 <- x",
                              "    encode x -> rate\n    dense  h  : 16 <- x")
        plain = compile_source(src).graph
        fast = compile_source(src, passes=["direct_input"]).graph
        self.assertTrue(any(n.op == "encode" for n in plain.nodes))
        self.assertFalse(any(n.op == "encode" for n in fast.nodes))


class TestCodegen(unittest.TestCase):
    @unittest.skipUnless(HAS_TORCH, "executing the generated module needs torch")
    def test_module_runs_and_backprops(self):
        import torch
        r = compile_source(MINIMAL)
        Cls = load_module(r, os.path.join(tempfile.mkdtemp(), "m.py"))
        m = Cls()
        x = torch.rand(2, 1, 8, 8)
        y = m(x)
        self.assertEqual(tuple(y.shape), (2, 4))
        y.sum().backward()
        grads = [p.grad for p in m.parameters()]
        self.assertTrue(all(g is not None for g in grads))

    def test_surrogate_lowering(self):
        r = compile_source(MINIMAL)
        self.assertIn("surrogate_fast_sigmoid", r.module_source)
        self.assertNotIn("> v_th", r.module_source)

    @unittest.skipUnless(HAS_TORCH, "executing the generated module needs torch")
    def test_state_reset_between_calls(self):
        import torch
        r = compile_source(MINIMAL)
        Cls = load_module(r, os.path.join(tempfile.mkdtemp(), "m.py"))
        m = Cls()
        x = torch.rand(2, 1, 8, 8)
        a = m(x)
        b = m(x)
        self.assertTrue(torch.allclose(a, b), "membrane state leaked across calls")


class TestBackends(unittest.TestCase):
    def test_nir_export(self):
        try:
            import nir  # noqa: F401
        except Exception:
            self.skipTest("nir not installed")
        import nir as nirlib
        from snndsl.backends.nir_export import export_nir
        g = compile_source(MINIMAL).graph
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "g.nir")
            rep = export_nir(g, path)
            self.assertGreater(rep["node_count"], 0)
            back = nirlib.read(path)
            self.assertEqual(len(back.edges), len(rep["edges"]))

    @unittest.skipUnless(HAS_TORCH, "executing the generated module needs torch")
    def test_energy_model_monotone(self):
        from snndsl.runner import energy_report
        g = compile_source(MINIMAL).graph
        dense = energy_report(g, {n.id: 1.0 for n in g.nodes})
        sparse = energy_report(g, {n.id: 0.1 for n in g.nodes})
        self.assertLess(sparse["snn_energy_nj"], dense["snn_energy_nj"])


class TestSnnSpecificChecks(unittest.TestCase):
    """Guards against silently compiling something that is not a spiking net."""

    NO_NEURON = """
program p
temporal:
    steps = 4
pipeline:
    input  x : 8
    dense  h : 4 <- x
    output out <- h
"""

    def test_ann_without_spikes_is_rejected(self):
        with self.assertRaises(SNNError) as ctx:
            compile_source(self.NO_NEURON, filename="ann.snn")
        msg = ctx.exception.render(self.NO_NEURON, "ann.snn")
        self.assertIn("no spiking neuron", msg)
        self.assertIn("neuron h -> LIF", msg)     # the hint must be actionable

    def test_missing_temporal_warns(self):
        src = """
program p
neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
pipeline:
    input  x : 8
    dense  h : 4 <- x
    neuron h -> LIF
    output out <- h
"""
        res = compile_source(src, filename="t.snn", quiet=True)
        self.assertTrue(any("no 'temporal:' block" in w for w in res.warnings))

    def test_recurrent_connection_gets_a_real_explanation(self):
        src = """
program p
temporal:
    steps = 4
neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
pipeline:
    input  x : 8
    dense  h : 4 <- x
    neuron h -> LIF
    dense  g : 4 <- h, g
    output out <- g
"""
        with self.assertRaises(SNNError) as ctx:
            compile_source(src, filename="r.snn")
        msg = ctx.exception.render(src, "r.snn")
        self.assertIn("recurrent", msg)
        self.assertIn("state", msg)      # points at the workaround


class TestReadout(unittest.TestCase):
    """The two readout modes must produce genuinely different tensors."""

    SRC = """
program ro
temporal:
    steps = 4
neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v - s * v_th
    surrogate = fast_sigmoid(alpha=4.0)
pipeline:
    input  x : 8
    encode x -> rate
    dense  y : 4 <- x with scale=2.0
    neuron y -> LIF
    output out <- y
infer:
    readout = {mode}
"""

    def _build(self, mode):
        import torch
        from snndsl.pipeline import load_module
        res = compile_source(self.SRC.format(mode=mode), filename="ro.snn")
        with tempfile.TemporaryDirectory() as d:
            Cls = load_module(res, os.path.join(d, "m.py"))
            net = Cls()
            net.collect = True
            torch.manual_seed(0)
            x = torch.rand(6, 8)
            return net, net(x)

    @unittest.skipUnless(HAS_TORCH, "executing the generated module needs torch")
    def test_membrane_readout_is_not_one_hot(self):
        _net, out = self._build("membrane")
        self.assertEqual(tuple(out.shape), (6, 4))
        # every row must have more than one non-zero logit, i.e. real margins
        nonzero = (out.abs() > 1e-9).sum(dim=1)
        self.assertTrue(bool((nonzero > 1).all()), f"degenerate logits: {out}")

    @unittest.skipUnless(HAS_TORCH, "executing the generated module needs torch")
    def test_spike_and_membrane_differ(self):
        import torch
        _, a = self._build("spikes")
        _, b = self._build("membrane")
        self.assertFalse(torch.allclose(a, b), "readout mode had no effect")

    def test_unknown_readout_warns_but_compiles(self):
        # an unknown mode must not crash the build; it falls through to the
        # spike path and the value is preserved for the report to flag
        res = compile_source(self.SRC.format(mode="banana"), filename="ro.snn")
        self.assertIn("banana", str(res.graph.infer.get("readout")))


class TestPythonFrontend(unittest.TestCase):
    """Both front ends must reach the same IR, or the abstraction is a lie."""

    TEXT = """
program same
temporal:
    steps = 8
neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v - s * v_th
    surrogate = fast_sigmoid(alpha=4.0)
pipeline:
    input  x : 1, 28, 28
    encode x -> rate with gain=1.0
    dense  h : 64 <- x with scale=2.0
    neuron h -> LIF
    dense  y : 10 <- h
    neuron y -> LIF
    output out <- y
infer:
    readout = membrane
"""

    def _python(self):
        import snndsl as snn
        LIF = snn.LIF(tau=2.0, v_th=1.0)
        net = snn.Net("same", steps=8, readout="membrane")
        x = net.encode(net.input(1, 28, 28), "rate", gain=1.0)
        h = net.neuron(net.dense(x, 64, scale=2.0), LIF)
        y = net.neuron(net.dense(h, 10), LIF)
        net.output(y)
        return net

    def test_same_ir_as_text_frontend(self):
        from_text = compile_source(self.TEXT, filename="same.snn", quiet=True)
        from_py = self._python().compile(quiet=True)
        a = [(n.op, tuple(n.shape)) for n in from_text.graph.nodes]
        b = [(n.op, tuple(n.shape)) for n in from_py.graph.nodes]
        self.assertEqual(a, b)
        self.assertEqual(from_text.graph.stats()["params"],
                         from_py.graph.stats()["params"])
        self.assertEqual(len(from_text.module_source.splitlines()),
                         len(from_py.module_source.splitlines()))

    def test_python_loop_unrolls_blocks(self):
        import snndsl as snn
        LIF = snn.LIF()
        net = snn.Net("loop", steps=4)
        h = net.input(16)
        for _ in range(5):
            h = net.neuron(net.dense(h, 8), LIF)
        net.output(h)
        g = net.compile(quiet=True).graph
        self.assertEqual(sum(1 for n in g.nodes if n.op == "dense"), 5)
        self.assertEqual(sum(1 for n in g.nodes if n.op == "neuron"), 5)

    def test_cli_compiles_a_python_file_like_a_snn_file(self):
        """`snndsl inspect/build/deploy net.py` must work, not just net.snn.

        Two frontends, one compiler is a claim about the toolchain, not about
        the API: if the CLI only accepts .snn, the second frontend is a library
        feature and every downstream command has to be justified twice.
        """
        import tempfile

        from snndsl.pipeline import compile_file

        src = (
            "import snndsl as snn\n"
            "def build():\n"
            "    net = snn.Net('smoke', steps=4, readout='membrane')\n"
            "    LIF = snn.LIF(tau=2.0, v_th=1.0)\n"
            "    x = net.encode(net.input(1, 28, 28), 'rate', gain=1.0)\n"
            "    net.output(net.neuron(net.dense(x, 8), LIF))\n"
            "    return net\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "net.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(src)
            res = compile_file(path, quiet=True)
        self.assertEqual([n.op for n in res.graph.nodes][:2], ["input", "encode"])
        self.assertIn("class SnnNet", res.module_source)

    def test_python_file_without_a_network_says_so(self):
        import tempfile

        from snndsl.pipeline import compile_file

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "not_a_net.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("VALUE = 3\n")
            with self.assertRaises(SNNError) as ctx:
                compile_file(path, quiet=True)
        self.assertIn("exposes no network", str(ctx.exception))

    def test_python_frontend_rejects_ann(self):
        import snndsl as snn
        net = snn.Net("ann", steps=4)
        net.output(net.dense(net.input(16), 4))
        with self.assertRaises(SNNError) as ctx:
            net.compile(quiet=True)
        self.assertIn("no spiking neuron", str(ctx.exception))

    def test_custom_neuron_equations(self):
        import snndsl as snn
        net = snn.Net("alif", steps=4)
        alif = snn.ALIF(tau=2.0, beta=0.3)
        x = net.neuron(net.dense(net.input(16), 8), alif)
        net.output(net.neuron(net.dense(x, 3), snn.LIF()))
        g = net.compile(quiet=True).graph
        spec = g.neurons["ALIF"]
        self.assertEqual(set(spec.states), {"v", "a"})     # two state variables
        self.assertIn("v_th", spec.body[2][2])             # adaptive threshold


class TestDiagnostics(unittest.TestCase):
    def test_render_includes_caret(self):
        src = "program p\npipeline:\n    dense h : 3 <- ghost\n"
        try:
            compile_source(src)
        except SNNError as err:
            text = err.render(src, "t.snn")
            self.assertIn("t.snn:", text)
            self.assertIn("ghost", text)
        else:
            self.fail("expected an error")


class TestDuplicateSpike(unittest.TestCase):
    """Two spike lines with the same target used to be accepted in silence.

    They lower to two assignments of the same variable in ``forward``, so the
    first predicate is dead. Found while building a doc example: the generated
    code contained ``s = surrogate_(v - v_th)`` immediately followed by
    ``s = surrogate_(v - v_th * 0.5)``, and nothing anywhere said so.
    """

    SRC = """program dup

temporal:
    steps = 4
    dt = 1.0
    reset = zero

neuron BAD:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    spike s = v > v_th * 0.5
    v = v * (1 - s)

pipeline:
    input  x : 1, 4, 4
    encode x -> rate with gain=1.0
    dense  h : 8 <- x
    neuron h -> BAD
    output h
"""

    def test_second_spike_line_is_an_error(self):
        with self.assertRaises(SemaError) as ctx:
            compile_source(self.SRC)
        msg = str(ctx.exception)
        self.assertIn("twice", msg)
        self.assertIn("s", msg)

    def test_the_error_points_at_the_offending_line(self):
        with self.assertRaises(SemaError) as ctx:
            compile_source(self.SRC)
        diag = getattr(ctx.exception, "diagnostic", None)
        self.assertIsNotNone(diag)
        self.assertEqual(diag.line, 14)          # the second `spike s`


class TestNotAnSnndslNet(unittest.TestCase):
    """A file exposing an ordinary ``nn.Module`` must be rejected, not half-read.

    This is the trap that matters most for someone coming from an ANN
    background: they write ``class SNN(nn.Module)``, name the instance ``net``,
    and expect it to work. ``torch.nn.Module`` has had a ``compile`` method
    since 2.0, so a duck-typed ``hasattr(net, "compile")`` check accepted it and
    then called ``torch.compile`` instead -- with kwargs it raised a TypeError
    about ``passes``, without kwargs it **returned None silently**.
    """

    SOURCE = """
import torch
import torch.nn as nn


class LIF(nn.Module):
    def __init__(self, tau=2.0, v_th=1.0):
        super().__init__()
        self.tau, self.v_th = tau, v_th

    def forward(self, x):
        v = 0.0
        for _ in range(4):
            v = v * (1 - 1.0 / self.tau) + x
            s = (v > self.v_th).float()
            v = v * (1 - s)
        return s


class SNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 4)
        self.a = LIF()

    def forward(self, x):
        return self.a(self.fc(x))


net = SNN()
"""

    def _write(self, tmpdir):
        path = os.path.join(tmpdir, "std_snn.py")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.SOURCE)
        return path

    def test_nn_module_is_rejected_by_name(self):
        from snndsl.pipeline import compile_python_file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir)
            with self.assertRaises(SNNError) as ctx:
                compile_python_file(path)
        msg = str(ctx.exception)
        self.assertIn("SNN", msg)                 # the class it found
        self.assertIn("snndsl.Net", msg)          # the type it wanted

    def test_never_returns_none(self):
        """The old bug: no kwargs -> torch.compile -> None, no error at all."""
        from snndsl.pipeline import compile_python_file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir)
            try:
                result = compile_python_file(path)
            except SNNError:
                return                            # correct: refused
            self.fail(f"silently returned {result!r} instead of refusing")

    def test_error_is_not_a_torch_type_error(self):
        from snndsl.pipeline import compile_python_file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write(tmpdir)
            try:
                compile_python_file(path, passes=None)
            except TypeError as exc:
                self.fail(f"leaked torch's TypeError: {exc}")
            except SNNError:
                pass


class TestUnsupportedOp(unittest.TestCase):
    """An op the grammar accepts but the backend never lowered must say so.

    ``decode`` is in the lexer, in ``sema`` and in ``ir.IO_OPS``, so a program
    using it parses and type-checks fine. ``torch_codegen`` has no
    ``_op_decode``. Before this test that combination died inside codegen with
    ``AttributeError: 'TorchCodeGen' object has no attribute '_op_decode'`` --
    a stack trace pointing at our internals, from a file that looked legal.
    """

    DECODE = """
program dec

temporal:
    steps = 4
    dt = 1.0
    reset = zero

neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v * (1 - s)
    surrogate = fast_sigmoid(alpha=4.0)

pipeline:
    input  x : 1, 4, 4
    encode x -> rate with gain=1.0
    dense  h : 8 <- x with scale=2.0
    neuron h -> LIF
    decode h -> rate with window=4
    output h
"""

    def test_decode_reports_missing_lowering(self):
        with self.assertRaises(SNNError) as ctx:
            compile_source(self.DECODE)
        msg = str(ctx.exception)
        self.assertIn("decode", msg)
        self.assertIn("no PyTorch lowering", msg)

    def test_error_is_not_an_attribute_error(self):
        """The whole point: a readable SNNError, never a bare AttributeError."""
        try:
            compile_source(self.DECODE)
        except AttributeError:
            self.fail("codegen leaked AttributeError instead of SNNError")
        except SNNError:
            pass


if __name__ == "__main__":
    unittest.main()
