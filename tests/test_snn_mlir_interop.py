"""Interop with SNN-MLIR: our ``.nir`` in, their MLIR out.

The interesting claim here is not "your file parses". It is the split between
two questions that look alike:

* ``accepts`` — will the backend convert the file
* ``faithful`` — does converting it mean running *this* network

``tests/test_export_integrity.py`` already covers the file surviving our own
round-trip. This file covers the next hop: a third-party frontend, its own
checker, its own lowering, its own C driver. Where a test needs that frontend it
is marked ``needs_cli`` and skips without it — the assertions are about what the
backend really does, so a fake would prove nothing.

The one test that matters most is
``test_preflight_agrees_with_backend_on_every_example``: a pre-flight nobody
cross-checks is just a second opinion. Sweeping every ``examples/*.snn`` against
the backend's exit code keeps the rules honest as the backend moves.
"""
from __future__ import annotations

import glob
import json
import os

import pytest
import snndsl as snn
from snndsl.backends import snn_mlir
from snndsl.backends.nir_export import export_nir, recognise_lif
from snndsl.pipeline import compile_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES = os.path.join(ROOT, "examples")

needs_cli = pytest.mark.skipif(
    snn_mlir.cli_path() is None,
    reason="the snn-mlir CLI is not installed (pip install snn-mlir); these "
           "assertions are about what the backend does, not about our own code")


def _examples() -> list:
    return sorted(glob.glob(os.path.join(EXAMPLES, "*.snn")))


def _gain2_graph():
    """A neuron that integrates twice its input: legal IR, exports ``r = 2*tau/dt``."""
    gain2 = snn.custom("Gain2", params={"tau": 2.0, "v_th": 1.0}, state={"v": 0.0},
                       body=["v = v * (1 - dt / tau) + 2 * i",
                             "spike s = v > v_th",
                             "v = v * (1 - s)"])
    net = snn.Net("gain2", steps=8, dt=1.0, reset="zero")
    x = net.encode(net.input(1, 28, 28), "rate", 1.0)
    h = net.neuron(net.dense(x, 32, name="h1"), gain2)
    net.output(net.neuron(net.dense(h, 10, name="y"), gain2))
    return net.compile().graph


# ── the file we hand over ────────────────────────────────────────────────────

def test_export_renames_terminals_and_keeps_the_originals(tmp_path):
    graph = compile_file(os.path.join(EXAMPLES, "mnist_hw.snn")).graph
    out = str(tmp_path / "m.nir")
    rep = export_nir(graph, out, target="snn-mlir")

    assert rep["target"] == "snn-mlir"
    assert rep["terminals"]["input"]["emitted"] == "input"
    assert rep["terminals"]["output"]["emitted"] == "output"
    # the original names are still recoverable from the report, not overwritten
    assert rep["terminals"]["input"]["original"] == "x"
    assert rep["terminals"]["output"]["original"] == "out"


def test_plain_export_keeps_the_program_s_own_names(tmp_path):
    """The rename is a target-specific concession, not the new default."""
    graph = compile_file(os.path.join(EXAMPLES, "mnist_hw.snn")).graph
    out = str(tmp_path / "m.nir")
    rep = export_nir(graph, out)
    assert rep["terminals"] is None
    import nir
    names = set(nir.read(out).nodes)
    assert {"x", "out"} <= names
    assert not ({"input", "output"} & names)


def test_rename_survives_in_the_file_s_metadata(tmp_path):
    """A reader that thinks in the source program should not have to diff names."""
    import nir
    graph = compile_file(os.path.join(EXAMPLES, "mnist_hw.snn")).graph
    out = str(tmp_path / "m.nir")
    export_nir(graph, out, target="snn-mlir")
    meta = nir.read(out).metadata
    assert meta["snndsl.target"] == "snn-mlir"
    assert json.loads(meta["snndsl.terminals"])["input"]["original"] == "x"


def test_a_node_already_named_input_is_a_loud_error(tmp_path):
    """Silently renaming over a user's node would lose a layer, not a name."""
    net = snn.Net("clash", steps=4)
    x = net.input(1, 8)
    net.output(net.neuron(net.dense(x, 4, name="input"), snn.LIF()))
    graph = net.compile().graph
    with pytest.raises(snn.SNNError, match="already uses that name"):
        export_nir(graph, str(tmp_path / "c.nir"), target="snn-mlir")


def test_unknown_target_is_refused(tmp_path):
    graph = compile_file(os.path.join(EXAMPLES, "mnist_hw.snn")).graph
    with pytest.raises(snn.SNNError, match="unknown export target"):
        export_nir(graph, str(tmp_path / "m.nir"), target="lava")


# ── the pre-flight, checked against the backend itself ───────────────────────

@needs_cli
def test_preflight_agrees_with_backend_on_every_example(tmp_path):
    """If these disagree, one of the two is wrong and we would not know which."""
    rows = []
    for path in _examples():
        graph = compile_file(path).graph
        pf = snn_mlir.preflight(graph)
        out = str(tmp_path / (os.path.basename(path) + ".nir"))
        export_nir(graph, out, target="snn-mlir")
        backend = snn_mlir.check(out)
        rows.append((os.path.basename(path), pf["accepts"], backend["ok"]))
    assert len(rows) >= 8, rows
    disagree = [r for r in rows if r[1] != r[2]]
    assert not disagree, f"preflight vs backend mismatch: {disagree}"


@needs_cli
def test_backend_accepts_networks_it_should_not_call_equivalent(tmp_path):
    """``supported`` is not the same answer as ``faithful``. Three cases."""
    expected = {
        "adaptive_lif.snn": "neuron_downgraded_to_li",
        "cifar_hw.snn": "maxpool_as_avgpool",
        "cifar_cnn.snn": "maxpool_as_avgpool",
    }
    for name, rule in expected.items():
        graph = compile_file(os.path.join(EXAMPLES, name)).graph
        pf = snn_mlir.preflight(graph)
        out = str(tmp_path / (name + ".nir"))
        export_nir(graph, out, target="snn-mlir")
        backend = snn_mlir.check(out)

        assert backend["ok"], f"{name}: backend was expected to accept the file"
        assert pf["accepts"], f"{name}: preflight must not claim a refusal"
        assert not pf["faithful"], f"{name}: preflight must flag the loss"
        assert rule in {f["rule"] for f in pf["lossy"]}, pf["lossy"]


@needs_cli
def test_a_refusal_is_a_refusal_on_both_sides(tmp_path):
    graph = compile_file(os.path.join(EXAMPLES, "residual_snn.snn")).graph
    pf = snn_mlir.preflight(graph)
    out = str(tmp_path / "r.nir")
    export_nir(graph, out, target="snn-mlir")
    backend = snn_mlir.check(out)
    assert not pf["accepts"] and not backend["ok"]
    assert "no_merge_primitive" in {f["rule"] for f in pf["blockers"]}


def test_input_gain_rule_names_the_decay_the_backend_will_use():
    """``r`` carries the timestep, so a gain of 2 becomes a different neuron.

    SNN-MLIR recovers ``dt = tau/r`` then ``decay = 1 - dt/tau``
    (``snn_mlir/nodes/lif.py:89-90``). With ``r = 2*tau/dt`` that yields 0.75 for
    a neuron whose real decay is 0.5 — and the input is still fed at gain 1.
    """
    graph = _gain2_graph()
    neuron = next(n for n in graph.nodes if n.op == "neuron")
    lif = recognise_lif(graph, neuron)
    assert lif["r"] == pytest.approx(4.0)
    assert lif["tau"] == pytest.approx(2.0)
    assert 1 - 1 / lif["r"] == pytest.approx(0.75)
    assert 1 - graph.temporal.dt / lif["tau"] == pytest.approx(0.5)

    pf = snn_mlir.preflight(graph)
    assert pf["accepts"] and not pf["faithful"]
    detail = " ".join(f["detail"] for f in pf["lossy"])
    assert "0.75" in detail and "0.5" in detail


def test_preflight_needs_the_backend_to_say_nothing():
    """The analysis is ours; it must work on a machine without the CLI."""
    graph = compile_file(os.path.join(EXAMPLES, "residual_snn.snn")).graph
    pf = snn_mlir.preflight(graph)
    assert set(pf) == {"accepts", "faithful", "blockers", "lossy", "notes", "cli"}
    assert pf["blockers"] and pf["blockers"][0]["source"].startswith("snn_mlir/")


# ── what the backend does with the file ──────────────────────────────────────

@needs_cli
def test_lowered_mlir_keeps_time_outside_the_program(tmp_path):
    """One function, no loop: T is a parameter, not something baked into the IR."""
    graph = compile_file(os.path.join(EXAMPLES, "mnist_hw.snn")).graph
    out = str(tmp_path / "m.nir")
    export_nir(graph, out, target="snn-mlir")
    res = snn_mlir.to_mlir(out, str(tmp_path / "m.mlir"))
    assert res["ok"], res["stderr"]

    text = open(res["mlir"], encoding="utf-8").read()
    neurons = sum(1 for n in graph.nodes if n.op == "neuron")
    assert text.count("func.func") == 1
    assert "scf.for" not in text
    assert text.count("snn.lif") == neurons
    assert "snn_forward_step" in text
    # the state is a function argument, which is what makes T the caller's job
    assert "%voltage_" in text


@needs_cli
def test_codegen_takes_the_step_count_from_the_input_not_from_us(tmp_path):
    """The backend's C driver loops T times because our input said T, not because
    we told it: ``.nir`` has nowhere to carry a step count."""
    import numpy as np
    graph = compile_file(os.path.join(EXAMPLES, "mnist_hw.snn")).graph
    folder = tmp_path / "model"
    folder.mkdir()
    export_nir(graph, str(folder / "m.nir"), target="snn-mlir")
    entry = next(n for n in graph.nodes if n.op == "input")
    width = 1
    for d in entry.shape:
        width *= int(d)
    steps = int(graph.temporal.steps)
    np.save(str(folder / "input.npy"), np.ones((steps, width), np.int8))

    res = snn_mlir.codegen(str(folder))
    assert res["ok"], res["stderr"]
    header = open(os.path.join(res["build"], "snn_data.h"), encoding="utf-8").read()
    driver = open(os.path.join(res["build"], "main.c"), encoding="utf-8").read()
    assert f"#define N_STEPS      {steps}" in header
    assert any(f"step < {steps}" in ln and ln.strip().startswith("for (")
               for ln in driver.splitlines())


@needs_cli
def test_run_refuses_rather_than_pretending_when_the_toolchain_is_absent(tmp_path):
    """The last mile is not installable from PyPI, and that must not read as success."""
    graph = compile_file(os.path.join(EXAMPLES, "mnist_hw.snn")).graph
    folder = tmp_path / "model"
    folder.mkdir()
    export_nir(graph, str(folder / "m.nir"), target="snn-mlir")
    import numpy as np
    np.save(str(folder / "input.npy"),
            np.ones((int(graph.temporal.steps), 784), np.int8))
    res = snn_mlir.run(str(folder))
    if not res["ok"]:
        text = (res["stderr"] + res["stdout"]).lower()
        assert "toolchain" in text or "snn-opt" in text
