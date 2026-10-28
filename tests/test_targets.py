"""Tests for the parts that make a `.nir` file usable by *someone else's* code.

Two things are covered and they are different claims:

* the file loads on the whole nir 1.0.x line, not just the release we build with
* a foreign frontend either reproduces our spikes or says why it cannot

The second is deliberately allowed to fail: a vendor frontend that cannot read
nir 1.0 is a fact about the ecosystem, not a bug in this repository. The tests
assert that such a failure is *reported*, never that it is silent.
"""
import os
import tempfile

import numpy as np
import pytest
import torch
from snndsl.backends import targets as targets_mod
from snndsl.backends.nir_export import _downgrade_dialect, export_nir
from snndsl.pipeline import compile_file, load_module

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _compile(example):
    path = os.path.join(ROOT, "examples", example)
    r = compile_file(path)
    cls = load_module(r, os.path.join(tempfile.gettempdir(), "t_net.py"))
    net = cls()
    net.eval()
    w = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    return r, net, w


# ------------------------------------------------------------------- dialect
def test_zero_reset_file_drops_v_reset_so_older_nir_can_read_it(tmp_path):
    """v_reset was added in a later 1.0 patch; carrying it locks out 1.0.4.

    Omitting it is not a loss: nir's own reader substitutes zeros, which is what
    a zero reset is.
    """
    import h5py

    r, _net, w = _compile("mnist_hw.snn")
    out = str(tmp_path / "zero.nir")
    report = export_nir(r.graph, out, w, strict=True)

    assert report["dialect"]["dropped_fields"], "zero reset should drop v_reset"
    with h5py.File(out, "r") as fh:
        names = []

        def walk(g):
            for k in g:
                names.append(k)
                if isinstance(g[k], h5py.Group):
                    walk(g[k])
        walk(fh)
    assert "v_reset" not in names


def test_subtract_reset_drops_the_field_but_keeps_the_fact(tmp_path):
    """A subtract reset has reset target 0, so the field goes — the *semantics*
    must still be recorded, or the file silently claims a zero reset."""
    r, _net, w = _compile("mnist_mlp.snn")
    out = str(tmp_path / "sub.nir")
    export_nir(r.graph, out, w, strict=True)

    # what the equations do is visible in metadata, not lost with the field
    import nir
    g = nir.read(out)
    lifs = [n for n in g.nodes.values() if type(n).__name__ == "LIF"]
    assert lifs and all(n.metadata.get("snndsl_reset") for n in lifs)
    assert any(n.metadata.get("snndsl_reset") == "subtract" for n in lifs)


def test_downgrade_leaves_nonzero_reset_alone(tmp_path):
    """The rewrite is a targeted drop, not a blanket strip."""
    import nir

    path = str(tmp_path / "g.nir")
    g = nir.NIRGraph(nodes={
        "in": nir.Input(np.array([2])),
        "n": nir.LIF(tau=np.full(2, 2.0), r=np.ones(2), v_leak=np.zeros(2),
                     v_threshold=np.ones(2), v_reset=np.full(2, 0.7)),
        "out": nir.Output(np.array([2])),
    }, edges=[("in", "n"), ("n", "out")])
    nir.write(path, g)
    removed = _downgrade_dialect(path, "1.0.4")
    assert removed == []
    reread = nir.read(path)
    assert float(np.asarray(reread.nodes["n"].v_reset).flatten()[0]) == pytest.approx(0.7)


# -------------------------------------------------------------------- probes
def test_probe_reports_not_installed_instead_of_crashing():
    """A missing vendor stack is information, not an exception."""
    res = targets_mod.probe(
        os.path.join(ROOT, "build_hw", "mnist_hw.nir") if os.path.exists(
            os.path.join(ROOT, "build_hw", "mnist_hw.nir")) else _write_minimal_nir(),
        targets=["definitely-not-a-target"])
    assert res == []          # unknown keys are skipped, not fatal


def _write_minimal_nir() -> str:
    import nir
    path = os.path.join(tempfile.gettempdir(), "minimal_probe.nir")
    g = nir.NIRGraph(nodes={
        "in": nir.Input(np.array([2])),
        "a": nir.Affine(np.eye(2, dtype=np.float32), np.zeros(2, dtype=np.float32)),
        "n": nir.LIF(tau=np.full(2, 2.0), r=np.ones(2), v_leak=np.zeros(2),
                     v_threshold=np.ones(2)),
        "out": nir.Output(np.array([2])),
    }, edges=[("in", "a"), ("a", "n"), ("n", "out")])
    nir.write(path, g)
    return path


def test_probe_survives_a_broken_interpreter():
    """If the vendor interpreter cannot even start, say so and keep going."""
    old = os.environ.get(targets_mod.VENDOR_PYTHON_ENV)
    os.environ[targets_mod.VENDOR_PYTHON_ENV] = "/nonexistent/python"
    try:
        res = targets_mod.probe(_write_minimal_nir(), targets=["sinabs"])
    finally:
        if old is None:
            os.environ.pop(targets_mod.VENDOR_PYTHON_ENV, None)
        else:
            os.environ[targets_mod.VENDOR_PYTHON_ENV] = old
    assert len(res) == 1
    assert res[0].status in ("error", "not-installed", "upstream-gap")


def test_third_party_reproduces_our_spikes_on_a_real_batch(tmp_path):
    """The load-bearing claim: a foreign runtime, same file, same spikes."""
    r, _net, w = _compile("mnist_hw.snn")
    nir_path = str(tmp_path / "x.nir")
    export_nir(r.graph, nir_path, w, strict=True)

    import nir
    from snndsl.backends import nir_sim

    torch.manual_seed(0)
    sample = torch.rand(4, 1, 28, 28) * 0.9
    g = nir.read(nir_path)
    ref = nir_sim.readout(g, nir_sim.simulate(
        g, sample, steps=int(r.graph.temporal.steps),
        dt=float(r.graph.temporal.dt)), mode="spikes")
    assert ref.any(), "network must fire, otherwise the comparison is vacuous"

    x_np = str(tmp_path / "in.npy")
    r_np = str(tmp_path / "ref.npy")
    np.save(x_np, sample.numpy())
    np.save(r_np, ref.numpy())

    res = targets_mod.probe(nir_path, x_np, r_np,
                            steps=int(r.graph.temporal.steps), batch=4,
                            targets=["nirtorch"])[0]
    assert res.status == "ok", res.detail
    assert res.numbers["max_abs_error"] < 1e-5
    assert res.numbers["top1_agreement"] == 1.0


def test_matrix_never_reports_a_load_only_backend_as_agreement():
    """'it loaded' must not be printed in the same column as 'it matched'."""
    fake = [
        targets_mod.TargetResult("a", "vendor-a", "V", "ok", "", "1.0.8",
                                 {"max_abs_error": 0.0, "top1_agreement": 1.0}),
        targets_mod.TargetResult("b", "vendor-b", "V", "load-only", "", "1.0.8",
                                 {"convention": "per-step", "spike_rate": 13.0}),
        targets_mod.TargetResult("c", "vendor-c", "V", "upstream-gap",
                                 "frontend reads pre-1.0 field names"),
    ]
    text = targets_mod.format_matrix(fake)
    assert "max|Δ|" in text
    assert "load-only" in text
    assert "not a binary spike stream" in text
    assert "tau_mem" in text or "pre-1.0" in text


# -------------------------------------------------------------------- report
def test_report_is_one_file_with_no_external_references(tmp_path):
    """These reports get mailed to locked-down laptops; a CDN reference blanks them."""
    from snndsl.deploy_html import build_deploy_report

    path = str(tmp_path / "r.html")
    info = build_deploy_report(
        os.path.join(ROOT, "examples", "mnist_hw.snn"), path,
        samples=4, probe_targets=False)
    text = open(path, encoding="utf-8").read()

    assert info["deployable"] is True
    assert "<script" not in text.lower()
    assert "http://" not in text and "https://" not in text
    assert "DEPLOYABLE" in text
    # a weightless report must say so rather than look authoritative
    assert "随机初始化" in text
    assert len(text) > 4000


def test_report_states_what_it_did_not_prove(tmp_path):
    from snndsl.deploy_html import build_deploy_report

    path = str(tmp_path / "r2.html")
    build_deploy_report(os.path.join(ROOT, "examples", "mnist_hw.snn"), path,
                        samples=4, probe_targets=False)
    text = open(path, encoding="utf-8").read()
    assert "没在真实芯片上跑过" in text


def test_report_on_a_blocked_network_says_not_deployable(tmp_path):
    """A convolutional SNN cannot cross NIR 1.0.8; the report must lead with that."""
    from snndsl.deploy_html import build_deploy_report

    path = str(tmp_path / "c.html")
    info = build_deploy_report(
        os.path.join(ROOT, "examples", "cifar_hw.snn"), path,
        samples=2, probe_targets=False)
    text = open(path, encoding="utf-8").read()
    assert info["deployable"] is False
    assert info["blockers"] >= 1
    assert "NOT DEPLOYABLE" in text
