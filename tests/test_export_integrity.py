"""Tests for export integrity - does the file that ships match the model?

These exist because of a real failure: training produced a ``.nir`` file with
all-zero weights, and ``deploy --verify`` reported PRESERVED, because it was
verifying an in-memory graph that still had the weights while the file on disk
had none. A zero-weight spiking network emits nothing, so every downstream
comparison was zero against zero.

The four tests below pin down the four ways that can happen again:

1. exporting without weights silently zero-filling the matrices
2. a write that does not survive a read-back
3. verifying the in-memory graph instead of the artifact
4. comparing outputs that are constant, which proves nothing either way
"""
from __future__ import annotations

import os

import nir
import numpy as np
import pytest
import torch
from snndsl import compile_source, load_module
from snndsl.backends.nir_export import (
    build_nir_graph,
    export_nir,
    parameter_fingerprint,
    roundtrip_problems,
)
from snndsl.deploy import verify

SMALL = """
program tiny
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
pipeline:
    input  x  : 1, 4, 4
    dense  h1 : 16 <- x  with scale=3.0
    neuron h1 -> LIF
    dense  y  : 3 <- h1
    neuron y  -> LIF
    output out <- y
infer:
    readout = spikes
"""


def _compile(name: str):
    res = compile_source(SMALL)
    net = load_module(res, f"/tmp/_ei_{name}.py")()
    net.eval()
    return res, net


def _weights(net):
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


def _train_like(net, seed=0):
    """Give the layers non-trivial weights without running a training loop."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(torch.randn_like(p) * 0.3)
    return net


# --------------------------------------------------------------- 1. zero fill
def test_weightless_export_is_refused_in_strict_mode(tmp_path):
    """A graph with layers but no weights must not be written silently."""
    res, _ = _compile("strict")
    path = str(tmp_path / "x.nir")
    with pytest.raises(RuntimeError, match="weightless"):
        export_nir(res.graph, path, None, strict=True)


def test_weightless_export_is_recorded_when_not_strict(tmp_path):
    """Structure-only export stays legal, but it has to say what it is."""
    res, _ = _compile("nonstrict")
    path = str(tmp_path / "y.nir")
    rep = export_nir(res.graph, path, None, strict=False)
    assert rep["weights_status"] == "absent"
    assert rep["missing_weight_nodes"], "zero-filled layers must be listed"
    assert set(rep["zero_weight_nodes"]) == set(rep["missing_weight_nodes"])


def test_supplied_weights_are_not_zero_filled(tmp_path):
    res, net = _compile("supplied")
    _train_like(net)
    path = str(tmp_path / "z.nir")
    rep = export_nir(res.graph, path, _weights(net), strict=True)
    assert rep["weights_status"] == "supplied"
    assert rep["missing_weight_nodes"] == []
    back = nir.read(path)
    for node in back.nodes.values():
        if type(node).__name__ == "Affine":
            assert np.any(np.asarray(node.weight)), "weights vanished on write"


# ------------------------------------------------------------- 2. round-trip
def test_roundtrip_diff_detects_a_dropped_field():
    """The checker must catch a field that the serialiser loses."""
    res, net = _compile("rt")
    _train_like(net)
    front, _ = build_nir_graph(res.graph, _weights(net))
    back = nir.read(_write(front))
    assert roundtrip_problems(front, back) == []

    # simulate a writer that drops the bias: NIR models that as Linear, which
    # has no bias field at all, so the type change *is* the dropped field
    back.nodes["h1"] = nir.Linear(weight=back.nodes["h1"].weight)
    problems = roundtrip_problems(front, back)
    assert any("changed type" in p for p in problems), problems


def test_fingerprint_changes_with_weights():
    res, net = _compile("fp")
    empty, _ = build_nir_graph(res.graph, None)
    _train_like(net)
    trained, _ = build_nir_graph(res.graph, _weights(net))
    assert parameter_fingerprint(empty) != parameter_fingerprint(trained)


# -------------------------------------------------- 3. verify the real artifact
def test_verify_flags_a_zero_weight_file(tmp_path):
    """The exact bug: the artifact is empty, the module is trained."""
    res, net = _compile("blank")
    _train_like(net)
    path = str(tmp_path / "blank.nir")
    export_nir(res.graph, path, None)          # what training used to write

    out = verify(res.graph, net, torch.rand(8, 1, 4, 4), nir_path=path)
    assert out["nir_source"] == "file"
    assert out["blank_weight_nodes"], "all-zero weights must be reported"
    assert out["equivalent"] is False, "silence is not equivalence"


def test_verify_reads_the_file_it_is_given(tmp_path):
    """Two files, same graph, different weights -> different verdicts."""
    res, net = _compile("two")
    _train_like(net, seed=1)

    good = str(tmp_path / "good.nir")
    export_nir(res.graph, good, _weights(net), strict=True)

    other, net2 = _compile("two_b")
    _train_like(net2, seed=7)
    bad = str(tmp_path / "bad.nir")
    export_nir(res.graph, bad, _weights(net2), strict=True)

    x = torch.rand(8, 1, 4, 4)
    a = verify(res.graph, net, x, nir_path=good)
    b = verify(res.graph, net, x, nir_path=bad)
    assert a["equivalent"] is True
    assert b["equivalent"] is False, "weights from another model must not pass"


# ------------------------------------------------------- 4. vacuous comparisons
def test_constant_output_is_not_counted_as_a_match():
    """Zero input, zero output, zero difference - and no information."""
    res, net = _compile("vac")
    _train_like(net)
    out = verify(res.graph, net, torch.zeros(8, 1, 4, 4))
    assert out["vacuous"] is True
    assert out["equivalent"] is False
    assert out["relative_error"] == 0.0, "this is why the flag is needed"


def _write(graph) -> str:
    path = "/tmp/_ei_roundtrip.nir"
    nir.write(path, graph)
    return path


# --------------------------------------------- 5. third-party cross-validation
nirtorch = pytest.importorskip("nirtorch", reason="cross-validation needs nirtorch")
snn = pytest.importorskip("snntorch", reason="cross-validation needs snntorch")


def test_third_party_backend_reproduces_our_spikes(tmp_path):
    """nirtorch builds the graph, snnTorch runs the neuron - we only map params."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
    from cross_validate import build_third_party, reset_state

    res, net = _compile("xval")
    _train_like(net, seed=3)
    path = str(tmp_path / "x.nir")
    export_nir(res.graph, path, _weights(net), strict=True)

    _, third, mapping = build_third_party(path, 1.0)
    neurons = [m for m in mapping if "beta" in m]
    assert neurons, "the mapping log must contain the neuron mappings"
    assert all(m["who_integrates"] == "snnTorch" for m in neurons), (
        "the neuron must be executed by snnTorch, not by us")

    torch.manual_seed(0)
    x = torch.rand(12, 1, 4, 4)
    flat = x.reshape(x.shape[0], -1)

    with torch.no_grad():
        ours = net(x).detach()
    reset_state(third)
    acc = torch.zeros_like(ours)
    with torch.no_grad():
        for _ in range(res.graph.temporal.steps):
            acc = acc + third(flat)
    theirs = acc / res.graph.temporal.steps

    assert torch.equal(ours, theirs), (
        f"third-party backend disagrees: max diff "
        f"{(ours - theirs).abs().max().item():.3e}")
