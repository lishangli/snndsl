"""Tests for deployment analysis and the NIR reference interpreter.

These lock in the behaviours that make the deploy check trustworthy:

* the exporter carries the current gain ``r`` that the DSL equations imply
* a zero-reset network is bit-exact across PyTorch and NIR
* a subtract-reset network is reported as divergent, not silently accepted
* a header that disagrees with the equations is a blocker, not a warning
"""
from __future__ import annotations

import re

import numpy as np
import pytest
import torch

from snndsl import compile_source, load_module
from snndsl.backends import nir_sim
from snndsl.backends.nir_export import build_nir_graph, recognise_lif
from snndsl.deploy import analyze, verify

HW = "examples/mnist_hw.snn"
MLP = "examples/mnist_mlp.snn"
MEM = "examples/mnist_membrane.snn"

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
    dense  h1 : 16 <- x
    neuron h1 -> LIF
    dense  y  : 3 <- h1
    neuron y  -> LIF
    output out <- y
infer:
    readout = spikes
"""


def _compile(text: str, name: str):
    res = compile_source(text)
    net = load_module(res, f"/tmp/_t_{name}.py")()
    net.eval()
    return res, net


def _weights(net):
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


# ------------------------------------------------------------------ recognise
def test_current_gain_is_tau_over_dt():
    """`v = v*(1-dt/tau) + i` feeds an unscaled current: r = tau/dt in NIR."""
    res = compile_source(SMALL)
    node = next(n for n in res.graph.nodes if n.op == "neuron")
    lif = recognise_lif(res.graph, node)
    assert lif is not None
    assert lif["tau"] == pytest.approx(2.0)
    assert lif["r"] == pytest.approx(2.0), "current gain must not default to 1"
    assert lif["reset_mode"] == "zero"


def test_adaptive_neuron_is_not_mistaken_for_lif():
    """ALIF has two states; it must not be silently exported as a plain LIF."""
    res = compile_source(open("examples/adaptive_lif.snn").read())
    alif_nodes = [n for n in res.graph.nodes if n.op == "neuron"
                  and len(res.graph.neurons[n.attrs["model"]].states) > 1]
    assert alif_nodes, "fixture should contain a multi-state neuron"
    for node in alif_nodes:
        assert recognise_lif(res.graph, node) is None


# ------------------------------------------------------------------- analyze
def test_spike_readout_is_deployable():
    res = compile_source(open(HW).read())
    rep = analyze(res.graph)
    assert rep.deployable
    assert rep.readout == "spikes"


def test_membrane_readout_blocks_deployment():
    """A spiking chip emits spikes; it cannot hand back a pre-reset potential."""
    res = compile_source(open(MEM).read())
    rep = analyze(res.graph)
    assert not rep.deployable
    assert any(f.code == "membrane-readout" for f in rep.blockers)


def test_subtract_reset_is_flagged_but_not_blocking():
    res = compile_source(open(MLP).read())
    rep = analyze(res.graph)
    assert rep.deployable
    assert any(f.code == "reset-subtract" for f in rep.warnings)


def test_header_body_mismatch_is_a_blocker():
    """`reset = zero` in the header while the body subtracts must not pass."""
    src = open(HW).read().replace("v = v * (1 - s)", "v = v - s * v_th")
    res = compile_source(src)
    rep = analyze(res.graph)
    assert not rep.deployable
    codes = {f.code for f in rep.blockers}
    assert "reset-mismatch" in codes


# -------------------------------------------------------------------- numerics
def test_zero_reset_is_bit_exact():
    """The whole promise: train here, run there, same spikes."""
    res, net = _compile(open(HW).read(), "hw")
    torch.manual_seed(0)
    x = torch.rand(8, 1, 28, 28)
    out = verify(res.graph, net, x, weights=_weights(net))
    assert out["equivalent"], out
    assert out["max_abs_error"] == 0.0
    assert out["top1_agreement"] == 1.0


def test_subtract_reset_diverges_and_is_reported():
    """The failure mode is real, so the checker must catch it, not hide it."""
    res, net = _compile(open(MLP).read(), "mlp")
    torch.manual_seed(0)
    x = torch.rand(16, 1, 28, 28)
    out = verify(res.graph, net, x, weights=_weights(net))
    assert not out["equivalent"]


def test_simulator_matches_hand_executed_lif():
    """Independent check of the interpreter against a hand-written recurrence."""
    res, net = _compile(SMALL, "tiny")
    w = _weights(net)
    graph, _ = build_nir_graph(res.graph, w)
    torch.manual_seed(1)
    x = torch.rand(3, 1, 4, 4)
    sim = nir_sim.simulate(graph, x, steps=4, dt=1.0)

    v1 = torch.zeros(3, 16)
    v2 = torch.zeros(3, 3)
    spikes = []
    for _ in range(4):
        v1 = v1 * 0.5 + torch.nn.functional.linear(
            x.reshape(3, -1),
            torch.as_tensor(w["w_h1"]), torch.as_tensor(w["b_h1"]))
        s1 = (v1 > 1.0).float()
        v1 = v1 * (1 - s1)
        v2 = v2 * 0.5 + torch.nn.functional.linear(
            s1, torch.as_tensor(w["w_y"]), torch.as_tensor(w["b_y"]))
        s2 = (v2 > 1.0).float()
        v2 = v2 * (1 - s2)
        spikes.append(s2)
    expected = torch.stack(spikes)
    assert torch.allclose(sim["spikes"]["y_1"], expected, atol=1e-6)


def test_sequence_flag_is_not_inferred_from_shape():
    """A batch of N must never be read as N time steps."""
    res, net = _compile(SMALL, "tiny_seq")
    graph, _ = build_nir_graph(res.graph, _weights(net))
    x = torch.rand(3, 1, 4, 4)
    one = nir_sim.simulate(graph, x, steps=4)
    seq = nir_sim.simulate(graph, x.unsqueeze(0).repeat(4, 1, 1, 1, 1),
                           is_sequence=True)
    assert one["steps"] == seq["steps"] == 4
    assert torch.allclose(one["output"], seq["output"])


def test_simulator_reports_unknown_nodes():
    """Silence is the enemy: an unmodelled node must raise, not be skipped."""
    nir = pytest.importorskip("nir")
    import numpy as np
    g = nir.NIRGraph(
        nodes={
            "in": nir.Input(input_type={"input": np.array([2])}),
            "d": nir.Delay(delay=np.array([1.0, 1.0])),
            "out": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("in", "d"), ("d", "out")],
    )
    with pytest.raises(nir_sim.NIRSimError):
        nir_sim.simulate(g, torch.zeros(1, 2), steps=2)


# ------------------------------------------------- convolutional deployment

CONV_AVG = "examples/cifar_hw_avg.snn"
CONV_MAX = "examples/cifar_hw.snn"


def test_convolutional_snn_is_deployable_and_bit_exact():
    """conv -> LIF -> pool -> conv -> dense round-trips through NIR exactly.

    Regression guard for a wrong call made in the other direction: the exporter
    used to write neuron parameters as a flat ``(C*H*W,)`` vector, which made a
    ``(C,H,W)`` activation fail NIR's shape inference, and ``deploy`` then
    reported *every* conv SNN as blocked - attributing to the standard a
    limitation that was only our own flattening. A NIR neuron derives its
    ``input_type`` from the shape of its parameter array, so per-element
    parameters of rank 3 are legal, and the full convolutional graph is exact.
    """
    res, net = _compile(open(CONV_AVG).read(), "convavg")
    rep = analyze(res.graph)
    assert rep.deployable, [f.message for f in rep.blockers]

    torch.manual_seed(0)
    x = torch.rand(4, 3, 32, 32)
    out = verify(res.graph, net, x, weights=_weights(net))
    assert out["max_abs_error"] == 0.0, out
    assert out["top1_agreement"] == 1.0


def test_max_pooling_is_the_real_blocker_for_conv_snns():
    """NIR 1.0.x has AvgPool2d and SumPool2d, no MaxPool2d.

    Exporting a max pool as sum pooled the 2x2 windows and multiplied the drive
    by four; the first layer's spike rate went 0.02 -> 0.37. That is a different
    network, so it has to block rather than warn.
    """
    res = compile_source(open(CONV_MAX).read())
    rep = analyze(res.graph)
    assert not rep.deployable
    assert any(f.code == "max-pool" for f in rep.blockers)


def test_neuron_parameters_keep_the_activation_rank():
    """The exported LIF must carry (C,H,W) parameters, not a flattened vector.

    Element count is identical either way - only the rank differs - so this
    costs no bytes and is the single change that makes conv SNNs deployable.
    """
    res = compile_source(open(CONV_AVG).read())
    graph, _ = build_nir_graph(res.graph, None)
    conv_neurons = [
        n for n in graph.nodes.values()
        if type(n).__name__ == "LIF" and np.asarray(n.tau).ndim == 3
    ]
    assert conv_neurons, "expected at least one rank-3 LIF parameter array"
    assert all(np.asarray(n.tau).size > 1 for n in conv_neurons)
