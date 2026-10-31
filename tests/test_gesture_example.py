"""The gesture example is the "design a new SNN" walkthrough; these pin its claims.

Every assertion here corresponds to a sentence in ``docs/DESIGN_A_NETWORK.md``.
If one of them starts failing, the document is wrong too -- that is the point.

No training happens: the whole file is compile / export / analyse, so it stays
under a few seconds. Accuracy claims live in the doc, with the command to
reproduce them.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_NIR = importlib.util.find_spec("nir") is not None
pytestmark = pytest.mark.skipif(
    not (HAVE_TORCH and HAVE_NIR), reason="needs torch + nir")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "examples"))

import nir  # noqa: E402
import snndsl as snn  # noqa: E402
import torch  # noqa: E402
from snndsl.backends import nir_sim  # noqa: E402
from snndsl.backends.nir_export import export_nir  # noqa: E402
from snndsl.backends.snn_mlir import preflight  # noqa: E402
from snndsl.deploy import analyze, verify  # noqa: E402


@pytest.fixture(autouse=True)
def _seed():
    """Weight init draws from the global RNG, so a test order change is a
    different network. Pin it: these assertions are about behaviour, not luck."""
    torch.manual_seed(0)

from gesture_snn import N_CLASSES, GestureNet, make_dataset, merge_polarity  # noqa: E402


# ------------------------------------------------------------------- dataset
def test_dataset_is_six_classes_with_balanced_polarity():
    x, y = make_dataset(2, seed=0)
    assert x.shape == (2 * N_CLASSES, 2, 32, 32)
    assert sorted(set(y.tolist())) == list(range(N_CLASSES))
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0

    # right vs left must NOT be separable by how much activity there is: if they
    # were, the network could pass by counting events and the task would say
    # nothing about structure.
    x, y = make_dataset(40, seed=5)
    on = {c: float(x[y == c][:, 0].mean()) for c in range(N_CLASSES)}
    assert abs(on[0] - on[1]) < 0.02 * (on[0] + on[1]), on


def test_merging_polarity_destroys_the_information():
    x, _ = make_dataset(2, seed=0)
    merged = merge_polarity(x)
    assert merged.shape[1] == 2
    # both channels identical -> a right sweep and a left sweep become the same
    # tensor, which is what makes the ablation in the doc a real ablation
    assert torch.allclose(merged[:, 0], merged[:, 1])


# ------------------------------------------------------------------ structure
def test_forward_compiles_to_the_documented_graph():
    model = GestureNet()
    graph = model.compile(quiet=True).graph
    ops = [(n.id, n.op) for n in graph.nodes]
    assert ops == [
        ("x", "input"), ("e", "encode"), ("conv1", "conv"), ("lif1", "neuron"),
        ("pool1", "pool"), ("conv2", "conv"), ("lif2", "neuron"),
        ("pool2", "pool"), ("fc1", "dense"), ("lif3", "neuron"),
        ("fc2", "dense"), ("lif4", "neuron"), ("out", "output"),
    ]
    assert [tuple(n.shape) for n in graph.nodes][2] == (16, 32, 32)
    assert sum(p.numel() for p in model._compiled.parameters()) == 136_470


def test_two_neuron_definitions_appear_when_only_one_threshold_differs():
    """lif3 uses v_th=0.6, so the graph carries LIF *and* LIF2.

    This is the cost of ``neuron:`` being structure rather than configuration:
    a different threshold is a different neuron model, not a hyper-parameter.
    """
    text = GestureNet().to_snn()
    assert "neuron LIF:" in text
    assert "neuron LIF2:" in text
    assert "param v_th = 0.6" in text


# ------------------------------------------------------- deployment verdicts
def test_membrane_readout_is_a_blocker_but_exports_cleanly():
    model = GestureNet()
    graph = model.compile(quiet=True).graph
    rep = analyze(graph)
    assert rep.readout == "membrane"
    assert not rep.deployable
    blockers = [f for f in rep.findings if f.severity == "blocker"]
    assert [f.where for f in blockers] == ["out"]
    # ... and yet the file itself is fine: the blocker is about what the target
    # can read back, not about the export being wrong
    assert preflight(graph)["faithful"] is True


def test_spike_readout_is_deployable():
    model = GestureNet()
    model.readout = "spikes"
    rep = analyze(model.compile(quiet=True).graph)
    assert rep.deployable
    assert rep.readout == "spikes"


def test_maxpool_and_alif_are_accepted_but_not_faithful():
    """The backend says yes to both; the network it runs is not the one we built."""
    model = GestureNet(neuron="alif", pool="max", reset="subtract")
    pf = preflight(model.compile(quiet=True).graph)
    assert pf["accepts"] is True
    assert pf["faithful"] is False
    rules = {item["rule"] for item in pf["lossy"]}
    assert rules == {"maxpool_as_avgpool", "neuron_downgraded_to_li"}


# ------------------------------------------------------------ fidelity oracle
class LinearHead(snn.Module):
    """Output fed by a *linear* node, not by a neuron.

    ``GestureNet`` ends in a neuron, so it exercises the spike/membrane stream
    path. This one exercises the path where there is no train to decode at all
    -- the branch that used to fall back to the last step.
    """

    input_shape = (8,)
    steps = 4
    readout = "membrane"

    def __init__(self):
        super().__init__()
        self.fc1 = snn.Dense(16, scale=2.0)
        self.lif1 = snn.LIF(tau=2.0)
        self.fc2 = snn.Dense(4)

    def forward(self, x):
        return self.fc2(self.lif1(self.fc1(x)))


def test_output_stream_returns_the_window_not_the_last_step(tmp_path):
    """The regression that motivated the fix, pinned at its actual cause.

    ``codegen`` accumulates every output node over T and divides by T. The
    interpreter had no train to decode when the Output node is fed by a linear
    head, so it fell back to ``sim["output"]`` -- the *last* step. On the gesture
    example that mismatch showed up as max|Δ|=1.05 while top-1 agreed on 100% of
    samples, i.e. a correct export reported as ``equivalent=False``.

    Asserting on the stream (rather than on a float comparison) keeps this test
    independent of the separate, real phenomenon below: at float32 a membrane
    potential sitting on the threshold can flip a spike, which moves a logit by
    ~0.08 no matter how correct either side is.
    """
    model = LinearHead()
    result = model.compile(quiet=True)
    graph, net = result.graph, model._compiled
    assert graph.nodes[-1].op == "output"
    assert graph.nodes[-2].op == "dense"       # the branch under test

    weights = {k: v.detach().numpy() for k, v in net.state_dict().items()}
    path = str(tmp_path / "lin.nir")
    export_nir(graph, path, weights=weights, target="snn-mlir")
    x = torch.linspace(0, 1, 32).reshape(4, 8)
    sim = nir_sim.simulate(nir.read(path), x, steps=graph.temporal.steps)

    stream = nir_sim.output_stream(nir.read(path), sim, "membrane")
    assert stream is not None, "no stream -> readout would fall back to one step"
    assert stream.shape == (graph.temporal.steps, 4, 4)
    # the last step and the window average are genuinely different on this graph,
    # so a fallback to either one cannot pass for the other
    assert not torch.allclose(stream[-1], stream.mean(0))

    got = nir_sim.readout(nir.read(path), sim, "membrane")
    assert torch.allclose(got, stream.mean(0))

    out = verify(graph, net, x, weights=weights, nir_path=path)
    # a spike flip is worth ~0.08 here; the old fallback was worth ~1.0
    assert out["max_abs_error"] < 0.2, out["max_abs_error"]
    assert out["top1_agreement"] == 1.0


def test_untrained_network_is_not_silent():
    """A net whose neurons never fire trains on nothing; catch it before that.

    The first version of this example divided the event counts by 3, which put
    the mean input at 0.05. ``lif1`` fired at 7e-4, the next two layers at
    exactly 0, every logit was 0 and the loss sat at ln(6) for three epochs.
    The fix was 99th-percentile normalisation of the input (auto-exposure).
    """
    model = GestureNet()
    model.compile(quiet=True)
    net = model._compiled
    net.collect = True
    with torch.no_grad():
        x, _ = make_dataset(4, seed=1)
        logits = net(x)
    rates = dict(net.spike_trace)
    net.collect = False

    assert set(rates) == {"lif1", "lif2", "lif3", "lif4"}
    # The broken version had lif1 at 7e-4 and the next two at *exactly* 0, so
    # "strictly positive" is the discriminating assertion -- exact rates depend
    # on weight init and a batch of 12 makes them noisy.
    assert all(rates[k] > 1e-3 for k in ("lif1", "lif2", "lif3")), rates
    assert rates["lif1"] > 0.1, rates
    # lif4 may be exactly 0 before training, and that asymmetry is the point:
    # a membrane readout still sees its potential, a spike count sees nothing.
    assert rates["lif4"] >= 0.0
    assert len({round(float(v), 6) for v in logits.ravel()}) > 1
