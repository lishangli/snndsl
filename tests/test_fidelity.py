"""保真报告必须说数字，所以它自己得先被数字约束住。

四条，分别钉住：自比较为零、量化越狠越差、tau 接近 dt 时两种离散化分道扬镳、
沉默的网络不许被报成「等价」。
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
nir = pytest.importorskip("nir")

from snndsl.backends import fidelity  # noqa: E402


def _build(tau: float, width: int = 32, out: int = 16, n_in: int = 64):
    """A two-layer LIF net with a chosen tau, built straight from the nir API."""
    rng = np.random.default_rng(3)
    nodes = {
        "x": nir.Input(input_type={"input": np.array([n_in])}),
        "fc1": nir.Affine(
            weight=rng.standard_normal((width, n_in)).astype(np.float32) / np.sqrt(n_in),
            bias=np.zeros(width, dtype=np.float32)),
        "l1": nir.LIF(tau=np.full(width, tau, dtype=np.float32),
                      r=np.full(width, tau, dtype=np.float32),
                      v_leak=np.zeros(width, dtype=np.float32),
                      v_threshold=np.full(width, 0.5, dtype=np.float32),
                      v_reset=None),
        "fc2": nir.Affine(
            weight=rng.standard_normal((out, width)).astype(np.float32) / np.sqrt(width),
            bias=np.zeros(out, dtype=np.float32)),
        "l2": nir.LIF(tau=np.full(out, tau, dtype=np.float32),
                      r=np.full(out, tau, dtype=np.float32),
                      v_leak=np.zeros(out, dtype=np.float32),
                      v_threshold=np.full(out, 0.5, dtype=np.float32),
                      v_reset=None),
        "y": nir.Output(output_type={"output": np.array([out])}),
    }
    edges = [("x", "fc1"), ("fc1", "l1"), ("l1", "fc2"), ("fc2", "l2"), ("l2", "y")]
    return nir.NIRGraph(nodes=nodes, edges=edges, type_check=False)


def test_every_variant_reports_a_bounded_percentage():
    """A percentage outside [0, 100] means the counter compares mismatched shapes."""
    graph = _build(tau=8.0)
    rep = fidelity.report(graph, steps=8, samples=4, seed=1)
    assert rep.steps == 8 and rep.samples == 4
    assert len(rep.variants) == 3
    assert all(0.0 <= v.bit_flip <= 100.0 for v in rep.variants)
    assert all(0.0 <= v.top1 <= 1.0 for v in rep.variants)
    # the layer breakdown has to cover the neurons that actually fired
    assert set(rep.variants[0].layers) <= set(graph.nodes)


def test_quantisation_error_grows_as_bits_shrink():
    """int4 must hurt more than int8 -- if not, the quantiser is not doing anything."""
    graph = _build(tau=8.0)
    rep = fidelity.report(graph, steps=8, samples=6, seed=2)
    by_name = {v.name: v for v in rep.variants}
    i8 = by_name["int8 权重（逐通道）"].bit_flip
    i4 = by_name["int4 权重（逐通道）"].bit_flip
    assert i4 > i8, (i4, i8)
    assert i8 < 5.0, f"int8 already at {i8:.2f}% -- the scale logic is suspect"


def test_tau_discretisation_matters_when_tau_is_close_to_dt():
    """The whole point: two legal readings of the same tau, far apart when tau ~ dt.

    Measured on this fixture: tau=20 gives 0.03%, tau=1.2 gives 1.95% -- a factor
    of 65. The absolute number stays small because both formulas keep the neuron
    in the same steady firing regime; what changes is *where* the spikes land, so
    an absolute threshold would be arbitrary. The ratio is the claim being tested.
    """
    slow = fidelity.report(_build(tau=20.0), steps=8, samples=6, seed=4)
    fast = fidelity.report(_build(tau=1.2), steps=8, samples=6, seed=4)
    tau_slow = {v.name: v for v in slow.variants}["tau：欧拉 → 指数"].bit_flip
    tau_fast = {v.name: v for v in fast.variants}["tau：欧拉 → 指数"].bit_flip
    assert tau_fast > tau_slow * 10, (tau_fast, tau_slow)


def test_a_silent_network_is_refused_rather_than_called_equivalent():
    """Zero against zero is not agreement; it is an empty comparison."""
    graph = _build(tau=8.0)
    for node in graph.nodes.values():
        if getattr(node, "weight", None) is not None:
            node.weight = np.zeros_like(np.asarray(node.weight))
    with pytest.raises(fidelity.FidelityError, match="silence"):
        fidelity.report(graph, steps=8, samples=2, seed=0)


def test_a_folded_file_carries_its_own_step_count(tmp_path):
    """`fold` writes T into metadata; fidelity must read it instead of asking."""
    from snndsl.backends import nir_fold

    graph = _build(tau=8.0)
    graph.metadata = {"snndsl.timesteps": "8"}
    path = tmp_path / "net.nir"
    nir_fold.save(graph, str(path))

    rep = fidelity.report_file(str(path), samples=2, seed=1)
    assert rep.steps == 8
