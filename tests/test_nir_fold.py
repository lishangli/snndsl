"""The time loop that a `.nir` cannot hold, and the fold that puts it back.

Everything here is measured against PyTorch, because the claim being made is
numerical: the folded graph computes the same spike train, and the unrolled one
does not answer in a single execution.  A test that only counted nodes would
pass on a graph that computes nothing.
"""
from __future__ import annotations

import numpy as np
import pytest
from snndsl.backends import nir_fold

pytest.importorskip("torch")
snn = pytest.importorskip("snntorch")
nirtorch = pytest.importorskip("nirtorch")
nir = pytest.importorskip("nir")

T = 4
D_IN, D_HID, D_OUT = 16, 8, 4


def _to_nir(layer):
    """snnTorch -> NIR for one neuron module.

    Two details decide whether the converted network is the same network, and
    neither is checked by anything: ``beta`` maps to a *pair* (tau, r) with one
    free parameter, and the input scaling is the free one.  snnTorch integrates
    ``v <- beta*v + i``; NIR integrates ``v <- (1 - dt/tau)*v + (dt/tau)*r*i``.
    Matching them needs ``r = tau``, not ``r = 1``.  With r = 1 the file is still
    valid and the network is 9.5x slower -- the exact class of silent drift this
    project exists to catch, found here in our own converter first.
    """
    tau = -1.0 / np.log(float(layer.beta))
    return nir.LIF(
        tau=np.array([tau]),
        r=np.array([tau]),
        v_leak=np.array([0.0]),
        v_threshold=np.array([float(layer.threshold)]),
        v_reset=None,                      # None == subtract-on-spike
    )


def _build(T_steps: int = T):
    import torch

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(D_IN, D_HID)
            self.lif1 = snn.Leaky(beta=0.9, threshold=0.6)
            self.fc2 = torch.nn.Linear(D_HID, D_OUT)
            self.lif2 = snn.Leaky(beta=0.9, threshold=0.6)

        def forward(self, x):                       # the snnTorch tutorial shape
            m1 = self.lif1.init_leaky()
            m2 = self.lif2.init_leaky()
            for _ in range(T_steps):
                sp1, m1 = self.lif1(self.fc1(x), m1)
                sp2, m2 = self.lif2(self.fc2(sp1), m2)
            return sp2

        def trace(self, x):
            import torch as _t

            m1 = self.lif1.init_leaky()
            m2 = self.lif2.init_leaky()
            a, b = [], []
            with _t.no_grad():
                for _ in range(T_steps):
                    sp1, m1 = self.lif1(self.fc1(x), m1)
                    sp2, m2 = self.lif2(self.fc2(sp1), m2)
                    a.append(sp1)
                    b.append(sp2)
            return _t.stack(a), _t.stack(b)

    torch.manual_seed(0)
    model = Net().eval()
    return model, torch.randn(D_IN) * 4


def _export(model):
    graph = nirtorch.torch_to_nir(
        model, {snn.Leaky: _to_nir},
        stateful_modules={snn.Leaky}, type_check=False)
    return graph


def test_detect_finds_the_repeats_the_file_does_not_admit():
    model, _ = _build()
    per = nir_fold.detect(_export(model))
    assert per.repeats == T, per.summary()
    assert per.period == 4, "two affine + two LIF per step"
    assert per.identical is True
    # the copies share no edge: the recurrence lived in a Python variable
    assert per.chained is False
    assert per.foldable is True


def test_folded_graph_reproduces_pytorch_position_by_position():
    model, x = _build()
    _, ref = model.trace(x)
    folded, per = nir_fold.fold(_export(model))

    from snndsl.backends import nir_sim

    got = nir_sim.simulate(folded, x.numpy(), steps=per.repeats)
    assert got["spikes"]["lif2"].shape == ref.shape
    differ = float(np.abs(got["spikes"]["lif2"].numpy() - ref.numpy()).max())
    assert differ == 0.0, "folded graph is not the trained network"
    assert float(got["spikes"]["lif1"].sum()) == float(model.trace(x)[0].sum())


def test_unrolled_graph_does_not_answer_in_one_execution():
    """One execution is one timestep; the unrolled file disagrees with that."""
    model, x = _build()
    _, ref = model.trace(x)
    graph = _export(model)

    from snndsl.backends import nir_sim

    one = nir_sim.simulate(graph, x.numpy(), steps=1)["spikes"]["lif2_3"]
    assert np.abs(one.numpy() - ref.numpy()).max() > 0, \
        "if this ever passes, the unroll is harmless and the fold is pointless"

    # and it costs T copies x T steps instead of 1 x T
    per = nir_fold.detect(graph)
    assert per.repeats * per.repeats == T * T


def test_different_weights_are_a_real_unroll_and_are_refused():
    model, _ = _build()
    graph = _export(model)
    # perturb one copy: this is now a network, not a loop
    victim = graph.nodes["fc1_3"]
    victim.weight = victim.weight * 2.0

    per = nir_fold.detect(graph)
    assert per.foldable is False
    with pytest.raises(nir_fold.FoldError):
        nir_fold.fold(graph)


def test_file_carries_the_timestep_count_it_had_no_field_for(tmp_path):
    model, x = _build()
    _, ref = model.trace(x)
    src = tmp_path / "net.nir"
    nir_fold.save(_export(model), str(src))

    out, per = nir_fold.fold_file(str(src))
    again = nir.read(out)
    assert int(again.metadata["snndsl.timesteps"]) == T
    assert len(again.nodes) < len(nir.read(str(src)).nodes)

    from snndsl.backends import nir_sim

    got = nir_sim.simulate(again, x.numpy(),
                           steps=int(again.metadata["snndsl.timesteps"]))
    assert float(np.abs(got["spikes"]["lif2"].numpy() - ref.numpy()).max()) == 0.0


def test_serialised_unroll_gains_one_output_per_dangling_copy(tmp_path):
    """A pathology worth pinning: the file, once written, declares T outputs.

    ``nir.write`` materialises an ``Output`` for every node with no successor.
    The unrolled unroll leaves T-1 copies dangling, so the round-tripped file has
    T outputs and every consumer that expects one rejects it -- ``nir_sim``
    raises ``expected exactly one Output node, found 4``.  Folding drops them.
    """
    model, _ = _build()
    src = tmp_path / "net.nir"
    nir_fold.save(_export(model), str(src))
    back = nir.read(str(src))
    outs = [n for n in back.nodes if type(back.nodes[n]).__name__ == "Output"]
    assert len(outs) == T

    _, per = nir_fold.fold_file(str(src), str(tmp_path / "folded.nir"))
    assert per.dropped_outputs == T - 1
    folded = nir.read(str(tmp_path / "folded.nir"))
    assert len([n for n in folded.nodes
                if type(folded.nodes[n]).__name__ == "Output"]) == 1
