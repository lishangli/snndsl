"""The NIR -> snnTorch bridge, tested where it is easy to be quietly wrong.

Every case here is a defect that was found by a disagreement between two
implementations and then traced back to this file. None of them raise; all of
them produce a graph that loads, runs, and reports numbers - just not the right
ones. That is why they get tests instead of a comment.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
nir = pytest.importorskip("nir")

from snndsl.backends import snntorch_bridge as bridge  # noqa: E402


def _flatten_node():
    return nir.Flatten(start_dim=0, end_dim=-1, input_type={"input": np.array([4, 2, 2])})


def _avgpool_node():
    return nir.AvgPool2d(
        kernel_size=np.array([2, 2]), stride=np.array([2, 2]), padding=np.array([0, 0])
    )


def test_unmapped_nodes_fall_back_to_nirtorch_default_map():
    """``DEFAULT_MAP`` is a dict, not the callable older nirtorch exposed.

    Calling it raised ``TypeError``, a blanket ``except`` rewrote that into
    ``NotImplementedError(<node type>)``, and every convolutional graph was then
    reported as blocked by an upstream gap that did not exist.
    """
    pytest.importorskip("nirtorch", reason="make_map falls back to its DEFAULT_MAP")
    mapper = bridge.make_map(1.0)
    assert isinstance(mapper(_flatten_node()), torch.nn.Flatten)
    assert isinstance(mapper(_avgpool_node()), torch.nn.AvgPool2d)


def test_unknown_node_types_still_raise():
    """The fallback must not turn into "anything goes"."""

    pytest.importorskip("nirtorch", reason="make_map falls back to its DEFAULT_MAP")

    class NotANode:
        pass

    with pytest.raises(NotImplementedError):
        bridge.make_map(1.0)(NotANode())


def test_avg_pool_is_a_mean_and_not_a_sum():
    """nirtorch maps ``AvgPool2d`` to ``LPPool2d(norm_type=1)``, which sums.

    ``(sum |x|^1)^(1/1)`` is a 2x2 window sum: four times the drive of an
    average pool. Measured directly: a constant 2.0 input gives 8.0 through
    LPPool and 2.0 through AvgPool.
    """
    node = _avgpool_node()
    ours = bridge.avgpool_module(node)
    theirs = torch.nn.LPPool2d(norm_type=1, kernel_size=(2, 2), stride=(2, 2))
    x = torch.full((1, 1, 4, 4), 2.0)
    assert float(ours(x).flatten()[0]) == 2.0
    assert float(theirs(x).flatten()[0]) == 8.0


def test_multi_dimensional_neuron_parameters_are_rejected_not_averaged():
    """A genuinely per-neuron parameter must not be reduced to ``arr[0]``."""
    with pytest.raises(ValueError, match="non-uniform"):
        bridge._const(np.array([1.0, 2.0, 3.0]), 1.0)
    # a constant array (what every exporter writes) is fine
    assert bridge._const(np.full((4, 8, 8), 2.0), 1.0) == 2.0


def test_reset_states_clears_the_membrane_between_samples():
    """Streaming state carried across samples costs exactly one spike in eight.

    That is how the convolutional cross-check first disagreed with nirtorch:
    same weights, same currents, 0.125 difference on one output class.
    """
    neuron = torch.nn.Identity()

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = neuron
            self.inner.mem = torch.ones(3)

    holder = Holder()
    bridge.reset_states(holder)
    assert float(holder.inner.mem.abs().sum()) == 0.0
