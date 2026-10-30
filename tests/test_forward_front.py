"""The forward-style front end must produce the *same IR*, not a similar one.

Every test here pins a behaviour that would otherwise be a silent wrong answer:
a loop that unrolls time, a layer that renames itself, a graph that quietly
drops the skip connection.
"""
from __future__ import annotations

import pytest
import snndsl as snn
from snndsl.errors import SNNError


class Tiny(snn.Module):
    input_shape = (8,)
    steps = 4
    readout = "membrane"

    def __init__(self):
        super().__init__()
        self.fc1 = snn.Dense(16, scale=2.0)
        self.fc2 = snn.Dense(10)
        self.lif1 = snn.LIF(tau=2.0)
        self.lif2 = snn.LIF(tau=2.0)

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        return self.lif2(self.fc2(x))


def graph_of(model):
    return model.compile(load=False).graph


# ------------------------------------------------------------------ same graph
def test_forward_matches_handle_style_exactly():
    """Both Python front ends must reach the identical op sequence and shapes."""
    from snndsl.pyfront import LIF, Net

    handle = Net("tiny", steps=4, readout="membrane")
    x = handle.encode(handle.input(8), "rate", 1.0)
    h = handle.neuron(handle.dense(x, 16, scale=2.0, name="fc1"), LIF(tau=2.0),
                      name="lif1")
    handle.output(handle.neuron(handle.dense(h, 10, name="fc2"), LIF(tau=2.0),
                                name="lif2"), name="out")

    a = graph_of(Tiny())
    b = handle.compile().graph

    assert [(n.op, n.shape) for n in a.nodes] == [(n.op, n.shape) for n in b.nodes]
    assert a.temporal.steps == b.temporal.steps == 4
    assert a.infer["readout"] == b.infer["readout"] == "membrane"


def test_node_names_come_from_attribute_names():
    ids = [n.id for n in graph_of(Tiny()).nodes]
    assert ids == ["x", "e", "fc1", "lif1", "fc2", "lif2", "out"]


def test_auto_encode_can_be_disabled():
    class NoEnc(Tiny):
        auto_encode = False

        def forward(self, x):
            return self.lif1(self.fc1(self.encode(x)))

    ops = [n.op for n in graph_of(NoEnc()).nodes]
    assert ops.count("encode") == 1

    class Wrong(Tiny):
        def forward(self, x):
            return self.lif1(self.fc1(self.encode(x)))

    with pytest.raises(SNNError, match="already applied"):
        Wrong().compile(load=False)


# ----------------------------------------------------------- loops mean blocks
def test_range_of_attribute_is_block_duplication_not_time():
    """`range(self.depth)` emits `depth` nodes; the time loop is T in the IR."""
    class Stacked(snn.Module):
        input_shape = (8,)
        steps = 4

        def __init__(self, depth=3):
            super().__init__()
            self.depth = depth
            self.fc = snn.Dense(8, scale=2.0)
            self.lif = snn.LIF(tau=2.0)
            self.head = snn.Dense(10)

        def forward(self, x):
            for _ in range(self.depth):
                x = self.lif(self.fc(x))
            return self.head(x)

    g = graph_of(Stacked())
    assert [n.op for n in g.nodes].count("dense") == 4    # 3 blocks + head
    assert [n.op for n in g.nodes].count("neuron") == 3
    # every block is its own node, so a 2-block model really is smaller
    assert len(graph_of(Stacked(depth=2)).nodes) == len(g.nodes) - 2


def test_literal_range_equal_to_steps_is_rejected():
    """`for t in range(8)` in a steps=8 net would unroll time into 8 blocks."""
    class Typo(snn.Module):
        input_shape = (8,)
        steps = 8

        def __init__(self):
            super().__init__()
            self.fc = snn.Dense(10)

        def forward(self, x):
            for _ in range(8):
                x = self.fc(x)
            return x

    with pytest.raises(SNNError, match="same number as steps"):
        Typo().compile(load=False)


def test_explicit_time_loop_is_pinned_to_one_iteration():
    """Writing the T loop is redundant but must not unroll the graph."""
    class Looped(snn.Module):
        input_shape = (8,)
        steps = 6

        def forward(self, x):
            for _ in range(self.steps):
                x = self.lif(self.fc(x))
            return x

        def __init__(self):
            super().__init__()
            self.fc = snn.Dense(10)
            self.lif = snn.LIF()

    g = graph_of(Looped())
    assert [n.op for n in g.nodes].count("dense") == 1
    assert g.temporal.steps == 6


# ------------------------------------------------------------------ rejections
def test_same_instance_bound_to_two_names_is_rejected():
    class Shared(snn.Module):
        input_shape = (8,)

        def __init__(self):
            super().__init__()
            lif = snn.LIF(tau=2.0)
            self.a = lif
            self.b = lif

        def forward(self, x):
            return x

    with pytest.raises(SNNError, match="bound to both"):
        Shared()


def test_missing_input_shape_says_what_to_add():
    class NoShape(snn.Module):
        def forward(self, x):
            return x

    with pytest.raises(SNNError, match="input_shape"):
        NoShape().compile(load=False)


def test_returning_a_non_tensor_is_rejected():
    class ReturnsNone(snn.Module):
        input_shape = (8,)

        def forward(self, x):
            self.fc(x)
            return None

        def __init__(self):
            super().__init__()
            self.fc = snn.Dense(10)

    with pytest.raises(SNNError, match="returned NoneType"):
        ReturnsNone().compile(load=False)


def test_layer_outside_a_trace_says_so():
    fc = snn.Dense(10)
    with pytest.raises(SNNError, match=r"forward\(\) is being traced"):
        fc(snn.Tensor(None, None))  # type: ignore[arg-type]


def test_adding_a_raw_number_is_rejected():
    """`x + 1` would silently become a graph op; it must not."""
    class Biased(snn.Module):
        input_shape = (8,)

        def __init__(self):
            super().__init__()
            self.fc = snn.Dense(10)

        def forward(self, x):
            return self.fc(x) + 1

    with pytest.raises(SNNError, match="cannot add int"):
        Biased().compile(load=False)


def test_residual_uses_the_add_op():
    class Res(snn.Module):
        input_shape = (8,)

        def __init__(self):
            super().__init__()
            self.fc = snn.Dense(8, scale=1.5)
            self.lif = snn.LIF(tau=2.0)

        def forward(self, x):
            return self.fc(self.lif(self.fc(x))) + x

    ops = [n.op for n in graph_of(Res()).nodes]
    assert "add" in ops


# ------------------------------------------------------------- compiled object
def test_compile_returns_a_real_torch_module_and_roundtrips_to_snn():
    model = Tiny()
    result = model.compile()
    assert result.graph.temporal.steps == 4
    assert "class SnnNet" in result.module_source

    import torch
    assert isinstance(model._compiled, torch.nn.Module)

    text = model.to_snn()
    again = snn.compile_source(text, filename="<roundtrip>")
    assert [n.op for n in again.graph.nodes] == [n.op for n in result.graph.nodes]


def test_calling_the_description_runs_the_compiled_module():
    import torch

    model = Tiny()
    out = model(torch.zeros(2, 1, 8))
    assert tuple(out.shape) == (2, 10)


# ------------------------------------------------------------------- file route
def test_cli_picks_the_only_module_subclass(tmp_path):
    src = tmp_path / "one.py"
    src.write_text(
        "import snndsl as snn\n"
        "class Only(snn.Module):\n"
        "    input_shape = (8,)\n"
        "    steps = 2\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = snn.Dense(4)\n"
        "        self.lif = snn.LIF()\n"
        "    def forward(self, x):\n"
        "        return self.lif(self.fc(x))\n",
        encoding="utf-8")

    r = snn.compile_file(str(src), passes=[])
    assert r.graph.name == "Only"
    assert r.graph.temporal.steps == 2


def test_cli_refuses_to_guess_between_two_subclasses(tmp_path):
    src = tmp_path / "two.py"
    src.write_text(
        "import snndsl as snn\n"
        "class A(snn.Module):\n"
        "    input_shape = (4,)\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = snn.Dense(2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
        "class B(A):\n"
        "    pass\n",
        encoding="utf-8")

    with pytest.raises(SNNError, match=r"2 snndsl\.Module subclasses"):
        snn.compile_file(str(src), passes=[])


def test_an_example_file_compiles_from_the_cli():
    from pathlib import Path
    ex = Path(__file__).resolve().parent.parent / "examples" / "mnist_forward.py"
    if not ex.exists():
        pytest.skip("example not present")
    r = snn.compile_file(str(ex), passes=[])
    assert r.graph.stats()["params"] == 203_530
