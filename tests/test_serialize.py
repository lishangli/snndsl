"""`.snn` is the IR's text form, not a language the IR is compiled from.

That sentence is the whole design argument, and it is only worth anything if it
survives being checked. So every test here does a round trip and compares the
resulting op sequence -- not the text, which is allowed to be prettier, and not
the node ids, which are derived from layer names and legitimately differ
(`h1` in source, `h1$1` in the IR).

The two claims under test:

* ``compile(to_snn(compile(src)))`` has the same ops, shapes and blocks as
  ``compile(src)``, for every example in the repository;
* a graph built from Python (loops, parameters, helper functions) can be
  printed to ``.snn`` and read back, which is what makes the two front ends
  alternatives rather than stages of a pipeline.
"""
from __future__ import annotations

import glob
import os

import pytest
from snndsl import LIF, Net
from snndsl.errors import SNNError
from snndsl.ir import IRGraph, NeuronSpec
from snndsl.pipeline import compile_source
from snndsl.serialize import to_snn

EXAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..",
                                         "examples", "*.snn")))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ops(graph):
    return [n.op for n in graph.nodes]


def _shapes(graph):
    return [tuple(n.shape) for n in graph.nodes]


def _roundtrip(src: str):
    first = compile_source(src).graph
    text = to_snn(first)
    second = compile_source(text).graph
    return first, second, text


@pytest.mark.parametrize("path", EXAMPLES, ids=os.path.basename)
def test_every_example_roundtrips(path):
    """Ten examples, including add / ALIF / max-pool, must survive printing."""
    src = open(path, encoding="utf-8").read()
    first, second, _ = _roundtrip(src)
    assert _ops(first) == _ops(second)
    assert _shapes(first) == _shapes(second)
    assert first.temporal.steps == second.temporal.steps
    assert first.temporal.reset == second.temporal.reset
    assert first.infer.get("readout") == second.infer.get("readout")


def test_printed_source_has_no_internal_ids():
    """`h1$1` is an IR id; it must never leak into a file a human reads."""
    _, _, text = _roundtrip(open(os.path.join(HERE, "examples", "mnist_conv.snn"),
                                 encoding="utf-8").read())
    assert "$" not in text


def test_spike_line_is_a_comparison_again():
    """The IR stores `surrogate_fast_sigmoid((v) - (v_th), 4)`; source wants `v > v_th`."""
    _, _, text = _roundtrip(open(os.path.join(HERE, "examples", "mnist_hw.snn"),
                                 encoding="utf-8").read())
    assert "spike s = v > v_th" in text
    assert "surrogate_fast_sigmoid((" not in text


def test_python_front_end_prints_and_reloads():
    """A loop-generated network has no .snn original, so print one and read it."""
    net = Net("from_python", steps=8, readout="membrane")
    x = net.encode(net.input(1, 28, 28), "rate", gain=1.0)
    h = x
    for i in range(3):
        h = net.neuron(net.dense(h, 64, scale=2.0, name=f"h{i + 1}"),
                       LIF(tau=2.0), name=f"a{i + 1}")
    net.output(net.neuron(net.dense(h, 10, name="y"), LIF(), name="ay"))
    net.train(data="mnist", epochs=4, batch=128, lr=1e-3)

    graph = net.compile(quiet=True).graph
    text = net.to_snn(comment="three blocks, generated in a loop")
    second = compile_source(text).graph

    assert _ops(graph) == _ops(second)
    assert len(second.nodes) == 11
    assert second.temporal.steps == 8
    assert second.infer.get("readout") == "membrane"
    assert second.train["epochs"] == 4
    # the comment survives as a comment, not as a directive
    assert text.startswith("# three blocks")


def test_two_front_ends_accept_the_same_checkpoint(tmp_path):
    """Same network, same module: a .pt trained from the text version loads here.

    This is the practical meaning of "one IR, two spellings". If the parameter
    names differed the two files would merely look alike.
    """
    from snndsl.pipeline import compile_file, load_module

    text_res = compile_file(os.path.join(HERE, "examples", "mnist_hw.snn"))
    py_res = compile_file(os.path.join(HERE, "examples", "mnist_hw_py.py"))
    text_net = load_module(text_res, str(tmp_path / "a.py"))()
    py_net = load_module(py_res, str(tmp_path / "b.py"))()

    a = {k: tuple(v.shape) for k, v in text_net.state_dict().items()}
    b = {k: tuple(v.shape) for k, v in py_net.state_dict().items()}
    assert a == b
    assert _ops(text_res.graph) == _ops(py_res.graph)


def test_lif_reset_reaches_the_body_not_just_the_header():
    """`reset=` must change the update rule; a header alone is what deploy blocks."""
    assert LIF().body[-1] == "v = v - s * v_th"
    assert LIF(reset="zero").body[-1] == "v = v * (1 - s)"
    assert LIF(reset="zero").reset == "zero"
    with pytest.raises(SNNError):
        LIF(reset="membrane")


def test_zero_reset_python_network_is_deployable():
    """The .py twin of mnist_hw must clear the same static check as the .snn one."""
    from snndsl.deploy import analyze
    from snndsl.pipeline import compile_file

    res = compile_file(os.path.join(HERE, "examples", "mnist_hw_py.py"))
    rep = analyze(res.graph)
    assert rep.deployable, [f.message for f in rep.blockers()]


def test_unprintable_spike_condition_raises_instead_of_guessing():
    """A straight-through predicate has no `a > b` form; say so, do not invent one."""
    graph = IRGraph(name="handmade")
    graph.neurons["ODD"] = NeuronSpec(
        name="ODD",
        params={"v_th": 1.0},
        states={"v": 0.0},
        body=[("assign", "v", "(v + i)"),
              ("spike", "s", "surrogate_straight_through((v > v_th), 4.0)")],
    )
    with pytest.raises(SNNError) as err:
        to_snn(graph)
    assert "cannot recover the spike condition" in str(err.value)


def test_emit_cli_writes_the_same_text(tmp_path, capsys):
    """`snndsl emit` is the same function, so it cannot drift from the API."""
    from snndsl.cli import main

    out = tmp_path / "norm.snn"
    assert main(["emit", os.path.join(HERE, "examples", "mnist_hw.snn"),
                 "-o", str(out)]) == 0
    printed = out.read_text(encoding="utf-8")
    api = to_snn(compile_source(
        open(os.path.join(HERE, "examples", "mnist_hw.snn"), encoding="utf-8").read()
    ).graph)
    assert printed == api
    captured = capsys.readouterr()
    assert "nodes" in captured.out
