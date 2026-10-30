"""Print an IR graph back to ``.snn`` text.

Why this exists
---------------
``.snn`` is not a programming language. It has constants, arithmetic, blocks
and a topology list -- no functions, no loops, no types. It is one *printing*
of the IR, and ``snn.Net`` is another. Both call the same
``build_neuron_spec`` / ``_emit`` path, so neither can drift into its own
semantics.

Making the round trip explicit settles the question "why not just use Python?":
because ``.snn`` is the serialisation format, and Python is the front end. You
write networks in Python (loops, parameters, helper functions), and ``.snn``
keeps only what a text file is genuinely better at -- being diffed, reviewed,
and read by a tool that must not execute your code.

Two facts this module had to work around, both visible in the output:

* ``encode`` / ``neuron`` *rebind* the name in place in ``.snn`` source, while
  the IR gives every op its own node id (``h1`` -> ``h1$1``). The printer maps
  back through ``IRNode.name``.
* the spike line is stored already lowered, as ``surrogate_x((v) - (v_th), 4)``.
  Recovering ``v > v_th`` is a pattern match, not a parse -- when it fails the
  printer raises instead of guessing a condition.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from .errors import SNNError
from .ir import IRGraph, IRNode, NeuronSpec

# surrogate_x((lhs) - (rhs), alpha)  <-  this is what lower_spike() produces
_SPIKE_RE = re.compile(
    r"^surrogate_(?P<kind>\w+)\(\((?P<lhs>.+?)\)\s*-\s*\((?P<rhs>.+?)\),\s*"
    r"(?P<alpha>[-+0-9.eE]+)\)$"
)
_IDENT_RE = re.compile(r"^[A-Za-z_]\w*$")

# ops whose .snn form rebinds the incoming name instead of minting a new one
_REBIND_OPS = {"encode", "neuron", "decode"}


def _num(value: Any) -> str:
    """Print a number the way a human would have typed it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    f = float(value)
    return str(int(f)) if f.is_integer() else repr(f)


def _value(value: Any) -> str:
    """Print a config value; quote anything that is not a bare identifier."""
    if isinstance(value, (bool, int, float)):
        return _num(value)
    s = str(value)
    return s if _IDENT_RE.match(s) else f'"{s}"'


def _spike_condition(lowered: str, spec: NeuronSpec) -> str:
    m = _SPIKE_RE.match(lowered)
    if m is None:
        raise SNNError(
            f"cannot recover the spike condition of neuron {spec.name!r} "
            f"from {lowered!r}",
            hint="the IR stores the lowered surrogate call; a straight-through "
                 "or hand-written predicate has no source form to print back")
    return f"{m.group('lhs')} > {m.group('rhs')}"


def _neuron_block(spec: NeuronSpec) -> List[str]:
    out = [f"neuron {spec.name}:"]
    for k, v in spec.params.items():
        out.append(f"    param {k} = {_num(v)}")
    for k, v in spec.states.items():
        out.append(f"    state {k} = {_num(v)}")
    for kind, target, src in spec.body:
        if kind == "spike":
            out.append(f"    spike {target} = {_spike_condition(src, spec)}")
        else:
            out.append(f"    {target} = {src}")
    out.append(f"    surrogate = {spec.surrogate}(alpha={_num(spec.surrogate_alpha)})")
    return out


def _ref(graph: IRGraph, node_id: str, _depth: int = 0) -> str:
    """The name this value carries in .snn source.

    Two translations happen here, both forced by the text syntax:

    * ids get a ``$1`` suffix (``h1$1``), while the source writes ``h1``;
    * ``encode`` / ``neuron`` / ``decode`` rebind their operand's name instead
      of introducing a new one, so their output *is* the input's name.
    """
    try:
        node = graph.node(node_id)
    except KeyError:
        return node_id
    if node.op in _REBIND_OPS and node.inputs and _depth < 32:
        return _ref(graph, node.inputs[0], _depth + 1)
    return node.name or node.id


def _layer_line(node: IRNode, graph: IRGraph) -> str:
    name = node.name or node.id
    a = node.attrs
    src = lambda: _ref(graph, node.inputs[0])  # noqa: E731 - terse in a table

    if node.op == "input":
        dims = a.get("dims") or list(node.shape)
        return f"    input   {name} : {', '.join(str(d) for d in dims)}"
    if node.op == "encode":
        return (f"    encode  {_ref(graph, node.id)} -> {a.get('scheme', 'rate')} "
                f"with gain={_num(a.get('gain', 1.0))}")
    if node.op == "decode":
        return (f"    decode  {_ref(graph, node.id)} -> {a.get('scheme', 'rate')} "
                f"with window={_num(a.get('window', graph.temporal.steps))}")
    if node.op == "dense":
        line = f"    dense   {name} : {a['units']} <- {src()}"
        extra = []
        if float(a.get("weight_scale", 1.0)) != 1.0:
            extra.append(f"scale={_num(a['weight_scale'])}")
        if not a.get("bias", True):
            extra.append("bias=false")
        if a.get("init") and a["init"] != "kaiming_normal":
            extra.append(f"init={a['init']}")
        return line + (f" with {', '.join(extra)}" if extra else "")
    if node.op == "conv":
        line = (f"    conv    {name} : {a['out_channels']} <- {src()} "
                f"with k={a['kernel']}, s={a['stride']}, p={a['padding']}")
        if float(a.get("weight_scale", 1.0)) != 1.0:
            line += f", scale={_num(a['weight_scale'])}"
        return line
    if node.op == "pool":
        line = (f"    pool    {name} : {a['kernel']} <- {src()} "
                f"with kind={a.get('kind', 'avg')}")
        if int(a.get("stride", a["kernel"])) != int(a["kernel"]):
            line += f", stride={a['stride']}"
        return line
    if node.op == "flatten":
        return f"    flatten {name} <- {src()}"
    if node.op == "dropout":
        return f"    dropout {name} : {_num(a.get('p', 0.1))} <- {src()}"
    if node.op == "neuron":
        return f"    neuron  {_ref(graph, node.id)} -> {a['model']}"
    if node.op == "add":
        return f"    add     {name} : {', '.join(_ref(graph, i) for i in node.inputs)}"
    if node.op == "concat":
        refs = ", ".join(_ref(graph, i) for i in node.inputs)
        return f"    concat  {name} : {refs} with axis={a.get('axis', 1)}"
    if node.op == "output":
        line = f"    output  {name} <- {src()}"
        readout = graph.infer.get("readout")
        if readout:
            line += f" with readout={readout}"
        return line
    raise SNNError(f"cannot print op {node.op!r} as .snn source",
                   hint=f"node {node.id!r}; add a branch to serialize._layer_line")


def to_snn(graph: IRGraph, comment: str = "") -> str:
    """Render ``graph`` as ``.snn`` source.

    The result recompiles to the same op sequence; node *ids* may differ
    because ``.snn`` derives them from layer names (``h1`` -> ``h1$1``).
    """
    lines: List[str] = []
    if comment:
        for ln in comment.strip().splitlines():
            lines.append(f"# {ln}".rstrip())
        lines.append("")

    lines.append(f"program {graph.name}")
    lines.append("")

    t = graph.temporal
    lines.append("temporal:")
    lines.append(f"    steps = {t.steps}")
    lines.append(f"    dt = {_num(t.dt)}")
    lines.append(f"    reset = {t.reset}")
    lines.append("")

    for spec in graph.neurons.values():
        lines.extend(_neuron_block(spec))
        lines.append("")

    for enc_id, enc in graph.encoders.items():
        lines.append(f"encoder {enc_id}:")
        flat = enc.get("params", enc) if isinstance(enc, dict) else {}
        for k, v in flat.items():
            lines.append(f"    {k} = {_value(v)}")
        lines.append("")

    lines.append("pipeline:")
    for node in graph.nodes:
        lines.append(_layer_line(node, graph))
    lines.append("")

    for block_name, block in (("train", graph.train), ("infer", graph.infer),
                              ("budget", graph.budgets)):
        if not block:
            continue
        lines.append(f"{block_name}:")
        for k, v in _ordered(block):
            lines.append(f"    {k} = {_value(v)}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


_TRAIN_ORDER = ["data", "epochs", "batch", "lr", "opt", "weight_decay", "loss",
                "sched", "seed", "subset", "classes", "root", "workers",
                "augment", "val_split"]


def _ordered(block: Dict[str, Any]):
    """Keep keys in the order the examples use, so diffs stay small."""
    known = [(k, block[k]) for k in _TRAIN_ORDER if k in block]
    rest = [(k, v) for k, v in block.items() if k not in _TRAIN_ORDER]
    return known + rest
