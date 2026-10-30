"""Recover the time loop that a `.nir` file is not able to hold.

Why this module exists
----------------------
NIR is a *single-timestep* format.  Its nodes describe ``dx/dt = f(x, u)`` and a
discrete jump; nothing in the vocabulary says "run this T times".  That is a
sound decision for an exchange format, and it has a consequence nobody wrote
down: the loop lives in the *producer's* Python, so the file alone does not say
whether it is one step or T steps unrolled.

Both export shapes exist in the wild and they are indistinguishable without
analysis:

``forward(x, state)`` -- one step
    The graph holds one block.  Execute it T times and the membrane carries
    over.  This is the shape ``nirtorch``'s own docstring describes.

``forward(x)`` with ``for t in range(T)`` inside -- the snnTorch tutorial shape
    The tracer follows the loop, so the graph holds T copies of the block.  The
    copies are *not* chained: the recurrence between step t and t+1 lived in the
    Python variable that held the membrane, and a NIR edge cannot express it, so
    the copies all hang off the input in parallel.

The second shape is not wrong.  It is expensive, and the price is easy to miss.
Every copy sees the same constant input and starts from the same zero state, so
every copy replays the *entire* T-step simulation.  Only the last one feeds
``Output``; the other T-1 are computed and thrown away.  Reading ``Output`` once
per execution does reproduce the trained trace -- at T^2 block evaluations for a
network whose cost should be T.

Measured on a two-layer LIF net (``tests/test_nir_fold.py``), T=4:

    unrolled   18 nodes   4 copies x 4 nodes   block evaluations: 16
    folded      6 nodes   1 block              block evaluations:  4
    spike trace of the last LIF, all 4 steps: identical to PyTorch, differ 0

A 4x factor at T=4, and the factor *is* T, so T=100 costs 100x.  For a format
whose whole selling point is energy per inference, that is the difference
between the claim and its opposite.  Folding restores the loop and writes T into
``metadata``, which is the only place a `.nir` can carry it.

Nothing is approximated: the copies are compared parameter by parameter, and
that comparison is the permission to fold.

What this is not
----------------
Not a size optimisation for its own sake.  Two structurally identical copies
carrying different weights are a real unrolled network, not a loop, and they are
refused -- `detect` reports why instead of silently picking the first copy.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..errors import SNNError

# Fields that describe the *interface* of a node rather than its computation.
_SKIP = {"metadata", "input_type", "output_type"}


class FoldError(SNNError):
    """The graph cannot be folded, with the reason a user can act on."""


@dataclass
class Period:
    """What ``detect`` found: how many copies of how big a block."""

    repeats: int = 1
    period: int = 0                 # nodes per block, excluding Input/Output
    nodes: int = 0
    identical: bool = False         # every copy carries bit-identical parameters
    chained: bool = False           # copies wired in series, i.e. the loop is in the file
    reason: str = ""                # why it is not foldable, when it is not
    inputs: List[str] = field(default_factory=list)
    body: List[str] = field(default_factory=list)     # first copy only
    order: List[str] = field(default_factory=list)    # every copy, copy-major
    components: int = 0                              # disjoint pieces found
    dropped_outputs: int = 0                         # dangling copies written as Outputs

    @property
    def foldable(self) -> bool:
        return self.repeats > 1 and self.identical and bool(self.body)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "repeats": self.repeats,
            "period": self.period,
            "nodes": self.nodes,
            "identical": self.identical,
            "chained": self.chained,
            "foldable": self.foldable,
            "components": self.components,
            "dropped_outputs": self.dropped_outputs,
            "reason": self.reason,
        }

    def summary(self) -> str:
        if self.foldable:
            shape = "chained" if self.chained else "parallel, no state edge"
            return (f"{self.repeats} identical copies of a {self.period}-node "
                    f"block, {shape}: {self.nodes} -> {self.period + 2} nodes")
        return self.reason or "single block, nothing to fold"


def _hash(value: Any) -> str:
    """Stable digest of a node parameter, tolerant of None and nested arrays."""
    if value is None:
        return "-"
    arr = np.asarray(value)
    if arr.dtype == object:
        return hashlib.sha256(repr(value).encode()).hexdigest()[:16]
    # Round before hashing: bit equality on raw float32 is stricter than the
    # question being asked ("is this the same parameter, copied?").
    return hashlib.sha256(np.round(arr.astype(np.float64), 6).tobytes()).hexdigest()[:16]


def signature(node: Any) -> Tuple[Any, ...]:
    """Everything that decides what the node computes, and nothing else."""
    parts: List[Any] = [type(node).__name__]
    for name, value in sorted(vars(node).items()):
        if name in _SKIP:
            continue
        parts.append((name, _hash(value)))
    out = getattr(node, "output_type", None)
    if isinstance(out, dict) and out:
        parts.append(("out_shape",
                      tuple(int(np.asarray(v).reshape(-1)[0]) for v in out.values())))
    return tuple(parts)


def _topo(nodes: Dict[str, Any], edges: Sequence[Tuple[str, str]]) -> List[str]:
    incoming: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        if a in incoming and b in incoming:
            incoming[b].append(a)
    order: List[str] = []
    seen: Dict[str, int] = {}

    def visit(n: str) -> None:
        if n in seen:
            return
        seen[n] = 1
        for p in incoming[n]:
            visit(p)
        order.append(n)

    for n in nodes:
        visit(n)
    return order


def _components(body: Sequence[str],
                edges: Sequence[Tuple[str, str]]) -> List[List[str]]:
    """Weakly connected pieces of the body, in order of first appearance.

    An unchained time-unroll is not one component: step t and step t+1 share no
    edge, so every copy falls into its own piece.  That makes the copies
    findable without trusting node names or a particular topological order --
    both of which change when the graph is written and read back (HDF5 does not
    preserve edge order, and the DFS order silently becomes layer-major).
    """
    pos = {n: i for i, n in enumerate(body)}
    parent = list(range(len(body)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in pos and b in pos:
            ra, rb = find(pos[a]), find(pos[b])
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    groups: Dict[int, List[str]] = {}
    for i, n in enumerate(body):
        groups.setdefault(find(i), []).append(n)
    return list(groups.values())


def detect(graph: Any, max_period: Optional[int] = None) -> Period:
    """Find the smallest p such that the body is k identical copies of p nodes."""
    nodes: Dict[str, Any] = dict(graph.nodes)
    edges = list(graph.edges)
    order = _topo(nodes, edges)

    ins = [n for n in order if type(nodes[n]).__name__ == "Input"]
    outs = [n for n in order if type(nodes[n]).__name__ == "Output"]
    body = [n for n in order if n not in ins and n not in outs]
    res = Period(nodes=len(nodes), inputs=ins, order=list(body))
    if not body:
        res.reason = "graph has no computational nodes"
        return res

    comps = _components(body, edges)
    if len(comps) > 1 and len({len(c) for c in comps}) == 1:
        # copy-major: every component is one repetition, in its own topo order
        body = [n for c in comps for n in c]
        res.order = list(body)
        res.components = len(comps)

    sig = {n: signature(nodes[n]) for n in body}
    limit = max_period or max(1, len(body) // 2)
    for p in range(1, limit + 1):
        if len(body) % p:
            continue
        k = len(body) // p
        if k < 2:
            continue
        if any(sig[body[i]] != sig[body[i % p]] for i in range(len(body))):
            continue
        if not _same_wiring(body, p, edges):
            continue
        res.repeats, res.period = k, p
        res.identical = True
        res.chained = _is_chained(body, p, edges)
        res.body = body[:p]
        return res

    res.reason = ("body is not a repetition of a smaller block: no p divides "
                  "the graph with matching node parameters")
    return res


def _copy_of(name: str, body: Sequence[str], p: int) -> int:
    return body.index(name) // p


def _same_wiring(body: Sequence[str], p: int, edges: Sequence[Tuple[str, str]]) -> bool:
    pos = {n: i for i, n in enumerate(body)}

    def wiring(copy: int) -> set:
        return {((pos[a] - copy * p), (pos[b] - copy * p))
                for a, b in edges
                if a in pos and b in pos
                and pos[a] // p == copy and pos[b] // p == copy}

    base = wiring(0)
    return all(wiring(c) == base for c in range(1, len(body) // p))


def _is_chained(body: Sequence[str], p: int, edges: Sequence[Tuple[str, str]]) -> bool:
    """True when copy i feeds copy i+1 -- i.e. the file does encode the loop."""
    pos = {n: i for i, n in enumerate(body)}
    return any(a in pos and b in pos and pos[b] // p == pos[a] // p + 1
               for a, b in edges)


def _rel(name: str, body: Sequence[str], p: int) -> int:
    """Position inside its copy, for any node of the unrolled body."""
    return body.index(name) % p


def fold(graph: Any) -> Tuple[Any, Period]:
    """Return ``(one_block_graph, period)``; raise when folding is not provable."""
    import nir

    per = detect(graph)
    if not per.foldable:
        raise FoldError(
            "cannot fold this graph: " + (per.reason or per.summary())
            + ". Folding is only done when every copy carries identical "
              "parameters; a real unrolled network keeps its copies."
        )

    all_body = list(per.order or detect_body(graph))
    p = per.period
    first = per.body
    inputs = set(per.inputs)
    all_outs = [n for n in graph.nodes if type(graph.nodes[n]).__name__ == "Output"]

    # An unchained unroll leaves T-1 copies dangling. nir.write materialises one
    # Output per dangling node, so a written-and-read file declares T outputs --
    # and every consumer that expects one rejects it. Keep the one fed by the
    # last copy, which is the only one that carries the answer.
    def copy_index(out: str) -> int:
        srcs = [a for a, b in graph.edges if b == out and a in all_body]
        return max((_copy_of(s, all_body, p) for s in srcs), default=-1)

    outs = {max(all_outs, key=copy_index)} if all_outs else set()
    extra = [o for o in all_outs if o not in outs]

    keep = set(first) | inputs | outs
    nodes = {n: graph.nodes[n] for n in graph.nodes if n in keep}
    per.dropped_outputs = len(extra)

    edges: List[Tuple[str, str]] = []
    for a, b in graph.edges:
        if b in outs:
            # the unrolled graph feeds Output from the last copy; point it at the
            # same position in the first copy, which is the one that survives.
            if a in all_body:
                edges.append((first[_rel(a, all_body, p)], b))
        elif a in inputs and b in all_body:
            if _copy_of(b, all_body, p) == 0:
                edges.append((a, b))
        elif a in all_body and b in all_body:
            if _copy_of(a, all_body, p) == 0 and _copy_of(b, all_body, p) == 0:
                edges.append((a, b))

    meta = dict(getattr(graph, "metadata", {}) or {})
    # A .nir has no field for "run me T times". metadata is the only slot the
    # format leaves open, so the folded file states it there and nowhere else.
    # Strings on purpose: nir.write stores metadata as HDF5 datasets and
    # rejects None, and a plain int round-trips less predictably than str.
    meta["snndsl.timesteps"] = str(per.repeats)
    meta["snndsl.folded_from"] = str(per.nodes)
    merged = nir.NIRGraph(nodes=nodes, edges=edges, metadata=meta, type_check=False)
    return merged, per


_NEURON_PARAMS = ("tau", "r", "v_leak", "v_threshold", "v_reset", "tau_mem", "tau_syn")


def promote_params(graph: Any) -> List[str]:
    """Expand size-1 neuron parameters to one value per neuron.

    ``nir.LIF.__post_init__`` derives ``input_type`` from the parameter shape, so
    a LIF written with ``tau=np.array([9.5])`` *declares a one-neuron layer* no
    matter what feeds it.  ``nirtorch`` 2.6 writes exactly that for every layer,
    which makes the file fail NIR's own verifier on read:

        Type inference error: type mismatch: fc1.output: [[8]] -> lif1.input: [[1]]

    The arithmetic was never wrong -- a scalar broadcasts -- but the declared
    type is, and the type is what the verifier and every consumer read.  So the
    repair is at the parameter: repeat the value to the width of the activation.
    """
    producer = {b: a for a, b in graph.edges}
    fixed: List[str] = []
    for name, node in graph.nodes.items():
        src = producer.get(name)
        if src is None or src not in graph.nodes:
            continue
        want = _shapes(graph.nodes[src], "output_type")
        if not want:
            continue
        decl = [np.asarray(v) for v in want.values() if v is not None]
        if not decl or any(d.ndim != 1 for d in decl):
            # A conv activation is (C, H, W); its shape array is 3-D, and
            # collapsing it to channel-times-area would declare the wrong rank,
            # so leave it for a target that knows the layout.
            continue
        # ``output_type`` stores a *shape*, not a value: np.array([8]) is an
        # 8-wide layer, so the neuron count is the product of the shape.
        width = int(max(np.prod(d) for d in decl))
        if width <= 1:
            continue
        changed = False
        for pname in _NEURON_PARAMS:
            cur = getattr(node, pname, None)
            if cur is None:
                continue
            arr = np.asarray(cur)
            if arr.size == 1:
                setattr(node, pname, np.full(width, arr.reshape(-1)[0], dtype=arr.dtype))
                changed = True
        if changed:
            node.input_type = {"input": np.array([width])}
            node.output_type = {"output": np.array([width])}
            fixed.append(name)
    return fixed


def _shapes(node: Any, field: str) -> Dict[str, Any]:
    d = getattr(node, field, None)
    return d if isinstance(d, dict) else {}


def normalize_types(graph: Any) -> List[str]:
    """Make declared node types agree with the edges, which is what NIR checks.

    Two separate upstream defects make the most common producer unusable, and
    both are shape mismatches rather than logic errors:

    ``nirtorch`` 2.6 leaves ``Output.input_type = {'input': None}``
        ``nir.write`` 1.0.8 then dies inside h5py: ``TypeError: One of data,
        shape or dtype must be specified``.  The producer cannot save its own
        output.

    ``nirtorch`` writes neuron parameters as size-1 arrays
        ``nir.LIF.__post_init__`` derives ``input_type`` from the parameter
        shape, so a LIF behind an 8-wide ``Affine`` declares ``input_type = [1]``
        and ``nir.NIRGraph`` rejects the file on read with ``Type inference
        error: type mismatch: fc1.output: [[8]] -> lif1.input: [[1]]``.  The
        arithmetic is fine -- the parameter broadcasts -- but the declared type
        is not.

    Both are repaired here from the one fact the graph does know: whatever the
    producer emits.  Returns the names touched, so a report can say how much
    fixing the file needed.
    """
    producer = {b: a for a, b in graph.edges}
    touched: List[str] = []

    def fit(name: str) -> None:
        node = graph.nodes[name]
        src = producer.get(name)
        if src is None or src not in graph.nodes:
            return
        want = _shapes(graph.nodes[src], "output_type")
        if not want or all(v is None for v in want.values()):
            return
        have_in = _shapes(node, "input_type")
        if have_in and all(v is not None for v in have_in.values()) and \
                list(have_in.values()) == list(want.values()):
            return
        # key by role, not by copying the producer's dict: its key is 'output'
        node.input_type = {"input": next(iter(want.values()))}
        have_out = _shapes(node, "output_type")
        if not have_out or all(v is None for v in have_out.values()):
            node.output_type = {"output": next(iter(want.values()))}
        touched.append(name)

    promote_params(graph)
    for name in list(graph.nodes):
        if type(graph.nodes[name]).__name__ != "Input":
            fit(name)
    return touched


def fold_file(src: str, dst: Optional[str] = None) -> Tuple[str, Period]:
    """Fold ``src`` and write the result to ``dst`` (default: ``<src>.folded.nir``)."""
    import nir

    graph = nir.read(src)
    folded, per = fold(graph)
    normalize_types(folded)
    out = dst or (src[:-4] + ".folded.nir" if src.endswith(".nir") else src + ".folded.nir")
    nir.write(out, folded)
    return out, per


def save(graph: Any, path: str) -> List[str]:
    """Write a graph after repairing the types a tracer left inconsistent."""
    import nir

    touched = normalize_types(graph)
    nir.write(path, graph)
    return touched


def detect_body(graph: Any) -> List[str]:
    """Body node names in topological order (Input/Output excluded)."""
    nodes = dict(graph.nodes)
    order = _topo(nodes, list(graph.edges))
    return [n for n in order
            if type(nodes[n]).__name__ not in ("Input", "Output")]
