"""Optimisation passes.

Each pass is deliberately *measurable*: it returns before/after metrics so the
compile report can show a real delta instead of a claim. Passes never change
program semantics unless explicitly enabled (``direct_input`` is opt-in because
it swaps rate coding for direct-current injection).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List

from ..ir import IRGraph, IRNode


@dataclass
class PassReport:
    name: str
    changed: bool = False
    before: Dict[str, Any] = field(default_factory=dict)
    after: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pass": self.name, "changed": self.changed,
            "before": self.before, "after": self.after,
            "notes": self.notes, "details": self.details,
        }


class Pass:
    name = "pass"
    description = ""
    opt_in = False

    def run(self, graph: IRGraph) -> PassReport:  # pragma: no cover - interface
        raise NotImplementedError


def activation_bytes(graph: IRGraph, dtype_bytes: int = 4) -> int:
    """Total bytes if every node gets its own buffer (no reuse)."""
    total = 0
    for n in graph.nodes:
        if n.op in ("output",):
            continue
        total += n.numel() * dtype_bytes
    return total


def live_peak_bytes(graph: IRGraph, dtype_bytes: int = 4) -> Dict[str, int]:
    """Compute no-reuse total, theoretical peak under liveness, planned peak."""
    order = {n.id: i for i, n in enumerate(graph.nodes)}
    last_use: Dict[str, int] = {}
    for n in graph.nodes:
        for inp in n.inputs:
            last_use[inp] = max(last_use.get(inp, -1), order[n.id])
    # interval for each node's output value
    intervals: List[tuple] = []
    for n in graph.nodes:
        if n.op == "output":
            continue
        start = order[n.id]
        end = last_use.get(n.id, start)
        intervals.append((start, end, n.id, n.numel() * dtype_bytes))

    # theoretical peak: sweep positions, add at start, drop after last use
    events: Dict[int, List[tuple]] = {}
    for start, end, nid, size in intervals:
        events.setdefault(start, []).append(("open", size, nid))
        events.setdefault(end + 1, []).append(("close", size, nid))
    live = 0
    peak = 0
    for pos in sorted(events):
        for kind, size, _ in events[pos]:
            live += size if kind == "open" else -size
        peak = max(peak, live)

    # greedy interval colouring -> physical slots
    slots: List[int] = []      # slot -> occupied-until position
    slot_size: List[int] = []
    assigned: Dict[str, int] = {}
    for start, end, nid, size in sorted(intervals):
        placed = False
        for idx, occupied_until in enumerate(slots):
            if occupied_until < start:
                slots[idx] = end
                slot_size[idx] = max(slot_size[idx], size)
                assigned[nid] = idx
                placed = True
                break
        if not placed:
            slots.append(end)
            slot_size.append(size)
            assigned[nid] = len(slots) - 1
    for n in graph.nodes:
        if n.id in assigned:
            n.meta["buffer"] = assigned[n.id]

    return {
        "no_reuse_bytes": activation_bytes(graph, dtype_bytes),
        "theoretical_peak_bytes": peak,
        "planned_bytes": sum(slot_size),
        "slots": len(slots),
    }


class DeadCodeElimination(Pass):
    name = "dce"
    description = "remove nodes unreachable from the declared outputs"

    def run(self, graph: IRGraph) -> PassReport:
        rep = PassReport(self.name)
        rep.before = {"nodes": len(graph.nodes)}
        live = set()
        stack = list(graph.outputs)
        prod = graph.producers()
        while stack:
            nid = stack.pop()
            if nid in live:
                continue
            live.add(nid)
            node = prod.get(nid)
            if node is not None:
                stack.extend(node.inputs)
        dead = [n for n in graph.nodes if n.id not in live]
        if dead:
            graph.nodes = [n for n in graph.nodes if n.id in live]
            rep.changed = True
            rep.notes.append("removed " + ", ".join(f"{n.id}({n.op})" for n in dead))
        rep.after = {"nodes": len(graph.nodes)}
        rep.details = {"removed": [n.id for n in dead]}
        return rep


class OperatorFusion(Pass):
    name = "fuse"
    description = "fuse flatten into dense, fold dropout for inference targets"
    opt_in = False

    def run(self, graph: IRGraph) -> PassReport:
        rep = PassReport(self.name)
        rep.before = {"nodes": len(graph.nodes)}
        fused_flatten: List[str] = []
        removed_dropout: List[str] = []
        prod = graph.producers()
        drop_ids = set()

        for node in list(graph.nodes):
            if node.op != "dense" or not node.inputs:
                continue
            src = prod.get(node.inputs[0])
            if src is None or src.op != "flatten":
                continue
            if len(graph.consumers(src.id)) != 1:
                continue
            # dense already flattens its input; the reshape node is pure overhead
            node.inputs = list(src.inputs)
            node.attrs["flatten_input"] = True
            drop_ids.add(src.id)
            fused_flatten.append(f"{src.id}->{node.id}")

        for node in list(graph.nodes):
            if node.op == "dropout":
                drop_ids.add(node.id)
                removed_dropout.append(node.id)
                # reconnect consumers to the dropout's input
                for c in graph.nodes:
                    c.inputs = [node.inputs[0] if i == node.id else i for i in c.inputs]

        if drop_ids:
            graph.nodes = [n for n in graph.nodes if n.id not in drop_ids]
            rep.changed = True
        if fused_flatten:
            rep.notes.append(f"flatten folded into dense: {len(fused_flatten)} site(s)")
        if removed_dropout:
            rep.notes.append(f"dropout elided for inference: {len(removed_dropout)}")
        rep.after = {"nodes": len(graph.nodes)}
        rep.details = {"flatten_fused": fused_flatten,
                       "dropout_elided": removed_dropout}
        return rep


class DirectInputCoding(Pass):
    """Replace rate-encoding of the first layer by direct current injection.

    Saves one encode pass over the whole input and (for static images) removes a
    redundant stochastic/rate conversion. Opt-in because it changes numerics.
    """
    name = "direct_input"
    description = "inject analog input currents instead of rate-coded spikes"
    opt_in = True

    def run(self, graph: IRGraph) -> PassReport:
        rep = PassReport(self.name)
        encs = [n for n in graph.nodes if n.op == "encode"]
        rep.before = {"encode_nodes": len(encs)}
        if not encs:
            rep.after = rep.before
            rep.notes.append("no encoder to fold")
            return rep
        drop = set()
        for enc in encs:
            consumers = graph.consumers(enc.id)
            downstream = {"dense", "conv", "neuron", "add", "concat", "pool", "flatten"}
            if len(consumers) == 1 and consumers[0].op in downstream:
                consumers[0].attrs["direct_input"] = True
                for c in graph.nodes:
                    c.inputs = [enc.inputs[0] if i == enc.id else i for i in c.inputs]
                drop.add(enc.id)
        if drop:
            graph.nodes = [n for n in graph.nodes if n.id not in drop]
            rep.changed = True
            rep.notes.append(f"{len(drop)} encoder(s) replaced by direct current injection")
        rep.after = {"encode_nodes": len(graph.nodes) - len(
            [n for n in graph.nodes if n.op != "encode"])}
        rep.details = {"folded": sorted(drop)}
        return rep


class MemoryPlanner(Pass):
    name = "memplan"
    description = "reuse activation buffers by liveness interval colouring"

    def run(self, graph: IRGraph) -> PassReport:
        rep = PassReport(self.name)
        stats = live_peak_bytes(graph, 4)
        rep.before = {"no_reuse_bytes": stats["no_reuse_bytes"]}
        rep.after = {"planned_bytes": stats["planned_bytes"],
                     "theoretical_peak_bytes": stats["theoretical_peak_bytes"],
                     "slots": stats["slots"]}
        saved = stats["no_reuse_bytes"] - stats["planned_bytes"]
        rep.changed = saved > 0
        if stats["no_reuse_bytes"]:
            rep.notes.append(
                f"activation memory {stats['no_reuse_bytes']} -> "
                f"{stats['planned_bytes']} B "
                f"(-{100.0 * saved / stats['no_reuse_bytes']:.1f}%) "
                f"using {stats['slots']} slot(s)")
        rep.details = stats
        return rep


class TemporalSchedule(Pass):
    name = "schedule"
    description = "choose unrolled vs looped time-step execution"

    THRESHOLD = 4

    def run(self, graph: IRGraph) -> PassReport:
        rep = PassReport(self.name)
        t = graph.temporal.steps
        layers = len([n for n in graph.nodes if n.op in ("dense", "conv", "neuron")])
        rep.before = {"strategy": graph.temporal.strategy}
        graph.temporal.strategy = "unrolled" if t <= self.THRESHOLD else "looped"
        rep.after = {"strategy": graph.temporal.strategy,
                     "python_loop_iters": 0 if t <= self.THRESHOLD else t,
                     "layer_steps": layers * t}
        rep.changed = True
        rep.notes.append(
            f"T={t} <={'<=' if t <= self.THRESHOLD else '>'} {self.THRESHOLD} -> "
            f"{graph.temporal.strategy} (saves {layers * t} loop iterations)"
            if t <= self.THRESHOLD else
            f"T={t} > {self.THRESHOLD} -> looped (keeps code size at {layers} layers "
            f"instead of {layers * t})")
        return rep


ALL_PASSES: List[Callable[[], Pass]] = [
    DeadCodeElimination, OperatorFusion, MemoryPlanner, TemporalSchedule,
]
OPT_IN_PASSES: Dict[str, Callable[[], Pass]] = {
    "direct_input": DirectInputCoding,
}
