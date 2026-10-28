"""SNN IR: a typed, topologically ordered dataflow graph with explicit time.

Design notes
------------
* Values are SSA-like: each node defines exactly one value id.  The DSL lets a
  user *rebind* a name (``dense h1`` then ``neuron h1``), and the frontend
  resolves that into a fresh value id, which keeps the IR a clean DAG.
* Shapes are channel-first for spatial ops ``(C, H, W)`` and 1-D for feature
  vectors ``(F,)``.  Batch (B) and time (T) are implicit and added by backends,
  so a compiled graph is batch/time independent.
* ``attrs`` carries op-specific parameters; ``meta`` carries compiler statistics
  (sparsity, buffer assignment, quantisation scales) filled in by passes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

NodeShape = Tuple[int, ...]


@dataclass
class NeuronSpec:
    """Fully-resolved neuron model (all params folded to numbers)."""
    name: str
    params: Dict[str, float] = field(default_factory=dict)
    states: Dict[str, float] = field(default_factory=dict)
    body: List[Tuple[str, str, str]] = field(default_factory=list)
    # ^ [(kind, target, python_expr_source)] with kind in {'assign','spike'}
    spike_var: str = "s"
    surrogate: str = "fast_sigmoid"
    surrogate_alpha: float = 4.0
    reset: str = "subtract"

    def state_names(self) -> List[str]:
        return list(self.states.keys())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TemporalSpec:
    steps: int = 8
    dt: float = 1.0
    reset: str = "subtract"
    # compiler decision: unrolled | looped  (set by the temporal pass)
    strategy: str = "looped"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IRNode:
    id: str
    op: str
    inputs: List[str] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    shape: NodeShape = ()
    dtype: str = "float32"
    name: str = ""          # user-visible name (may repeat after rebinding)
    line: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= int(d)
        return n

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


@dataclass
class IRGraph:
    name: str = "snn"
    nodes: List[IRNode] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    temporal: TemporalSpec = field(default_factory=TemporalSpec)
    neurons: Dict[str, NeuronSpec] = field(default_factory=dict)
    encoders: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    infer: Dict[str, Any] = field(default_factory=dict)
    budgets: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    filename: str = "<snn>"

    # ------------------------------------------------------------- accessors
    def by_id(self) -> Dict[str, IRNode]:
        return {n.id: n for n in self.nodes}

    def node(self, nid: str) -> IRNode:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    def add(self, node: IRNode) -> IRNode:
        if any(n.id == node.id for n in self.nodes):
            raise ValueError(f"duplicate IR value id {node.id!r}")
        self.nodes.append(node)
        return node

    def consumers(self, nid: str) -> List[IRNode]:
        return [n for n in self.nodes if nid in n.inputs]

    def producers(self) -> Dict[str, IRNode]:
        return {n.id: n for n in self.nodes}

    def stats(self) -> Dict[str, Any]:
        params = 0
        macs = 0
        for n in self.nodes:
            if n.op == "dense":
                macs += n.attrs.get("in_features", 0) * n.attrs.get("units", 0)
                params += n.attrs.get("in_features", 0) * n.attrs.get("units", 0) \
                    + n.attrs.get("units", 0)
            elif n.op == "conv":
                cin = n.attrs.get("in_channels", 0)
                cout = n.attrs.get("out_channels", 0)
                k = n.attrs.get("kernel", 3)
                hw = 1
                for d in n.shape[1:]:
                    hw *= int(d)
                macs += cin * cout * k * k * hw
                params += cin * cout * k * k + cout
        return {
            "nodes": len(self.nodes),
            "params": params,
            "macs_per_step": macs,
            "macs_total": macs * self.temporal.steps,
            "time_steps": self.temporal.steps,
        }

    # ------------------------------------------------------------ (de)serial
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "temporal": self.temporal.to_dict(),
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": [n.to_dict() for n in self.nodes],
            "neurons": {k: v.to_dict() for k, v in self.neurons.items()},
            "encoders": self.encoders,
            "train": self.train,
            "infer": self.infer,
            "budgets": self.budgets,
            "meta": self.meta,
            "stats": self.stats(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # --------------------------------------------------------------- display
    def render_text(self) -> str:
        rows = ["id".ljust(22) + "op".ljust(10) + "shape".ljust(18) + "inputs"]
        rows.append("-" * 78)
        for n in self.nodes:
            rows.append(
                n.id.ljust(22) + n.op.ljust(10) +
                str(list(n.shape)).ljust(18) +
                ",".join(n.inputs)
            )
        return "\n".join(rows)


# ------------------------------------------------------------------ helpers
ELEMWISE_OPS = {"neuron", "dropout"}
SPATIAL_OPS = {"conv", "pool", "flatten"}
LINEAR_OPS = {"dense"}
MERGE_OPS = {"add", "concat"}
IO_OPS = {"input", "output", "encode", "decode"}

NEURON_FREE_OPS = IO_OPS | LINEAR_OPS | SPATIAL_OPS | MERGE_OPS | ELEMWISE_OPS
