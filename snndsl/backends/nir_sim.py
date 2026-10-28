"""A reference interpreter for NIR graphs.

Why this exists
---------------
Exporting a ``.nir`` file only proves that the file is syntactically valid. It
says nothing about whether the graph *means* the same thing as the network the
user trained. The gap between those two statements is where SNN deployment bugs
live: a mis-scaled current gain, a reset rule that differs from the trainer, a
synapse count that silently halves.

This module implements the forward semantics defined by the NIR specification
(Pedersen et al., Nature Communications 15, 8122 (2024)) so that a compiled
graph can be executed and compared against its PyTorch counterpart. It is
deliberately small and readable: it is the oracle, so it must be easy to audit.

Semantics implemented
---------------------
LIF::

    tau * dv/dt = (v_leak - v) + r * i
    v <- v_reset            where  v > v_threshold

discretised with explicit Euler at step size ``dt``::

    i_t = sum of incoming edges at step t
    v_t = v_{t-1} + dt/tau * ((v_leak - v_{t-1}) + r * i_t)
    s_t = 1[v_t > v_threshold]
    v_t = v_reset  where s_t == 1

``rho`` (the NIR per-neuron reset multiplier) is honoured when present:
``v <- rho * v`` after subtraction, which is the form used by some simulators.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class NIRSimError(RuntimeError):
    """Raised when a graph uses a construct this interpreter does not model."""


# --------------------------------------------------------------------- helpers
def _topo_order(nodes: Dict[str, Any],
                edges: Sequence[Tuple[str, str]]) -> List[str]:
    """Kahn's algorithm. NIR graphs are DAGs by construction."""
    indeg = {k: 0 for k in nodes}
    adj: Dict[str, List[str]] = {k: [] for k in nodes}
    for src, dst in edges:
        if src not in nodes or dst not in nodes:
            raise NIRSimError(f"edge {src!r}->{dst!r} references an unknown node")
        adj[src].append(dst)
        indeg[dst] += 1
    queue = [k for k, v in indeg.items() if v == 0]
    order: List[str] = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(nodes):
        raise NIRSimError("graph is not acyclic")
    return order


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(np.asarray(value), dtype=torch.float32)


def _attr(node: Any, name: str, default: Any = None) -> Any:
    """NIR nodes expose parameters as attributes; be forgiving about absence."""
    return getattr(node, name, default)


def _neuron_count(node: Any) -> int:
    """Number of neurons in a node: prefer the declared type, fall back to params."""
    it = _attr(node, "input_type", None)
    if isinstance(it, dict) and it:
        try:
            shape = np.asarray(next(iter(it.values()))).reshape(-1)
            total = int(np.prod(shape[shape > 0])) if shape.size else 0
            if total > 0:
                return total
        except Exception:
            pass
    for pname in ("v_threshold", "v_leak", "tau", "r"):
        p = _attr(node, pname, None)
        if p is not None:
            return int(np.asarray(p).reshape(-1).size)
    raise NIRSimError(f"cannot infer neuron count for {type(node).__name__}")


def _per_neuron(value: Any, n: int, dtype=torch.float32) -> torch.Tensor:
    """NIR stores neuron parameters as per-neuron arrays; accept scalars too."""
    t = _as_tensor(value).to(dtype).reshape(-1)
    if t.numel() == 1 and n != 1:
        return t.expand(n).clone()
    return t


class _NeuronState:
    """Membrane bookkeeping for one spiking node."""

    __slots__ = ("v", "mem", "spk")

    def __init__(self, n: int, v_init: float, dtype=torch.float32):
        self.v = torch.full((n,), float(v_init), dtype=dtype)
        self.mem: List[torch.Tensor] = []
        self.spk: List[torch.Tensor] = []


# ------------------------------------------------------------------ simulator
def simulate(nir_graph: Any,
             inputs: Any,
             steps: Optional[int] = None,
             dt: float = 1.0,
             v_init: float = 0.0,
             record: bool = True,
             is_sequence: bool = False) -> Dict[str, Any]:
    """Run a NIR graph for ``steps`` time steps.

    Parameters
    ----------
    nir_graph
        A ``nir.NIRGraph`` (or anything with ``.nodes`` and ``.edges``).
    inputs
        A single step ``(batch, *shape)``, replayed on every step — this matches a
        deterministic rate-coded current. Pass ``is_sequence=True`` with a
        ``(steps, batch, *shape)`` tensor to drive the graph with a real stream.
    steps
        Number of time steps. Ignored when ``is_sequence`` is set.
    is_sequence
        Explicit flag. It is *not* inferred from the tensor rank on purpose: a
        batch of 8 fed to an 8-step graph is indistinguishable from 8 steps, and
        guessing wrong silently scrambles the timeline.

    Returns
    -------
    dict with
        ``output``   final value at the graph's Output node, ``(batch, features)``
        ``outputs``  per-node list of per-step tensors
        ``spikes``   per-neuron-node list of per-step spike tensors
        ``steps``    number of steps actually run
    """
    nodes = dict(nir_graph.nodes)
    edges = list(nir_graph.edges)
    order = _topo_order(nodes, edges)

    incoming: Dict[str, List[str]] = {k: [] for k in nodes}
    for src, dst in edges:
        incoming[dst].append(src)

    # ---- normalise the input sequence
    x = _as_tensor(inputs)
    if x.dim() == 0:
        raise NIRSimError("inputs must have at least one dimension")
    if is_sequence:
        steps = int(x.shape[0])
        seq = [x[t] for t in range(steps)]
    else:
        steps = int(steps or 1)
        seq = [x] * steps

    # ---- states
    neuron_state: Dict[str, _NeuronState] = {}
    outs: Dict[str, List[torch.Tensor]] = {}

    for name in order:
        node = nodes[name]
        kind = type(node).__name__
        if kind in ("LIF", "LI", "IF", "CubaLIF", "CubaLI"):
            neuron_state[name] = _NeuronState(_neuron_count(node), v_init)
            outs[name] = []

    for t in range(steps):
        values: Dict[str, torch.Tensor] = {}
        for name in order:
            node = nodes[name]
            kind = type(node).__name__
            src_vals = [values[s] for s in incoming[name] if s in values]
            if incoming[name] and len(src_vals) != len(incoming[name]):
                missing = [s for s in incoming[name] if s not in values]
                raise NIRSimError(f"node {name!r} missing inputs from {missing}")

            if kind == "Input":
                values[name] = seq[t]
            elif kind == "Output":
                values[name] = src_vals[0] if src_vals else torch.zeros(1)
            elif kind == "Linear":
                w = _as_tensor(node.weight)
                values[name] = _apply_linear(src_vals[0], w)
            elif kind == "Affine":
                w = _as_tensor(node.weight)
                b = _as_tensor(node.bias)
                values[name] = _apply_linear(src_vals[0], w) + b
            elif kind in ("Flatten",):
                v = src_vals[0]
                start = int(_attr(node, "start_dim", 1))
                end = int(_attr(node, "end_dim", -1))
                values[name] = torch.flatten(v, start, end)
            elif kind in ("Conv1d", "Conv2d"):
                values[name] = _apply_conv(node, src_vals[0])
            elif kind in ("AvgPool2d", "SumPool2d", "MaxPool2d"):
                values[name] = _apply_pool(node, src_vals[0])
            elif kind == "Scale":
                values[name] = src_vals[0] * float(_attr(node, "scale", 1.0))
            elif kind == "Threshold":
                v = src_vals[0]
                values[name] = (v > float(_attr(node, "threshold", 0.0))).to(v.dtype)
            elif kind in ("LIF", "LI", "IF", "CubaLIF", "CubaLI"):
                st = neuron_state[name]
                cur = torch.stack(src_vals).sum(0) if src_vals else torch.zeros_like(st.v)
                v_pre, spk = _step_neuron(node, kind, st, cur, dt)
                # NIR semantics: a spiking node emits *spikes*; a leaky
                # integrator (LI) has no threshold and emits its membrane.
                values[name] = v_pre if kind in ("LI", "CubaLI") else spk
                st.mem.append(v_pre.detach().clone())
                st.spk.append(spk.detach().clone())
            elif kind in ("Delay",):
                raise NIRSimError(
                    "Delay nodes are not supported by this reference interpreter; "
                    "delays change the execution schedule and need a target-specific model")
            else:
                raise NIRSimError(f"unsupported NIR node type {kind!r} at {name!r}")

            if record:
                outs.setdefault(name, []).append(values[name].detach().clone())

    # ---- locate the graph output
    out_names = [n for n in nodes if type(nodes[n]).__name__ == "Output"]
    if len(out_names) != 1:
        raise NIRSimError(f"expected exactly one Output node, found {len(out_names)}")
    final = values[out_names[0]]

    mem_map: Dict[str, torch.Tensor] = {}
    spk_map: Dict[str, torch.Tensor] = {}
    for name, st in neuron_state.items():
        if st.mem:
            mem_map[name] = torch.stack(st.mem)
            spk_map[name] = torch.stack(st.spk)

    return {"output": final, "outputs": outs, "membrane": mem_map,
            "spikes": spk_map, "steps": steps, "node_values": values}


def output_source(nir_graph: Any) -> Optional[str]:
    """Name of the node feeding the graph's Output, or ``None``."""
    for name, node in nir_graph.nodes.items():
        if type(node).__name__ == "Output":
            for a, b in nir_graph.edges:
                if b == name:
                    return a
            break
    return None


def output_stream(nir_graph: Any, sim: Dict[str, Any],
                  mode: str = "spikes") -> Optional[torch.Tensor]:
    """The full (T, B, N) trace the readout circuit sees, or ``None``.

    ``readout`` collapses this to per-class rates. Callers that need to compare
    *where* two implementations fired — not just which class won — want the
    trace itself, because a rate that matches can hide two spike trains that
    disagree in opposite directions.
    """
    src = output_source(nir_graph)
    key = "membrane" if mode == "membrane" else "spikes"
    stream = sim.get(key, {}).get(src) if src else None
    return stream if (stream is not None and stream.numel()) else None


def readout(nir_graph: Any, sim: Dict[str, Any], mode: str = "spikes") -> torch.Tensor:
    """Decode one scalar-per-class prediction the way a target board would.

    ``sim["output"]`` is the value of the Output node on the *last* step, which
    is not what a chip reports: a neuromorphic core hands back a spike stream and
    the readout circuit counts it over the whole window. Comparing a last-step
    sample against a T-step count produces a large fake disagreement — measured
    at 0.875 on an MNIST net that is in fact identical — so the aggregation lives
    here, once, and both the verifier and the report use it.
    """
    src = None
    stream = output_stream(nir_graph, sim, mode)
    if stream is not None:
        return stream.mean(0)
    return sim["output"]


def _apply_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """NIR Linear is ``y = W x`` with ``W`` shaped ``(out, in)``.

    NIR declares the input node's type as flattened, but callers often hand in
    the spatial tensor the trainer used (``(batch, 1, 28, 28)``). Flatten when the
    element count matches; refuse when it does not, because that is a real shape
    bug rather than a formatting difference.
    """
    if x.shape[-1] != w.shape[-1]:
        batch = x.shape[0] if x.dim() > 1 else 1
        if batch and x.numel() % batch == 0 and x.numel() // batch == w.shape[-1]:
            x = x.reshape(batch, -1)
        else:
            raise NIRSimError(
                f"Linear weight expects {w.shape[-1]} inputs but got "
                f"{x.shape[-1]} (shape {tuple(x.shape)})")
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1])
    return (flat @ w.t()).reshape(*lead, w.shape[0])


def _apply_conv(node: Any, x: torch.Tensor) -> torch.Tensor:
    w = _as_tensor(node.weight)
    b = _as_tensor(_attr(node, "bias", np.zeros(w.shape[0]))).reshape(-1)
    stride = [int(v) for v in np.asarray(_attr(node, "stride", [1, 1])).reshape(-1)]
    pad = [int(v) for v in np.asarray(_attr(node, "padding", [0, 0])).reshape(-1)]
    dil = [int(v) for v in np.asarray(_attr(node, "dilation", [1, 1])).reshape(-1)]
    groups = int(_attr(node, "groups", 1))
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return F.conv2d(x, w, b, stride=stride, padding=pad, dilation=dil, groups=groups)


def _apply_pool(node: Any, x: torch.Tensor) -> torch.Tensor:
    kind = type(node).__name__
    k = [int(v) for v in np.asarray(_attr(node, "kernel_size", [1, 1])).reshape(-1)]
    s = [int(v) for v in np.asarray(_attr(node, "stride", k)).reshape(-1)]
    p = [int(v) for v in np.asarray(_attr(node, "padding", [0, 0])).reshape(-1)]
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if kind == "AvgPool2d":
        return F.avg_pool2d(x, k, s, p)
    if kind == "SumPool2d":
        return F.avg_pool2d(x, k, s, p) * float(k[0] * k[1])
    return F.max_pool2d(x, k, s, p)


def _step_neuron(node: Any, kind: str, state: _NeuronState,
                 current: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """One Euler step of a LIF/LI/IF neuron, per the NIR specification."""
    n = state.v.numel()
    cur = _as_tensor(current)
    if cur.shape != state.v.shape:
        # first step: broadcast the flat membrane to the incoming batch shape
        if cur.numel() % n == 0:
            state.v = state.v.expand_as(cur).clone()
        else:
            cur = cur.reshape(state.v.shape)

    if kind in ("LI", "CubaLI"):
        tau = _per_neuron(_attr(node, "tau", torch.ones(n)), n)
        r = _per_neuron(_attr(node, "r", torch.ones(n)), n)
        v_leak = _per_neuron(_attr(node, "v_leak", torch.zeros(n)), n)
        v = state.v + (dt / tau) * ((v_leak - state.v) + r * cur)
        state.v = v
        return v, torch.zeros_like(v)

    tau = _per_neuron(_attr(node, "tau", torch.ones(n)), n)
    r = _per_neuron(_attr(node, "r", torch.ones(n)), n)
    v_th = _per_neuron(_attr(node, "v_threshold", torch.ones(n)), n)

    if kind in ("IF",):
        v = state.v + (dt / tau) * (r * cur)
        v_leak = None
    else:
        v_leak = _per_neuron(_attr(node, "v_leak", torch.zeros(n)), n)
        v = state.v + (dt / tau) * ((v_leak - state.v) + r * cur)

    v_pre = v                       # potential *before* reset: what a membrane
                                    # readout sees, and what training accumulated
    spk = (v > v_th).to(v.dtype)

    # NIR reset: v <- v_reset where v > v_threshold.
    # `rho` is the optional per-neuron multiplicative reset used by some targets.
    # `v_reset` is optional in the standard, and *absent* does not mean "no
    # reset": nir's own ``LIF.__post_init__`` substitutes zeros, so a missing
    # v_reset resets to 0. Reading it as "leave v alone" silently turns a
    # zero-reset network into an integrator that never forgets a spike.
    v_reset = _attr(node, "v_reset", None)
    rho = _attr(node, "rho", None)
    if v_reset is None:
        v = torch.where(spk > 0, torch.zeros_like(v), v)
    else:
        vr = _per_neuron(v_reset, n)
        v = torch.where(spk > 0, vr, v)
    if rho is not None:
        rr = _per_neuron(rho, n)
        v = torch.where(spk > 0, rr * v, v)

    state.v = v
    return v_pre, spk
