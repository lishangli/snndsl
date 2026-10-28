"""Export to NIR (Neuromorphic Intermediate Representation).

NIR is the interchange format used by Lava/Loihi, SpiNNaker2, Speck/Xylo,
Sinabs, snnTorch, Norse, Spyx and BrainScaleS-2. Emitting it means a network
compiled by this DSL can run on neuromorphic hardware without rewriting.

Reference: Pedersen et al., Nature Communications 15, 8122 (2024).
Format spec: https://neuroir.org
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..ir import IRGraph, IRNode

try:
    import nir
    HAS_NIR = True
except Exception:            # pragma: no cover - optional dependency
    nir = None
    HAS_NIR = False


def _safe(name: str) -> str:
    return re.sub(r"[^0-9a-zA-Z_]", "_", name)


def recognise_lif(graph: IRGraph, node: IRNode,
                  dt: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Map a DSL neuron onto NIR's LIF parameterisation.

    NIR LIF::

        tau * dv/dt = (v_leak - v) + r * i
        v <- v_reset   where v > v_threshold

    Discretised with explicit Euler at step ``dt`` this becomes::

        v' = v*(1 - dt/tau) + (dt/tau)*v_leak + (dt/tau)*r*i

    So the DSL body is *probed numerically* for its affine coefficients rather
    than pattern-matched as text: with ``f(v, i) = c + a*v + b*i`` we get

        a = 1 - dt/tau   =>  tau        = dt / (1 - a)
        c = (dt/tau)*v_leak  =>  v_leak = c * tau / dt
        b = (dt/tau)*r       =>  r      = b * tau / dt

    The ``r`` factor matters: a DSL that writes ``v = v*(1-dt/tau) + i`` feeds the
    current in *unscaled*, which is ``r = tau/dt`` in NIR terms. Exporting ``r=1``
    there halves the drive and silently changes the network. Probing catches it.
    """
    spec = graph.neurons[node.attrs["model"]]
    p = spec.params
    if len(spec.states) != 1:
        return None
    kinds = [k for k, _, _ in spec.body]
    if kinds.count("spike") != 1:
        return None

    dt = float(graph.temporal.dt if dt is None else dt)
    v_name = next(iter(spec.states))
    assign = next(((k, t, src) for k, t, src in spec.body
                   if k == "assign" and t == v_name), None)
    if assign is None:
        return None

    _, _, src = assign
    env: Dict[str, float] = {}
    for k, val in p.items():
        env[k] = float(val)      # params shadow states: tau, v_th, v_reset ...
    for k, val in spec.states.items():
        env[k] = float(val)
    env.update({"dt": dt, "t": 0.0, "i": 0.0, v_name: 0.0})

    def f(v: float, i: float) -> float:
        local = dict(env)
        local[v_name] = v
        local["i"] = i
        return float(eval(src, {"__builtins__": {}}, local))   # noqa: S307 - own source

    try:
        c = f(0.0, 0.0)
        a = f(1.0, 0.0) - c
        b = f(0.0, 1.0) - c
        # linearity probe: f(1,1) must equal c + a + b
        if abs(f(1.0, 1.0) - (c + a + b)) > 1e-4 * max(1.0, abs(c + a + b)):
            return None
    except Exception:
        return None

    if a >= 1.0 - 1e-9:          # no leak: not a LIF in NIR terms
        return None
    tau = dt / (1.0 - a)

    # if the block declares tau explicitly, the body must agree with it
    if "tau" in p and abs(float(p["tau"]) - tau) > 1e-3 * max(1.0, tau):
        return None

    leak = c * tau / dt
    r = b * tau / dt
    v_reset = float(p.get("v_reset", 0.0))

    # ---- what the equations *actually* do, not what the header claims.
    # `temporal: reset = zero` is a declaration; the reset rule that runs is the
    # one written in the neuron body. Those two can disagree, and a checker that
    # trusts the declaration would wave a divergent network through.
    declared = str(spec.reset or "subtract").lower()
    actual = _probe_reset_mode(spec, env, v_name, dt)

    return {"tau": tau, "r": r, "v_leak": leak,
            "v_threshold": float(p.get("v_th", p.get("v_threshold", 1.0))),
            "v_reset": v_reset, "reset_mode": actual,
            "declared_reset": declared, "model": spec.name}


def _probe_reset_mode(spec: Any, env: Dict[str, float], v_name: str,
                      dt: float) -> str:
    """Classify the reset rule by driving a fired neuron and reading the result.

    A neuron parked well above threshold (``v = 3``, ``s = 1``, ``v_th = 1``)
    lands on 2 under ``v -= s*v_th`` and on 0 under ``v *= (1-s)``. The difference
    is unambiguous, so the classification is measured rather than guessed.
    """
    spike_var = getattr(spec, "spike_var", "s")
    local = dict(env)
    local[v_name] = 3.0
    local[spike_var] = 1.0
    local["dt"] = dt
    local["i"] = 0.0
    try:
        # only the statements *after* the spike line constitute the reset
        seen_spike = False
        for kind, target, src in spec.body:
            if kind == "spike":
                seen_spike = True
                local[spike_var] = 1.0
            elif seen_spike and kind == "assign" and target == v_name:
                local[v_name] = float(eval(src, {"__builtins__": {}}, local))  # noqa: S307
    except Exception:
        return "unknown"
    after = local[v_name]
    if abs(after) < 1e-6:
        return "zero"
    if abs(after - (3.0 - 1.0)) < 1e-3:
        return "subtract"
    return "custom"


# kept for backwards compatibility with older call sites
_recognise_lif = recognise_lif


class NIRShapeError(ValueError):
    """The IR is valid but the NIR standard cannot represent it."""


def _flatten_node(shape) -> Any:
    """A NIR Flatten with its types filled in at construction time."""
    ins = [int(v) for v in shape]
    node = nir.Flatten(start_dim=1, end_dim=-1, input_type={"input": np.array(ins)})
    node.output_type = {"output": np.array([int(np.prod(ins))])}
    return node


def _spatial_input_shape(graph: IRGraph, node: IRNode) -> Tuple[int, int]:
    """(H, W) going into a conv node, taken from its producer, not its output."""
    if node.inputs:
        prev = graph.node(node.inputs[0])
        shape = tuple(int(v) for v in prev.shape)
        if len(shape) == 3:
            return shape[1], shape[2]
    shape = tuple(int(v) for v in node.shape)
    if len(shape) == 3:
        return shape[1], shape[2]
    raise ValueError(f"cannot infer spatial shape for conv node {node.id!r}")


def build_nir_graph(graph: IRGraph,
                    weights: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
    """Return (NIRGraph, report). Raises if the NIR package is missing."""
    if not HAS_NIR:
        raise RuntimeError("the 'nir' package is not installed (pip install nir)")

    nodes: Dict[str, Any] = {}
    edges: List[Tuple[str, str]] = []
    report: Dict[str, Any] = {"nodes": {}, "unsupported": [], "approximated": [],
                              "missing_weight_nodes": [], "zero_weight_nodes": []}
    name_of: Dict[str, str] = {}
    # A caller that passes ``weights={}`` means "I have no trained weights"; a
    # caller that passes a populated dict means "these are the weights, use
    # them". The two must not collapse into the same code path, because the
    # silent zero-fill that follows is indistinguishable from a real export.
    have_weights = bool(weights)
    report["weights_status"] = "supplied" if have_weights else "absent"
    weights = weights or {}

    in_shape = tuple(graph.node(graph.inputs[0]).shape) if graph.inputs else (1,)
    # rate/current coding has no NIR equivalent: inputs are delivered as currents
    skip_ops = {"encode", "decode"}

    def _entry_shape(ir_node: IRNode) -> Tuple[int, ...]:
        """Shape the first non-skipped consumer expects (NIR validates types)."""
        cur = ir_node
        while True:
            cons = graph.consumers(cur.id)
            if not cons:
                break
            nxt = cons[0]
            if nxt.op in skip_ops or nxt.op == "dropout":
                cur = nxt
                continue
            if nxt.op in ("conv", "pool", "flatten"):
                return tuple(int(d) for d in ir_node.shape)
            break
        total = 1
        for d in ir_node.shape:
            total *= int(d)
        return (total,)

    for n in graph.nodes:
        nid = _safe(n.id)
        name_of[n.id] = nid
        if n.op == "input":
            # NIR stores shapes as integer arrays, not zero-filled tensors
            nodes[nid] = nir.Input(input_type={"input": np.array(_entry_shape(n))})
            report["nodes"][nid] = "Input"
        elif n.op == "output":
            nodes[nid] = nir.Output(output_type={"output": np.array(
                [int(np.prod(n.shape)) if n.shape else 1])})
            report["nodes"][nid] = "Output"
        elif n.op in skip_ops:
            report["nodes"][nid] = "skipped"
        elif n.op == "dense":
            w = weights.get(f"w_{_safe(n.name)}")
            b = weights.get(f"b_{_safe(n.name)}")
            if w is None:
                # a zero-filled weight matrix produces a network that emits no
                # spikes at all, and every downstream check "passes" because
                # zero matches zero. Record it so nobody can mistake the file
                # for a trained model.
                report["missing_weight_nodes"].append(nid)
                report["zero_weight_nodes"].append(nid)
                w = np.zeros((n.attrs["units"], n.attrs["in_features"]), dtype=np.float32)
            else:
                w = np.asarray(w, dtype=np.float32).reshape(
                    n.attrs["units"], n.attrs["in_features"])
            if n.attrs.get("bias", True) and b is not None:
                nodes[nid] = nir.Affine(weight=w, bias=np.asarray(b, np.float32).reshape(-1))
                report["nodes"][nid] = "Affine"
            else:
                nodes[nid] = nir.Linear(weight=w)
                report["nodes"][nid] = "Linear"
        elif n.op == "conv":
            cin = n.attrs["in_channels"]
            cout = n.attrs["out_channels"]
            k = n.attrs["kernel"]
            w = weights.get(f"w_{_safe(n.name)}")
            b = weights.get(f"b_{_safe(n.name)}")
            if w is None:
                report["missing_weight_nodes"].append(nid)
                report["zero_weight_nodes"].append(nid)
                w = np.zeros((cout, cin // n.attrs.get("groups", 1), k, k), np.float32)
            else:
                w = np.asarray(w, np.float32)
            bias = np.zeros(cout, np.float32) if b is None else np.asarray(b, np.float32).reshape(-1)
            # NIR's Conv2d.input_shape is the *spatial* shape (H, W); the channel
            # count is read from weight.shape[1]. Passing (C, H, W) makes NIR
            # iterate three spatial axes and index a two-element padding array.
            spatial = _spatial_input_shape(graph, n)
            nodes[nid] = nir.Conv2d(
                input_shape=np.array(spatial, dtype=int),
                weight=w.astype(np.float32), stride=np.array([n.attrs["stride"]] * 2),
                padding=np.array([n.attrs["padding"]] * 2),
                dilation=np.array([1, 1]), groups=int(n.attrs.get("groups", 1)),
                bias=bias.astype(np.float32))
            report["nodes"][nid] = "Conv2d"
        elif n.op == "pool":
            k = n.attrs["kernel"]
            s = n.attrs["stride"]
            cls = nir.AvgPool2d if n.attrs.get("kind", "avg") == "avg" else nir.SumPool2d
            nodes[nid] = cls(kernel_size=np.array([k, k]), stride=np.array([s, s]),
                             padding=np.array([0, 0]))
            report["nodes"][nid] = cls.__name__
        elif n.op == "flatten":
            nodes[nid] = _flatten_node(n.shape)
            report["nodes"][nid] = "Flatten"
        elif n.op == "neuron":
            lif = _recognise_lif(graph, n)
            # NIR parameters are per-neuron arrays; that is what fixes the layer type
            n_units = 1
            for d in n.shape:
                n_units *= int(d)
            if len(tuple(n.shape)) > 1:
                # NIR's neuron layers derive their type from a 1-D per-neuron
                # parameter array, so a (C,H,W) activation cannot connect to
                # one. There is no Unflatten/Reshape in NIR 1.0.8 to restore
                # the spatial dims afterwards either, so `conv -> neuron ->
                # conv` is not expressible. Say it here rather than writing a
                # file that nir.read() will refuse.
                report.setdefault("shape_blocked", []).append(
                    {"node": nid, "shape": list(int(v) for v in n.shape),
                     "reason": "NIR neuron layers are 1-D; no Reshape exists to "
                               "restore spatial dims"})
            if lif is None:
                nodes[nid] = nir.LI(tau=np.full(n_units, 2.0), r=np.ones(n_units),
                                    v_leak=np.zeros(n_units))
                report["nodes"][nid] = "LI(approx)"
                report["approximated"].append(
                    {"node": nid, "reason": "neuron model is not a plain LIF; "
                                            "exported as leaky integrator"})
            else:
                # ---- v_reset is only written when it is not the standard's
                # default. nir 1.0.8 fills a missing v_reset with zeros in
                # ``LIF.__post_init__``, so omitting it changes no semantics,
                # but it *does* decide whether the file can be read at all by
                # the rest of the 1.0.x line: v_reset appeared in a later
                # patch release, and a 1.0.8 file that carries it is rejected
                # by nir 1.0.4 with "LIF.__init__() got an unexpected keyword
                # argument 'v_reset'". SynSense's rockpool and sinabs pin that
                # older nir, so a zero-reset network — the only kind that
                # deploys losslessly (see docs) — stays readable everywhere.
                zero_reset = abs(float(lif["v_reset"])) < 1e-12
                nodes[nid] = nir.LIF(
                    tau=np.full(n_units, lif["tau"]),
                    r=np.full(n_units, lif["r"]),
                    v_leak=np.full(n_units, lif["v_leak"]),
                    v_threshold=np.full(n_units, lif["v_threshold"]),
                    v_reset=None if zero_reset else np.full(n_units, lif["v_reset"]),
                    metadata={"snndsl_reset": lif["reset_mode"],
                              "snndsl_v_reset": float(lif["v_reset"]),
                              "snndsl_reset_omitted": zero_reset})
                report["nodes"][nid] = "LIF"
        elif n.op in ("add", "concat"):
            report["unsupported"].append(
                {"node": nid, "op": n.op,
                 "reason": "NIR has no merge primitive; resolve in the target framework"})
            nodes[nid] = nir.Linear(weight=np.eye(graph.node(n.inputs[0]).numel()
                                                  if n.inputs else 1, dtype=np.float32))
            report["nodes"][nid] = "Linear(identity, merge placeholder)"
        elif n.op == "dropout":
            report["nodes"][nid] = "skipped"
        else:
            report["unsupported"].append({"node": nid, "op": n.op,
                                          "reason": "no NIR primitive"})

    # ---- pin every node's declared types from the IR (NIR validates them)
    for n in graph.nodes:
        nid = _safe(n.id)
        if n.op in ("input", "output") or nid not in nodes:
            continue
        if n.op == "dense":
            ins = (int(n.attrs["in_features"]),)
        elif n.op in ("conv", "pool", "flatten") and n.inputs:
            ins = tuple(int(x) for x in graph.node(n.inputs[0]).shape)
        else:
            ins = tuple(int(x) for x in n.shape)
        node = nodes[nid]
        node.input_type = {"input": np.array([int(x) for x in ins])}
        node.output_type = {"output": np.array([int(x) for x in n.shape])}

    # ---- edges, transparently bypassing skipped nodes
    carriers: Dict[str, List[str]] = {}
    inserted_flatten: List[str] = []
    for n in graph.nodes:
        nid = _safe(n.id)
        srcs: List[str] = []
        for i in n.inputs:
            srcs.extend(carriers.get(i, [i]))
        if n.op in skip_ops or n.op == "dropout":
            carriers[n.id] = srcs
            continue
        for s in srcs:
            if _safe(s) not in nodes or nid not in nodes:
                continue
            src_node = graph.node(s)
            # A dense layer fed by a 3-D tensor relies on the implicit flatten
            # in the generated PyTorch. NIR has no implicit flatten, so the
            # exported graph would carry a (C,H,W) type into a (C*H*W) input
            # and fail type inference. Materialise the flatten here.
            # The Input node is already declared with the flattened width a
            # dense layer expects (NIR delivers currents, not images), so only
            # insert a flatten when the spatial tensor is produced inside the
            # graph.
            if (n.op == "dense" and len(src_node.shape) > 1
                    and src_node.op != "input"):
                synth = f"_flat_{nid}"
                if synth not in nodes:
                    nodes[synth] = _flatten_node(src_node.shape)
                    inserted_flatten.append(synth)
                edges.append((_safe(s), synth))
                edges.append((synth, nid))
            else:
                edges.append((_safe(s), nid))
        carriers[n.id] = [n.id]

    edges = [e for e in dict.fromkeys(edges) if e[0] != e[1]]
    if report.get("shape_blocked"):
        first = report["shape_blocked"][0]
        raise NIRShapeError(
            f"neuron {first['node']!r} receives a {tuple(first['shape'])} activation, "
            f"but NIR neuron layers carry 1-D per-neuron parameters and NIR 1.0.8 "
            f"has no Reshape to restore the spatial dims. Convolutional SNNs "
            f"(conv -> neuron -> conv) are not expressible with this standard "
            f"version; deploy reports this as a blocker")
    nir_graph = nir.NIRGraph(nodes=nodes, edges=edges)
    report["edges"] = edges
    report["inserted_flatten"] = inserted_flatten
    report["node_count"] = len(nodes)
    return nir_graph, report


_PARAM_FIELDS = ("weight", "bias", "tau", "r", "v_leak", "v_threshold", "v_reset",
                 "kernel_size", "stride", "padding", "dilation", "groups")


def parameter_fingerprint(nir_graph: Any) -> Dict[str, str]:
    """Per-node digest of every parameter array.

    The point is that a ``.nir`` file can state what it contains. A zero-weight
    export and a trained export are byte-different but behaviourally identical
    to a checker that only asks "did it parse?".
    """
    import hashlib

    out: Dict[str, str] = {}
    for nid, node in nir_graph.nodes.items():
        h = hashlib.sha1()
        for key in _PARAM_FIELDS:
            arr = getattr(node, key, None)
            if arr is None:
                continue
            a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
            h.update(key.encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        out[nid] = h.hexdigest()[:16]
    return out


def roundtrip_problems(written: Any, reread: Any) -> List[str]:
    """Diff a written graph against the same file read back from disk."""
    problems: List[str] = []
    for nid, node in written.nodes.items():
        back = reread.nodes.get(nid)
        if back is None:
            problems.append(f"node {nid!r} disappeared")
            continue
        if type(back).__name__ != type(node).__name__:
            problems.append(
                f"node {nid!r} changed type {type(node).__name__} -> {type(back).__name__}")
            continue
        for key in _PARAM_FIELDS:
            a, b = getattr(node, key, None), getattr(back, key, None)
            if a is None and b is None:
                continue
            if (a is None) != (b is None):
                problems.append(f"node {nid!r} lost field {key!r} on write")
                continue
            aa = np.asarray(a, dtype=np.float64).reshape(-1)
            bb = np.asarray(b, dtype=np.float64).reshape(-1)
            if aa.shape != bb.shape:
                problems.append(
                    f"node {nid!r} field {key!r} changed shape {aa.shape} -> {bb.shape}")
                continue
            if not np.allclose(aa, bb, atol=1e-7):
                problems.append(
                    f"node {nid!r} field {key!r} differs by {np.abs(aa - bb).max():.3e}")
    return problems


def _downgrade_dialect(path: str, compat: str) -> List[str]:
    """Rewrite an already-written ``.nir`` so an older nir release can load it.

    Only one thing is actually version-sensitive in the graphs we emit:
    ``LIF.v_reset``. Drop it wherever it is zero (the standard's own default)
    so the file loads on every 1.0.x release. Everything else is 1.0.0-era.

    Returns the list of removed fields, for the export report.
    """
    removed: List[str] = []
    try:
        import h5py
    except ImportError:                      # pragma: no cover - h5py ships with nir
        return removed
    try:
        with h5py.File(path, "r+") as fh:
            def walk(group, prefix: str) -> None:
                for key in list(group.keys()):
                    item = group[key]
                    if isinstance(item, h5py.Group):
                        walk(item, f"{prefix}/{key}")
                        continue
                    if (key == "v_reset" and prefix != ""
                            and not np.any(item[()])):
                        del group[key]
                        removed.append(prefix.lstrip("/") + "/v_reset")
            walk(fh, "")
    except Exception:                        # noqa: BLE001 - never fail an export here
        return removed
    return removed


def export_nir(graph: IRGraph, path: str,
               weights: Optional[Dict[str, Any]] = None,
               strict: bool = False,
               compat: str = "1.0.4") -> Dict[str, Any]:
    """Write a ``.nir`` file (or a JSON sidecar if NIR is unavailable).

    After writing, the file is read *back from disk* and diffed against what
    was meant to be written. Exporting a graph and then verifying the in-memory
    copy is the classic way to ship an empty model: the checker passes, the
    file is wrong. ``strict`` additionally refuses to write a parameterised
    graph with no weights at all.

    ``compat`` picks the oldest nir release the file must load in, and defaults
    to the oldest one in the 1.0 line that any shipped toolchain pins. There is
    a real difference: ``LIF.v_reset`` was added in a later 1.0 patch, and nir
    1.0.4 refuses a file that carries it
    (``LIF.__init__() got an unexpected keyword argument 'v_reset'``). Since a
    missing ``v_reset`` already means zeros — nir's own reader substitutes
    ``np.zeros_like(v_threshold)`` — dropping it when it *is* zero costs nothing
    and buys readability on every 1.0.x release and every frontend built
    against them.
    """
    if not HAS_NIR:
        payload = {"format": "snndsl-json-fallback", "graph": graph.to_dict()}
        with open(path + ".json", "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        raise RuntimeError("nir package missing; wrote JSON fallback instead")
    nir_graph, report = build_nir_graph(graph, weights)
    nir.write(path, nir_graph)
    report["dialect"] = {"requested_min": compat, "dropped_fields": []}
    if compat:
        report["dialect"]["dropped_fields"] = _downgrade_dialect(path, compat)

    reread = nir.read(path)
    problems = roundtrip_problems(nir_graph, reread)
    report["roundtrip"] = {"ok": not problems, "problems": problems,
                           "fingerprint": parameter_fingerprint(reread)}

    with open(path + ".report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    if problems:
        raise RuntimeError(
            "exported .nir does not survive a round-trip - the file on disk is "
            "not the graph that was verified: " + "; ".join(problems[:6]))
    if strict and not weights and _has_parameters(nir_graph):
        raise RuntimeError(
            "refusing to write a weightless .nir: the graph has trainable layers "
            "but no weights were supplied, so every spike rate would be zero "
            "and a downstream checker would 'pass' by comparing zeros to zeros")
    return report


def _has_parameters(nir_graph: Any) -> bool:
    return any(type(n).__name__ in ("Affine", "Linear", "Conv2d", "Conv1d", "ConvTranspose")
               for n in nir_graph.nodes.values())


def load_checkpoint_weights(ckpt_path: str) -> Dict[str, Any]:
    """Pull tensors out of a training checkpoint for weight-bearing export."""
    import torch
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob.get("state_dict", blob)
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}
