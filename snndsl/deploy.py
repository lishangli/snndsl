"""Deployment analysis: can this network actually run on neuromorphic hardware?

The question every SNN team asks before committing to a chip is some version of
*"will my trained model survive the move?"* Today the only way to answer it is to
buy the chip, port by hand, and find out. This module answers it at compile time,
from the IR, before a single line of target code exists.

Two layers of answer:

``analyze``
    Static. Walks the IR and reports every construct that will not survive the
    trip, with a severity and a concrete fix. No weights, no data, no hardware.

``verify``
    Dynamic. Exports the graph to NIR, runs it through an independent reference
    interpreter (:mod:`snndsl.backends.nir_sim`), and compares the result against
    the PyTorch module the user actually trained. This is the difference between
    *"the file is valid"* and *"the behaviour is preserved"*.

Severity vocabulary
-------------------
``blocker``   the network cannot be deployed without a semantic change
``warning``   it deploys, but the target's behaviour differs from training
``info``      worth knowing, no action needed
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .ir import IRGraph, IRNode

# NIR primitives that have a faithful counterpart on the four mainstream
# neuromorphic targets (Loihi 2, SpiNNaker2, Speck/Xylo, BrainScaleS-2).
_NIR_NATIVE = {"input", "output", "dense", "conv", "pool", "flatten", "neuron"}


@dataclass
class Finding:
    severity: str
    code: str
    where: str
    message: str
    fix: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeployReport:
    findings: List[Finding] = field(default_factory=list)
    node_map: Dict[str, str] = field(default_factory=dict)
    readout: str = "spikes"
    temporal_steps: int = 1

    @property
    def blockers(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "blocker"]

    @property
    def warnings(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == "warning"]

    @property
    def deployable(self) -> bool:
        return not self.blockers

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployable": self.deployable,
            "readout": self.readout,
            "temporal_steps": self.temporal_steps,
            "counts": {
                "blocker": len(self.blockers),
                "warning": len(self.warnings),
                "info": sum(1 for f in self.findings if f.severity == "info"),
            },
            "findings": [f.to_dict() for f in self.findings],
            "node_map": self.node_map,
        }


# --------------------------------------------------------------------- static
def _readout_of(graph: IRGraph) -> str:
    """Readout mode, preferring the output node (self-contained) over infer cfg."""
    for oid in graph.outputs:
        node = graph.node(oid)
        ro = node.attrs.get("readout")
        if ro:
            return str(ro).lower()
    return str(graph.infer.get("readout", "spikes")).lower()


def analyze(graph: IRGraph) -> DeployReport:
    """Static deployability analysis. Pure function of the IR."""
    rep = DeployReport(readout=_readout_of(graph),
                       temporal_steps=int(graph.temporal.steps))
    findings = rep.findings

    for n in graph.nodes:
        if n.op not in _NIR_NATIVE:
            if n.op in ("encode", "decode", "dropout"):
                findings.append(Finding(
                    "info", "io-adapter", n.id,
                    f"{n.op} has no NIR equivalent and is dropped; the target "
                    f"receives raw currents instead",
                    "feed analog currents (rate) or an event sensor stream "
                    "(event) on the target side"))
            else:
                findings.append(Finding(
                    "blocker", "unsupported-op", n.id,
                    f"operation {n.op!r} has no NIR primitive",
                    "rewrite the branch so the merge happens in the target "
                    "framework, or split the graph"))
            continue

        if n.op == "neuron":
            _check_neuron(graph, n, findings)

        if n.op == "pool" and str(n.attrs.get("kind", "avg")) == "max":
            findings.append(Finding(
                "blocker", "max-pool", n.id,
                "max pooling has no NIR primitive - the standard defines "
                "AvgPool2d and SumPool2d only - so it would ship as average or "
                "sum pooling, which changes the drive and with it the whole "
                "spike train (measured on examples/cifar_hw.snn: exporting it "
                "as sum pooled the 2x2 windows and drove the first layer's "
                "spike rate from 0.02 to 0.37)",
                "retrain with kind=avg pooling, which maps exactly, or keep this "
                "conv stack off the deployment path"))

        if n.op == "conv":
            g = int(n.attrs.get("groups", 1))
            if g > 1:
                findings.append(Finding(
                    "warning", "grouped-conv", n.id,
                    f"grouped convolution (groups={g}) is not supported by all "
                    f"targets",
                    "verify the target's conv support or split into dense layers"))

    # ---- readout semantics: the single most common deployment surprise
    if rep.readout == "membrane":
        findings.append(Finding(
            "blocker", "membrane-readout", ",".join(graph.outputs) or "output",
            "the classification head reads the membrane potential *before* reset, "
            "but a NIR spiking neuron emits spikes, not potentials",
            "either train a spike-readout head (readout=spikes), or add a "
            "non-spiking readout layer on the target and read its state there"))
    else:
        findings.append(Finding(
            "info", "spike-readout", ",".join(graph.outputs) or "output",
            "spike readout maps 1:1 onto a NIR spiking layer",
            ""))

    # ---- temporal budget
    if rep.temporal_steps > 64:
        findings.append(Finding(
            "warning", "long-horizon", "temporal",
            f"{rep.temporal_steps} time steps exceeds the per-inference budget of "
            f"most edge targets",
            "reduce steps, or rely on the looped schedule the compiler already "
            "selects above T=4"))
    return rep


def _check_neuron(graph: IRGraph, node: IRNode, findings: List[Finding]) -> None:
    from .backends.nir_export import recognise_lif

    spec = graph.neurons[node.attrs["model"]]

    # ---- A (C,H,W) activation is fine. NIR derives a neuron's input type from
    # the shape of its parameter array (``LIF.__post_init__`` sets
    # ``input_type = {"input": np.array(r.shape)}``), so per-element parameters
    # of shape (C,H,W) are legal and ``conv -> neuron -> conv`` round-trips
    # bit-exactly. An earlier version of this check assumed neurons were 1-D and
    # blocked every conv SNN; that was wrong, and it was wrong in the direction
    # of blaming the standard for our own flattening. The only real cost is that
    # the file carries C*H*W parameter entries per neuron layer - identical to a
    # dense layer of the same size, and there is no broadcast form in NIR that
    # would shrink it.
    lif = recognise_lif(graph, node)
    if lif is None:
        # ---- a spiking neuron that is not LIF exports as a plain leaky
        # integrator: no threshold, no reset, no spikes. That is not an
        # approximation of the model, it is a different model. Measured on the
        # ALIF example (examples/adaptive_lif.snn): top-1 agreement 9.4% against
        # the trained module, i.e. chance. Calling that a "warning" would let
        # someone ship it, so it is a blocker and the fix is spelled out.
        spiking = "s" in getattr(spec, "states", {}) or "spike" in str(
            getattr(spec, "body", ""))
        findings.append(Finding(
            "blocker" if spiking else "info",
            "neuron-not-lif" if spiking else "neuron-integrator", node.id,
            (f"neuron model {spec.name!r} emits spikes but is not an exact LIF, so "
             f"it cannot map onto a NIR spiking neuron; exported as a leaky "
             f"integrator it loses threshold and reset entirely and the target "
             f"computes a different function" if spiking else
             f"neuron model {spec.name!r} is a pure leaky integrator; NIR's LI "
             f"primitive matches it exactly"),
            ("restructure the body into the LIF form "
             "tau*dv/dt = (v_leak - v) + r*i with `spike s = v > v_th` and a "
             "reset line, or keep this branch on the host and export only the "
             "LIF part" if spiking else "")))
        return

    # A declared reset that the equations do not implement is worse than either
    # alternative: the user believes they fixed it, the code did not change.
    declared = lif.get("declared_reset", "")
    actual = lif.get("reset_mode", "")
    if declared and actual not in ("", "unknown") and declared != actual:
        findings.append(Finding(
            "blocker", "reset-mismatch", node.id,
            f"the temporal header declares reset={declared!r} but the neuron "
            f"equations implement reset={actual!r}; the header is documentation, "
            f"the body is what runs",
            "fix the body, not the header — write `v = v * (1 - s)` for a zero "
            "reset, or change the header to match the body"))

    if actual == "subtract":
        findings.append(Finding(
            "warning", "reset-subtract", node.id,
            f"reset is 'subtract' (v <- v - v_th); NIR only models a fixed "
            f"v_reset, so a strongly driven neuron will diverge after firing "
            f"(exported v_reset={lif['v_reset']:.3g})",
            "use a zero reset in the body (v = v * (1 - s)); the NIR export is "
            "then bit-exact — verify with `snndsl deploy --verify`"))
    elif actual == "zero":
        findings.append(Finding(
            "info", "reset-zero", node.id,
            "zero reset maps exactly onto NIR's v_reset",
            ""))
    elif actual == "unknown":
        findings.append(Finding(
            "warning", "reset-unknown", node.id,
            "the reset rule could not be classified; equivalence was not verified",
            "keep the reset line as `v = v * (1 - s)` or `v = v - s * v_th`"))

    if abs(lif["r"] - 1.0) > 1e-6:
        findings.append(Finding(
            "info", "current-gain", node.id,
            f"input current is scaled by r={lif['r']:.4g} in NIR terms "
            f"(tau/dt); the exporter carries this factor so behaviour is preserved",
            ""))
    findings.append(Finding(
        "info", "neuron-native", node.id,
        f"{spec.name} maps to NIR LIF (tau={lif['tau']:.4g}, "
        f"v_th={lif['v_threshold']:.4g})",
        ""))


# -------------------------------------------------------------------- dynamic
def _upstream_of_output(nir_graph: Any) -> Optional[str]:
    """Name of the node feeding the graph's Output — the readout layer."""
    outs = [n for n, v in nir_graph.nodes.items()
            if type(v).__name__ == "Output"]
    if not outs:
        return None
    target = outs[0]
    for src, dst in nir_graph.edges:
        if dst == target:
            return src
    return None


def verify(graph: IRGraph,
           module: Any,
           sample: Any,
           weights: Optional[Dict[str, Any]] = None,
           steps: Optional[int] = None,
           nir_path: Optional[str] = None) -> Dict[str, Any]:
    """Compare the trained PyTorch module against the exported NIR graph.

    ``module`` must be the compiled module (already holding the trained weights).
    ``sample`` is one input batch in the shape the IR declares.
    ``nir_path`` verifies the file that will actually ship, not the in-memory
    graph. That distinction matters: the two can differ, and only one of them
    ends up on the chip.

    Returns a dict with the numerics, ready to be printed or embedded in a report.
    """
    import torch

    from .backends import nir_sim
    from .backends.nir_export import build_nir_graph

    steps = int(steps or graph.temporal.steps)
    if nir_path:
        import nir as nir_pkg
        nir_graph = nir_pkg.read(nir_path)
        export_report = {"source": "file", "path": nir_path}
    else:
        nir_graph, export_report = build_nir_graph(graph, weights)
        export_report = dict(export_report, source="memory")

    # ---- a weightless graph fires nothing, so a "match" against it is a match
    # against silence. Say so instead of reporting PRESERVED.
    blank = [nid for nid, node in nir_graph.nodes.items()
             if type(node).__name__ in ("Affine", "Linear", "Conv2d")
             and not np.any(np.asarray(getattr(node, "weight", 0)))]

    # ---- reference run: deterministic rate coding is a constant current, which
    # is exactly what an analog front-end injects, so replay the same current.
    x = torch.as_tensor(sample) if not isinstance(sample, torch.Tensor) else sample
    x = x.detach().float()
    if x.dim() == len(graph.node(graph.inputs[0]).shape):
        x = x.unsqueeze(0)

    # ---- The encoder has no NIR node, so it has to be folded into the drive.
    # A `rate` encoder injects ``x * gain`` as a constant current on every step,
    # which is precisely what an analog front-end does; dropping the gain and
    # then comparing against ``module(x)`` (which *does* apply it) compares a
    # 3x-scaled network against an unscaled one and reports a fake mismatch.
    # Non-rate schemes sample randomness, so no current reproduces them - those
    # stay reported as a semantic gap rather than being papered over.
    #
    # ``x_drive`` is deliberately a separate tensor: ``module`` below still gets
    # the raw ``x`` and applies the gain itself.
    x_drive = x
    for n in graph.nodes:
        if n.op != "encode":
            continue
        scheme = str(n.attrs.get("scheme", ""))
        if scheme == "rate":
            x_drive = x_drive * float(n.attrs.get("gain", 1.0))

    sim = nir_sim.simulate(nir_graph, x_drive, steps=steps, dt=float(graph.temporal.dt))

    with torch.no_grad():
        ref = module(x)

    ref_t = torch.as_tensor(ref).detach().float()

    # ---- read the output the way the target will read it. A neuromorphic chip
    # hands back a *stream* over T steps; counting spikes (or averaging the
    # membrane) happens in the readout circuit, outside the network. The
    # aggregation lives in nir_sim.readout so the verifier, the CLI and the HTML
    # report cannot drift into three different definitions of "the output".
    readout = _readout_of(graph)
    got_t = nir_sim.readout(nir_graph, sim, mode=readout).detach().float()

    if got_t.shape != ref_t.shape:
        got_t = got_t.reshape(ref_t.shape)

    diff = (got_t - ref_t).abs()
    scale = ref_t.abs().mean().clamp(min=1e-9)
    top1_ref = ref_t.reshape(ref_t.shape[0], -1).argmax(1)
    top1_got = got_t.reshape(got_t.shape[0], -1).argmax(1)

    # ---- per-layer spike-rate comparison explains *why* a mismatch happens:
    # a wrong current gain shows up as a spike-rate ratio far from 1.
    layer_stats = []
    for name, spk in sim["spikes"].items():
        if spk.numel() == 0:
            continue
        layer_stats.append({
            "node": name,
            "nir_spike_rate": float(spk.mean().item()),
            "steps": int(spk.shape[0]),
        })

    rel = float(diff.mean().item() / scale.item())
    # ---- if the reference itself is constant, the comparison carries no
    # information: random noise fed to a spiking net often fires nothing, and
    # "0.0 against 0.0" must never be reported as proof of anything.
    vacuous = bool(ref_t.numel() > 1 and float(ref_t.std().item()) == 0.0)
    return {
        "steps": steps,
        "max_abs_error": float(diff.max().item()),
        "mean_abs_error": float(diff.mean().item()),
        "relative_error": rel,
        "top1_agreement": float((top1_ref == top1_got).float().mean().item()),
        "reference_output_mean": float(ref_t.mean().item()),
        "nir_output_mean": float(got_t.mean().item()),
        "layers": layer_stats,
        "export": export_report,
        "nir_source": export_report.get("source", "memory"),
        "blank_weight_nodes": blank,
        "vacuous": vacuous,
        # a silent network cannot be "equivalent" to anything, however small
        # the arithmetic difference looks
        "equivalent": bool(rel < 1e-3 and not blank and not vacuous),
    }


def format_report(rep: DeployReport) -> str:
    """Human-readable summary, tuned for a terminal."""
    lines = []
    verdict = "DEPLOYABLE" if rep.deployable else "NOT DEPLOYABLE"
    lines.append(f"[{verdict}] readout={rep.readout}  steps={rep.temporal_steps}")
    for f in rep.findings:
        tag = {"blocker": "BLOCK", "warning": "WARN ", "info": "info "}[f.severity]
        lines.append(f"  {tag} {f.where:<14s} {f.message}")
        if f.fix:
            lines.append(f"        {'':14s} -> {f.fix}")
    return "\n".join(lines)
