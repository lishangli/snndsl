"""Pre-flight for the SNN-MLIR backend, and a thin bridge to its CLI.

Why a pre-flight instead of exporting and letting the backend say no: the
backend's *rejections* are cheap, but its **acceptances** are not all correct.
Reading ``snn_mlir/_graph.py`` and ``snn_mlir/nodes/lif.py`` separates two kinds
of finding, and they are not the same question:

``kind="refuse"``
    The backend will exit non-zero. Nothing to argue about.

``kind="lossy"``
    The backend will happily convert, and compile something that is *not this
    network*. Two cases, both measured on this repo's examples:
    a max-pool exported as average-pool (NIR has no ``MaxPool2d``), and an
    adaptive neuron exported as ``LI``. ``snn-mlir check`` reports ``supported``
    for both. A pre-flight that only mirrored the backend's own verdict would
    wave them through.

So the useful output is not one boolean but two:

``accepts``
    The backend's opinion — will it run.
``faithful``
    Whether running it means running *this* network.

Given ``accepts and not faithful``, the file converts, the C compiles, the
numbers move, and every check downstream still passes. That is the failure mode
this module exists to name.

The sharpest rule under ``lossy`` is ``r``. SNN-MLIR recovers the timestep from
the LIF node as ``dt = tau / r`` and then ``decay = 1 - dt / tau``. That is only
right when ``r == tau / dt`` — i.e. when the neuron integrates its input with
gain 1. A DSL neuron written as ``v = v*(1 - dt/tau) + 2*i`` exports
``r = 2*tau/dt``; the backend then computes ``decay = 1 - dt/(2*tau)`` while
still feeding the input at gain 1. Nothing raises. Our exporter is right here —
NIR's own equations do describe that network — the reinterpretation happens
inside the consumer.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

from ..ir import IRGraph

#: CLI name; ``$SNN_MLIR`` overrides it with an explicit path.
_CLI = "snn-mlir"


def cli_path() -> Optional[str]:
    """Where the backend CLI is, or ``None``. Also honours ``$SNN_MLIR``."""
    env = os.environ.get("SNN_MLIR")
    if env and os.path.exists(env):
        return env
    return shutil.which(_CLI)


def preflight(graph: IRGraph) -> Dict[str, Any]:
    """Classify every reason this graph cannot go down the SNN-MLIR path."""
    from .nir_export import recognise_lif

    findings: List[Dict[str, str]] = []
    notes: List[Dict[str, str]] = []

    # ── topology ─────────────────────────────────────────────────────────────
    # "only linear-chain graphs are supported" — snn_mlir/_graph.py:147-162
    for n in graph.nodes:
        if n.op in ("input", "output"):
            continue
        succ = graph.consumers(n.id)
        if len(succ) > 1:
            findings.append({
                "node": n.id, "kind": "refuse", "rule": "nonlinear_topology",
                "detail": f"{len(succ)} successors ({', '.join(c.id for c in succ)})",
                "source": "snn_mlir/_graph.py:147"})
        elif not succ:
            findings.append({
                "node": n.id, "kind": "refuse", "rule": "dead_end",
                "detail": "never reaches output",
                "source": "snn_mlir/_graph.py:154"})

    for n in graph.nodes:
        if n.op in ("add", "concat"):
            # We write an identity Linear so the NIR still type-checks. Linear is
            # a *synapse* to the backend, so a two-input Linear is neither a
            # merge nor a legal fan-in target (_graph.py:167-179).
            findings.append({
                "node": n.id, "kind": "refuse", "rule": "no_merge_primitive",
                "detail": f"'{n.op}' has no NIR primitive; the identity placeholder "
                          "is a synapse, not a merge",
                "source": "snn_mlir/_graph.py:167"})
        elif n.op == "pool" and str(n.attrs.get("kind", "avg")) == "max":
            findings.append({
                "node": n.id, "kind": "lossy", "rule": "maxpool_as_avgpool",
                "detail": "exported as AvgPool2d; the backend converts it without "
                          "complaint and the numbers are not the same network",
                "source": "nir_export.py: NIR has no MaxPool2d"})
        elif n.op == "conv":
            if int(n.attrs.get("groups", 1) or 1) != 1:
                findings.append({
                    "node": n.id, "kind": "refuse", "rule": "conv_groups",
                    "detail": f"groups = {n.attrs.get('groups')}",
                    "source": "snn_mlir/nodes/conv.py:195"})
            if int(n.attrs.get("dilation", 1) or 1) != 1:
                findings.append({
                    "node": n.id, "kind": "refuse", "rule": "conv_dilation",
                    "detail": f"dilation = {n.attrs.get('dilation')}",
                    "source": "snn_mlir/nodes/conv.py:190"})
        elif n.op == "neuron":
            findings.extend(_neuron_rules(graph, n, recognise_lif))

    # ── what the format cannot carry, whatever the backend thinks ────────────
    if any(n.op == "encode" for n in graph.nodes):
        notes.append({
            "node": "encode",
            "detail": "dropped: NIR delivers the input as a current, so rate/analog "
                      "coding has to be reproduced by whoever feeds the model"})
    steps = int(getattr(graph.temporal, "steps", 1) or 1)
    if steps > 1:
        notes.append({
            "node": "-",
            "detail": f"T={steps} lives in metadata only; the backend gets it from "
                      "the input's leading axis ([n_steps, n_features])"})

    blockers = [f for f in findings if f["kind"] == "refuse"]
    lossy = [f for f in findings if f["kind"] == "lossy"]
    return {"accepts": not blockers, "faithful": not blockers and not lossy,
            "blockers": blockers, "lossy": lossy, "notes": notes,
            "cli": cli_path()}


def _neuron_rules(graph: IRGraph, n: Any, recognise) -> List[Dict[str, str]]:
    """One neuron against ``nodes/lif.py:76-87`` plus the ``r`` reinterpretation."""
    out: List[Dict[str, str]] = []
    lif = recognise(graph, n)
    if lif is None:
        out.append({
            "node": n.id, "kind": "lossy", "rule": "neuron_downgraded_to_li",
            "detail": f"model {n.attrs.get('model')!r} has no NIR equivalent and is "
                      "exported as LI; the backend converts that without complaint",
            "source": "nir_export.py:328"})
        return out

    dt = float(graph.temporal.dt)
    ideal = lif["tau"] / dt
    if abs(lif["r"] - ideal) > 1e-3 * max(1.0, abs(ideal)):
        out.append({
            "node": n.id, "kind": "lossy", "rule": "input_gain_baked_into_r",
            "detail": f"r = {lif['r']:.6g} but tau/dt = {ideal:.6g}: the backend reads "
                      f"decay = 1 - 1/r = {1 - 1 / lif['r']:.6g} instead of "
                      f"{1 - dt / lif['tau']:.6g}, and feeds the input at gain 1",
            "source": "snn_mlir/nodes/lif.py:89"})
    if abs(lif["v_leak"]) > 1e-12:
        out.append({
            "node": n.id, "kind": "refuse", "rule": "v_leak_nonzero",
            "detail": f"v_leak = {lif['v_leak']:.6g}",
            "source": "snn_mlir/nodes/lif.py:76"})
    if abs(lif["v_reset"]) > 1e-12:
        out.append({
            "node": n.id, "kind": "refuse", "rule": "v_reset_nonzero",
            "detail": f"v_reset = {lif['v_reset']:.6g}; only zero reset is supported",
            "source": "snn_mlir/nodes/lif.py:86"})
    return out


# ── CLI bridge ────────────────────────────────────────────────────────────────

def _run(args: List[str], cwd: Optional[str] = None) -> Dict[str, Any]:
    cli = cli_path()
    if cli is None:
        raise RuntimeError(
            "the snn-mlir CLI is not installed (pip install snn-mlir); "
            "writing a .nir with target='snn-mlir' works without it")
    proc = subprocess.run([cli, *args], cwd=cwd, capture_output=True, text=True)
    return {"ok": proc.returncode == 0, "returncode": proc.returncode,
            "command": " ".join([_CLI, *args]),
            "stdout": proc.stdout, "stderr": proc.stderr}


def check(path: str) -> Dict[str, Any]:
    """``snn-mlir check``, with the JSON report parsed when it is available."""
    res = _run(["check", "--json", path])
    res["report"] = None
    for text in (res["stdout"], res["stderr"]):
        text = text.strip()
        if text.startswith("{"):
            try:
                res["report"] = json.loads(text)
                break
            except json.JSONDecodeError:
                continue
    return res


def to_mlir(path: str, out: Optional[str] = None, quantize: bool = False) -> Dict[str, Any]:
    """Lower a ``.nir`` to SNN-dialect MLIR text (one ``func.func``, no time loop)."""
    args = ["export", path]
    if out:
        args += ["-o", out]
    if quantize:
        args.append("-q")
    res = _run(args)
    res["mlir"] = out or (os.path.splitext(path)[0] + ".mlir")
    return res


def codegen(folder: str, quantize: bool = False) -> Dict[str, Any]:
    """Emit ``build/`` (``network.mlir`` + ``main.c`` + ``snn_data.h`` + ``input.h``).

    Needs one ``.nir`` and one ``input.csv``/``input.npy`` of shape
    ``[n_steps, n_features]`` in ``folder``. The input is baked in as ``int8``,
    so spike trains are the natural currency — analog values are refused with
    ``input.npy has fractional values``.
    """
    args = ["codegen", folder] + (["-q"] if quantize else [])
    res = _run(args)
    res["build"] = os.path.join(folder, "build")
    return res


def run(folder: str, quantize: bool = False) -> Dict[str, Any]:
    """Compile and execute on the CPU reference. Needs the ``snn-opt`` toolchain.

    That toolchain is *not* on PyPI: ``pip install snn-opt`` installs an
    unrelated SOCP solver that happens to share the name, and the real one is
    built from MLIR/LLVM sources (``scripts/build_snn_dialect.sh``). So this
    returns ``ok=False`` carrying the backend's own explanation rather than
    pretending the loop closed.
    """
    args = ["run", folder] + (["-q"] if quantize else [])
    res = _run(args)
    res["results"] = os.path.join(folder, "results.csv")
    return res
