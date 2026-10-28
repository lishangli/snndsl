"""NIR -> snnTorch, written down once because it is easy to get subtly wrong.

Two things in here are not obvious and both were found by a disagreement rather
than by reading the docs:

``reset_delay=False``
    snnTorch defaults it to ``True``, which applies the reset decided by the
    *previous* step. NIR resets on the step the neuron fires. Leaving the
    default in place shifts every spike train by one step and turns a correct
    export into a 98%-agreement mystery.

``AFFINE_OVERRIDE``
    nirtorch's own ``DEFAULT_MAP`` for ``nir.Affine`` ends with
    ``module.weight.bias = ...`` — an attribute on the weight tensor, not on the
    module. It silently discards every bias. Graphs still load, still run, and
    are quietly wrong.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

NEURON_TYPES = ("LIF", "LI", "IF")


def _const(param: Any, default: float) -> float:
    arr = np.asarray(param, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else default


class AdaptedLIF(nn.Module):
    """A snnTorch neuron plus the two numbers NIR needs but snnTorch lacks.

    NIR LIF:  tau dv/dt = (v_leak - v) + r*i
    snnTorch: mem <- beta*mem + input

    Discretising NIR with step ``dt`` gives

        v[k+1] = (1 - dt/tau) v[k] + (dt/tau)(v_leak + r i[k])

    so ``beta = 1 - dt/tau`` while the drive is ``(dt/tau)*(v_leak + r*i)``.
    Only that gain and offset live here; the membrane update, threshold compare
    and reset are executed by snnTorch.
    """

    def __init__(self, neuron: nn.Module, gain: float, offset: float):
        super().__init__()
        self.neuron = neuron
        self.gain = float(gain)
        self.offset = float(offset)
        self.neuron.init_leaky()          # single-argument (stateful) mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron(self.gain * x + self.offset)


def build_neuron(node: Any, dt: float, log: List[dict], node_name: str) -> nn.Module:
    """Turn one NIR neuron into a snnTorch module, recording how it was done."""
    import snntorch as snn

    kind = type(node).__name__
    tau = _const(getattr(node, "tau", None), 1.0)
    r = _const(getattr(node, "r", None), 1.0)
    v_leak = _const(getattr(node, "v_leak", None), 0.0)
    v_th = _const(getattr(node, "v_threshold", None), 1.0)
    v_reset = _const(getattr(node, "v_reset", None), 0.0)
    if tau <= 0:
        raise ValueError(f"{kind} with tau={tau} cannot be discretised")

    beta = 1.0 - dt / tau
    gain = (dt / tau) * r
    offset = (dt / tau) * v_leak

    if abs(v_reset) > 1e-9:
        raise ValueError(
            f"{node_name}: v_reset={v_reset} is not zero; snnTorch's Leaky can "
            f"only reset to zero, so this graph cannot be cross-validated with "
            f"this backend")

    neuron = snn.Leaky(beta=beta, threshold=v_th, reset_mechanism="zero",
                       init_hidden=True, reset_delay=False)
    log.append({"node": node_name, "type": kind, "snntorch": "snn.Leaky",
                "beta": round(beta, 6), "threshold": v_th, "reset": "zero",
                "input_gain": round(gain, 6), "input_offset": round(offset, 6),
                "who_integrates": "snnTorch"})
    if abs(gain - 1.0) < 1e-9 and abs(offset) < 1e-9:
        return neuron                  # unmodified third-party module
    return AdaptedLIF(neuron, gain, offset)


def affine_module(node: Any) -> nn.Module:
    """Rebuild an Affine ourselves - see ``AFFINE_OVERRIDE`` above."""
    w = torch.as_tensor(np.asarray(node.weight), dtype=torch.float32)
    b = torch.as_tensor(np.asarray(node.bias), dtype=torch.float32)
    module = nn.Linear(w.shape[1], w.shape[0], bias=True)
    with torch.no_grad():
        module.weight.copy_(w)
        module.bias.copy_(b)
    return module


def linear_module(node: Any) -> nn.Module:
    w = torch.as_tensor(np.asarray(node.weight), dtype=torch.float32)
    module = nn.Linear(w.shape[1], w.shape[0], bias=False)
    with torch.no_grad():
        module.weight.copy_(w)
    return module


def make_map(dt: float, log: Optional[List[dict]] = None):
    """Build a nirtorch model_map that delegates everything it can.

    Non-spiking layers go through nirtorch's own mapping; only ``Affine`` is
    taken over, and only because the shipped one drops the bias.
    """
    from nirtorch.nir_interpreter import DEFAULT_MAP

    log = log if log is not None else []

    def mapper(node: Any) -> Optional[nn.Module]:
        kind = type(node).__name__
        if kind in ("Input", "Output"):
            return None
        if kind == "Affine":
            return affine_module(node)
        if kind == "Linear":
            return linear_module(node)
        if kind in NEURON_TYPES:
            return build_neuron(node, dt, log, kind)
        try:
            return DEFAULT_MAP(node)
        except Exception:                # noqa: BLE001
            raise NotImplementedError(kind)

    return mapper
