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

``AVGPOOL_OVERRIDE``
    ``nir.AvgPool2d`` is mapped to ``torch.nn.LPPool2d(norm_type=1, ...)``,
    whose output is ``(sum |x|)^(1/1)`` — a *sum* over the window, not an
    average. On a 2x2 window that multiplies the drive by four, so a conv graph
    cross-checked against nirtorch disagreed on 1 output spike in 8 steps and
    looked like a rounding tie. Measured: ``LPPool2d(1)`` returns 8.0 where
    ``AvgPool2d`` returns 2.0 on a constant 2.0 input.
"""
from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import torch
import torch.nn as nn

NEURON_TYPES = ("LIF", "LI", "IF")


def _const(param: Any, default: float) -> float:
    """Read a NIR neuron parameter as the scalar snnTorch needs.

    Every exporter here writes a constant-per-neuron array (a conv layer stores
    ``(C,H,W)`` copies of one number), so ``arr[0]`` is the right answer - but
    only when they really are copies. A genuine per-neuron tau or threshold
    would be silently averaged away by taking the first element, which is the
    kind of quiet wrongness this project refuses elsewhere, so check instead.
    """
    arr = np.asarray(param, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return default
    if arr.size > 1 and not np.allclose(arr, arr[0], rtol=1e-9, atol=0.0):
        raise ValueError(
            f"neuron carries {arr.size} non-uniform parameter values "
            f"(spread {float(arr.max() - arr.min()):.3g}); the snnTorch "
            f"cross-check path only handles a shared parameter set")
    return float(arr[0])


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


def avgpool_module(node: Any) -> nn.Module:
    """Rebuild an average pool ourselves - see ``AVGPOOL_OVERRIDE`` above."""
    k = node.kernel_size
    s = node.stride
    p = getattr(node, "padding", 0)
    k = (int(k), int(k)) if np.isscalar(k) else tuple(int(v) for v in k)
    s = k if s is None else ((int(s), int(s)) if np.isscalar(s) else tuple(int(v) for v in s))
    p = (int(p), int(p)) if np.isscalar(p) else tuple(int(v) for v in p)
    return nn.AvgPool2d(kernel_size=k, stride=s, padding=p)


def reset_states(net: nn.Module) -> None:
    """Zero the membrane of every snnTorch neuron under ``net``.

    Feeding samples one at a time through a stateful graph carries the previous
    sample's membrane into the next one. That is correct for a streaming chip
    and wrong for a per-sample comparison, and it showed up as exactly one
    output spike in eight differing between nirtorch and this project.
    """
    for module in net.modules():
        mem = getattr(module, "mem", None)
        if torch.is_tensor(mem):
            module.mem = torch.zeros_like(mem)


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
        if kind == "AvgPool2d":
            return avgpool_module(node)
        if kind in NEURON_TYPES:
            return build_neuron(node, dt, log, kind)
        # ``DEFAULT_MAP`` is a *dict keyed by node type*, not the callable older
        # nirtorch releases accepted. Calling it raised ``TypeError`` which the
        # blanket except turned into ``NotImplementedError(kind)``, so every
        # unmapped op - ``Flatten``, ``AvgPool2d``, ``Conv2d`` - looked like an
        # upstream gap. It was this line.
        factory = DEFAULT_MAP.get(type(node))
        if factory is None:
            raise NotImplementedError(kind)
        return factory(node)

    return mapper
