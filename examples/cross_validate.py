"""Cross-validate our exported .nir against third-party implementations.

The claim "the exported network behaves like the trained network" is worthless
if the only implementation checking it is ours. This script checks it with
components we did not write:

    1. nir        - reads the .nir file            (neuromorphs/NIR, official)
    2. nirtorch   - maps and executes the graph    (neuromorphs/NIRTorch)
    3. snnTorch   - provides the LIF neuron        (jeshraghian/snntorch)
    4. torch      - provides the linear layers

NIRTorch's own ``DEFAULT_MAP`` supplies the Linear/Affine/Conv/Pool/Flatten
modules, so those layers are neither our implementation nor our mapping. The
only thing this project contributes is the *parameter mapping* for the neuron:
``beta = 1 - dt/tau``, the current gain ``dt/tau*r`` and the leak offset. The
integration, threshold comparison and reset are executed by ``snn.Leaky``.
That mapping is printed with the result so it can be audited.

    python3 examples/cross_validate.py
    python3 examples/cross_validate.py --nir build_hw/mnist_hw.nir \
        --checkpoint build_hw/mnist_hw.pt --program examples/mnist_hw.snn
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import nir
import nirtorch
import numpy as np
import snntorch as snn
import torch
import torch.nn as nn

# NIRTorch's own mappings for the non-spiking layers.
from nirtorch.nir_interpreter import DEFAULT_MAP
from snndsl import compile_source, load_module

NEURON_TYPES = (nir.LIF, nir.LI, nir.IF)


# --------------------------------------------------------------- NIR -> torch
def _const(param, default: float) -> float:
    arr = np.asarray(param, dtype=np.float64).reshape(-1)
    return float(arr[0]) if arr.size else default


class AdaptedLIF(nn.Module):
    """A snnTorch neuron plus the two numbers NIR needs but snnTorch lacks.

    NIR LIF:  tau dv/dt = (v_leak - v) + r*i
    snnTorch: mem <- beta*mem + input

    Discretising NIR with step ``dt`` gives

        v[k+1] = (1 - dt/tau) v[k] + (dt/tau)(v_leak + r i[k])

    so ``beta = 1 - dt/tau`` and the drive is ``(dt/tau)*(v_leak + r*i)``.
    Only that gain and offset live here; the membrane update, threshold
    comparison and reset are performed inside ``snn.Leaky``.
    """

    def __init__(self, neuron: nn.Module, gain: float, offset: float):
        super().__init__()
        self.neuron = neuron
        self.gain = float(gain)
        self.offset = float(offset)
        self.neuron.init_leaky()          # single-argument (stateful) mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.neuron(self.gain * x + self.offset)


def build_neuron(node, dt: float, log: list, node_name: str) -> nn.Module:
    """Turn one NIR neuron into a snnTorch module."""
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

    # snnTorch's Leaky can only reset to zero, so a non-zero v_reset is a
    # different neuron and must not be silently approximated.
    if abs(v_reset) > 1e-9:
        raise ValueError(
            f"{node_name}: v_reset={v_reset} is not zero; snnTorch's Leaky can "
            f"only reset to zero, so this graph cannot be cross-validated with "
            f"this backend")

    # reset_delay=False matters. snnTorch defaults to True, which applies the
    # reset decided by the *previous* step, so a neuron can fire again from an
    # un-reset membrane. NIR resets on the step it fires.
    neuron = snn.Leaky(beta=beta, threshold=v_th, reset_mechanism="zero",
                       init_hidden=True, reset_delay=False)

    log.append({
        "node": node_name,
        "type": kind,
        "snntorch": "snn.Leaky",
        "beta": round(beta, 6),
        "threshold": v_th,
        "reset": "zero",
        "input_gain": round(gain, 6),
        "input_offset": round(offset, 6),
        "who_integrates": "snnTorch",
    })
    if abs(gain - 1.0) < 1e-9 and abs(offset) < 1e-9:
        return neuron                  # unmodified third-party module
    return AdaptedLIF(neuron, gain, offset)


def _affine(node) -> nn.Module:
    """Rebuild an Affine ourselves - see ``AFFINE_OVERRIDE`` for why."""
    w = torch.as_tensor(np.asarray(node.weight), dtype=torch.float32)
    b = torch.as_tensor(np.asarray(node.bias), dtype=torch.float32)
    module = nn.Linear(w.shape[1], w.shape[0], bias=True)
    with torch.no_grad():
        module.weight.copy_(w)
        module.bias.copy_(b)
    return module


# nirtorch 2.6 ships this as its DEFAULT_MAP entry for ``nir.Affine``:
#
#     module.weight.data = torch.from_numpy(linear.weight)
#     module.weight.bias = torch.from_numpy(linear.bias)   # <- not module.bias
#
# The second line assigns to a throw-away attribute, so ``module.bias`` keeps
# its random initialisation and the converted model silently loses every bias.
# Measured here: agreement drops from 100% to 98.73% on 1024 MNIST samples,
# with no error raised anywhere. We therefore map Affine ourselves and say so.
AFFINE_OVERRIDE = ("nirtorch 2.6 DEFAULT_MAP sets module.weight.bias instead of "
                   "module.bias, so biases stay random; Affine is mapped here "
                   "until that is fixed upstream")


def make_model_map(dt: float, log: list, names: dict):
    """Neurons from snnTorch, other layers from NIRTorch's DEFAULT_MAP."""

    def model_map(node):
        kind = type(node)
        if kind in NEURON_TYPES:
            return build_neuron(node, dt, log, names.get(id(node), "?"))
        if kind is nir.Affine:
            log.append({"node": names.get(id(node), "?"), "type": "Affine",
                        "mapping": "ours", "reason": AFFINE_OVERRIDE})
            return _affine(node)
        fn = DEFAULT_MAP.get(kind)
        return fn(node) if fn else None    # None -> nirtorch's identity fallback

    return model_map


# ---------------------------------------------------------------------- driver
def reset_state(model: nn.Module) -> None:
    """Clear every neuron's membrane between batches.

    snnTorch keeps ``mem`` as an instance attribute, so a model reused across
    batches starts batch 2 with batch 1's membrane. That manufactures a
    mismatch that has nothing to do with the exported file: the first batch
    matches and every later one drifts.
    """
    for m in model.modules():
        if isinstance(m, AdaptedLIF):
            m.neuron.init_leaky()
        elif isinstance(m, snn.Leaky):
            m.init_leaky()


def build_third_party(nir_path: str, dt: float):
    graph = nir.read(nir_path)
    log: list = []
    names = {id(node): name for name, node in graph.nodes.items()}
    # nirtorch 2.6's documented replacement, ``nir_to_torch``, does not yet
    # handle a snnTorch neuron's ``(spike, membrane)`` return: it wires the
    # tuple straight into the next Linear and raises. ``load`` carries an
    # explicit snnTorch branch, which is why it works with the neuron backend
    # NIRTorch's own docs recommend.
    model = nirtorch.load(graph, make_model_map(dt, log, names), return_state=False)
    model.eval()
    return graph, model, log


def run(nir_path: str, program: str, checkpoint: str, dt: float,
        steps: int, samples: int, batch: int, data: str) -> dict:
    _graph, third, mapping = build_third_party(nir_path, dt)

    # ---- our own implementation, for the comparison
    result = compile_source(open(program, encoding="utf-8").read())
    net_cls = load_module(result, os.path.join(
        os.path.dirname(os.path.abspath(nir_path)), "_cross_net.py"))
    net = net_cls()
    blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(blob.get("state_dict", blob))
    net.eval()

    from snndsl.backends import runtime as rt
    shape = tuple(int(v) for v in result.graph.node(result.graph.inputs[0]).shape)
    loaders = rt.build_dataloaders({"data": data, "seed": 0}, shape, batch_size=batch)
    test = loaders["test"] if isinstance(loaders, dict) else loaders[-1]

    agree = total = correct_ours = correct_third = 0
    max_spike_diff = 0.0
    for xb, yb in test:
        xb, yb = xb[:batch], yb[:batch]
        flat = xb.reshape(xb.shape[0], -1)

        with torch.no_grad():
            ours = net(xb).detach()

        # ---- third party: T forward passes, each one step of the graph.
        # Neurons hold state, so clear it per batch or only the first batch is
        # actually being compared.
        reset_state(third)
        acc = torch.zeros_like(ours)
        for _ in range(steps):
            acc = acc + third(flat)
        thirdparty = acc / steps

        ours_p = ours.argmax(1)
        third_p = thirdparty.argmax(1)
        agree += int((ours_p == third_p).sum())
        correct_ours += int((ours_p == yb).sum())
        correct_third += int((third_p == yb).sum())
        total += int(yb.numel())
        with torch.no_grad():
            max_spike_diff = max(max_spike_diff,
                                 float((ours - thirdparty).detach().abs().max()))
        if total >= samples:
            break

    return {
        "nir_file": nir_path,
        "steps": steps,
        "samples": total,
        "accuracy_ours": round(correct_ours / total, 6),
        "accuracy_third_party": round(correct_third / total, 6),
        "prediction_agreement": round(agree / total, 6),
        "max_abs_output_diff": round(max_spike_diff, 9),
        "backend": {"graph": "nirtorch.load",
                    "layers": "nirtorch DEFAULT_MAP (Affine overridden, see notes)",
                    "neuron": "snnTorch snn.Leaky",
                    "affine_override": AFFINE_OVERRIDE},
        "parameter_mapping": mapping,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nir", default="build_hw/mnist_hw.nir")
    ap.add_argument("--checkpoint", default="build_hw/mnist_hw.pt")
    ap.add_argument("--program", default="examples/mnist_hw.snn")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--samples", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--data", default="mnist")
    ap.add_argument("--out", default=None, help="write the result as JSON here")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    out = run(args.nir, args.program, args.checkpoint, args.dt,
              args.steps, args.samples, args.batch, args.data)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return 0 if out["prediction_agreement"] == 1.0 else 1

    print("--- third-party cross-validation ---")
    print(f"  artifact      : {out['nir_file']}")
    print(f"  graph engine  : {out['backend']['graph']}")
    print(f"  layers        : {out['backend']['layers']}")
    print(f"  neuron        : {out['backend']['neuron']}")
    print(f"  steps         : {out['steps']}")
    print(f"  samples       : {out['samples']}")
    print("\n  parameter mapping (ours -> snnTorch):")
    for m in out["parameter_mapping"]:
        if "beta" in m:
            print(f"    {m['node']:<10s} beta={m['beta']}  th={m['threshold']}  "
                  f"gain={m['input_gain']}  offset={m['input_offset']}")
        else:
            print(f"    {m['node']:<10s} {m['type']}: mapped by us - {m['reason'][:72]}")
    print()
    print(f"  accuracy ours        : {out['accuracy_ours'] * 100:.2f}%")
    print(f"  accuracy third-party : {out['accuracy_third_party'] * 100:.2f}%")
    print(f"  prediction agreement : {out['prediction_agreement'] * 100:.2f}%")
    print(f"  max |output diff|    : {out['max_abs_output_diff']:.3e}")
    verdict = "CONSISTENT" if out["prediction_agreement"] == 1.0 else "MISMATCH"
    print(f"  verdict              : {verdict}")
    return 0 if verdict == "CONSISTENT" else 1


if __name__ == "__main__":
    sys.exit(main())
