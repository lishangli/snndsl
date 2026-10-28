"""Can a *foreign* toolchain actually eat the file we just wrote?

`deploy.verify` answers "is the file semantically what we trained". This module
answers a different and, for a team about to buy silicon, harder question:
*will the thing that runs on the chip be able to open it?*

Those two answers come apart in practice. A file can be perfectly spec-conformant
and still be rejected by every shipping frontend, because the standard moved and
the frontends did not. Two failures of exactly that kind were found while
building this check, and both are recorded verbatim rather than smoothed over:

``nir`` 1.0.4 vs 1.0.8
    ``LIF.v_reset`` was added in a later 1.0 patch. A file written by 1.0.8 that
    carries the field raises ``TypeError: LIF.__init__() got an unexpected
    keyword argument 'v_reset'`` on 1.0.4. SynSense's rockpool and sinabs pin the
    older release, so the *newest* writer produces files the vendor cannot read.
    ``nir_export`` now drops a zero ``v_reset`` (the standard's own default) so
    one file loads across the whole line.

``rockpool`` 3.1.0
    Its NIR frontend still reads ``node.tau_mem``, renamed to ``tau`` in nir 1.0.
    Every nir 1.0 LIF fails with ``AttributeError: 'LIF' object has no attribute
    'tau_mem'``. Nothing on our side can fix that; the check exists so a user
    finds out in a report instead of on a Friday night.

Probes run in a **subprocess**, because the vendor stacks pin a different nir
than we do and importing them in-process silently downgrades the one we are
testing against. That is not a hypothetical: installing rockpool and sinabs
dropped nir 1.0.8 to 1.0.4 and broke 14 of our own tests.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Interpreter that has the vendor stack installed. Kept separate on purpose:
# `nir` is pinned differently by each vendor, so no single environment can hold
# them all at the version we export against.
VENDOR_PYTHON_ENV = "SNNDSL_VENDOR_PYTHON"


@dataclass
class TargetResult:
    key: str
    label: str
    vendor: str
    status: str                      # ok | load-only | upstream-gap | not-installed | error
    detail: str = ""
    nir_version: str = ""
    numbers: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def usable(self) -> bool:
        return self.status in ("ok", "load-only")


# --------------------------------------------------------------------- probes
_SINABS = r"""
import json, sys
import numpy as np, nir, torch
p = json.load(sys.stdin)
out = {"nir_version": nir.__version__}
from sinabs.nir import from_nir
g = nir.read(p["nir"])
T, B = int(p["steps"]), int(p["batch"])
x = np.load(p["inputs"]).astype("float32").reshape(B, -1)
cur_t = torch.as_tensor(x)

# ---- sinabs' NIR importer builds LIF layers that want (batch, time, features)
# while the squeeze wrappers it also builds want 4-D input. Try the sequence
# convention first; fall back to stepping the module by hand. Whichever runs is
# recorded, because "it loaded" and "it runs the way we trained it" are not the
# same claim.
conv, acc = None, None
try:
    net = from_nir(g, batch_size=B, num_timesteps=T)
    net.eval()
    with torch.no_grad():
        y = net(cur_t.reshape(B, 1, -1).expand(B, T, -1).contiguous())
    spk = y[0] if isinstance(y, tuple) else y
    acc = torch.as_tensor(spk).detach().float().sum(1)
    conv = "sequence (batch, time, features)"
except Exception:
    net = from_nir(g, batch_size=B, num_timesteps=1)
    net.eval()
    with torch.no_grad():
        for _ in range(T):
            y = net(cur_t)
            spk = torch.as_tensor(y[0] if isinstance(y, tuple) else y).detach().float()
            acc = spk if acc is None else acc + spk
    conv = "per-step, stateful"

rates = (acc / T).numpy()
out["convention"] = conv
out["multi_spike"] = float(np.abs(rates - np.round(rates)).max()) > 1e-6 or \
                     bool(rates.max() > 1.0)
ref = np.load(p["reference"]).astype("float32") if p.get("reference") else None
if ref is not None:
    out["max_abs_error"] = float(np.abs(rates - ref).max())
    out["top1_agreement"] = float((rates.argmax(1) == ref.argmax(1)).mean())
    out["spike_rate"] = float(rates.mean())
    out["silent"] = bool(not np.any(rates))
# spikes above one per neuron per step are not binary event streams; the vendor
# is running a different time model, so a numeric match is not claimed
out["status"] = "load-only" if out["multi_spike"] else "ok"
print(json.dumps(out))
"""

_ROCKPOOL = r"""
import json, sys
import nir
p = json.load(sys.stdin)
out = {"nir_version": nir.__version__}
from rockpool.nn.modules.torch.nir import from_nir
from_nir(p["nir"])
out["status"] = "load-only"
print(json.dumps(out))
"""

# rockpool's own NIR frontend cannot read nir 1.0 (it still asks for ``tau_mem``),
# so this probe rebuilds the graph out of rockpool's *native* modules instead.
# The arithmetic is therefore SynSense's, not ours; only the wiring is ours.
# It runs the graph twice, once with each convention for turning the file's
# ``tau`` into a decay factor, because that convention is not in the standard
# and the two give measurably different spike trains.
_ROCKPOOL_NATIVE = r"""
import json, sys
import numpy as np, nir, torch
from rockpool.nn.modules import LinearTorch
from rockpool.nn.modules.torch.lif_neuron_torch import LIFNeuronTorch

p = json.load(sys.stdin)
out = {"nir_version": nir.__version__}
first = lambda o: o[0] if isinstance(o, tuple) else o
g = nir.read(p["nir"])
dt = float(p["dt"])

# ---- walk the chain from Input; anything we do not understand ends the probe
# with a reason instead of a wrong number
succ = {}
for a, b in g.edges:
    succ.setdefault(a, b)
order, cur = [], [k for k, v in g.nodes.items() if type(v).__name__ == "Input"]
cur = cur[0] if cur else None
while cur is not None and cur in g.nodes:
    order.append(cur)
    cur = succ.get(cur)
ops = []
for nid in order:
    node = g.nodes[nid]
    kind = type(node).__name__
    if kind in ("Input", "Output"):
        continue
    if kind in ("Affine", "Linear"):
        w = np.asarray(node.weight, dtype="float32")
        b = np.asarray(node.bias, dtype="float32") if kind == "Affine" else None
        ops.append(("linear", w, b))
    elif kind == "LIF":
        tau = np.asarray(node.tau, dtype="float32").reshape(-1)
        vth = np.asarray(node.v_threshold, dtype="float32").reshape(-1)
        vr = getattr(node, "v_reset", None)
        ops.append(("lif", tau, vth, vr if vr is not None else 0.0))
    else:
        out.update(status="load-only", note=(
            "chain contains %s, which this probe does not rebuild" % kind))
        print(json.dumps(out)); sys.exit(0)

lifs = [o for o in ops if o[0] == "lif"]
if not lifs:
    out.update(status="load-only", note="no LIF layer to compare")
    print(json.dumps(out)); sys.exit(0)
tau0 = float(lifs[0][1][0])
declared_reset = float(np.max(np.abs(np.asarray(lifs[0][3], dtype="float32"))))
out["declared_reset"] = "zero" if declared_reset == 0.0 else "constant(%.3g)" % declared_reset

x = np.load(p["inputs"]).astype("float32").reshape(int(p["batch"]), -1)
T = int(p["steps"])
xseq = torch.as_tensor(x).unsqueeze(1).expand(-1, T, -1).contiguous()


def run(tau_mem):
    # run the chain with one tau convention; returns (T, B, N) spike trains
    mods = []
    for op in ops:
        if op[0] == "linear":
            _, w, b = op
            lin = LinearTorch(shape=(w.shape[1], w.shape[0]),
                              weight=torch.as_tensor(w.T), has_bias=b is not None)
            if b is not None:
                with torch.no_grad():
                    lin.bias.copy_(torch.as_tensor(b))
            mods.append(("linear", lin))
        else:
            _, tau, vth, _ = op
            t = tau_mem if np.isscalar(tau_mem) else tau_mem[: len(tau)]
            mods.append(("lif", LIFNeuronTorch(
                shape=(len(tau),), tau_mem=float(np.asarray(t).reshape(-1)[0]),
                threshold=torch.as_tensor(vth), dt=dt, bias=0.0)))
    cur, outs = xseq, []
    with torch.no_grad():
        for kind, m in mods:
            cur = first(m(cur))
            if kind == "lif":
                cur = cur.reshape(cur.shape[0], cur.shape[1], -1)
                outs.append(cur.numpy().transpose(1, 0, 2))      # (T,B,N)
    return outs


def compare(trains, ref):
    if ref is None:
        return {}
    ref = ref.reshape(trains.shape)
    return {"mismatch": float((trains != ref).mean()),
            "ref_rate": float(ref.mean()), "got_rate": float(trains.mean())}


ref = np.load(p["trains"]).astype("float32") if p.get("trains") else None
naive = run(tau0)
# the two conventions: tau is a physical time constant, so the decay factor is
# exp(-dt/tau); Euler integration gives 1 - dt/tau. Same number in the file,
# different dynamics, and nothing in NIR says which one the writer meant.
alpha = max(1e-6, 1.0 - dt / tau0)
comp = run(-dt / np.log(alpha))

out["conventions"] = {
    "direct": {"tau_mem": tau0, "alpha": float(np.exp(-dt / tau0))},
    "euler_matched": {"tau_mem": float(-dt / np.log(alpha)), "alpha": alpha},
}
for name, trains in (("direct", naive), ("euler_matched", comp)):
    tail = trains[-1]
    out[name] = compare(tail, ref)
out["multi_spike_fraction"] = float((comp[-1] > 1.0).mean())
out["spike_rate"] = float(comp[-1].mean())
out["layers"] = len(ops)
# how much the tau convention alone moves the result: both runs are rockpool's
# own neuron, so reset form and spike rule are held fixed
out["tau_sensitivity"] = float((naive[-1] != comp[-1]).mean())

if ref is None:
    out.update(status="load-only", note="ran, but no reference spike train to compare")
elif float(np.abs(ref).sum()) == 0.0:
    # 参考没有脉冲时，"0 等于 0"不携带任何信息；这类伪阳性在 deploy 里已经
    # 出现过一次（噪声输入下网络不发放），这里不能重犯
    out.update(status="load-only",
               note="参考脉冲为空，这次比较不携带信息（加大电流或换输入）")
else:
    bits = []
    if out["multi_spike_fraction"] > 0:
        bits.append("%.1f%% 的 (神经元,时间步) 一步发多枚脉冲（NIR 未禁止、我们的 LIF 不发）"
                    % (100 * out["multi_spike_fraction"]))
    if out["declared_reset"] == "zero":
        bits.append("文件声明 zero reset，rockpool 的神经元只有 subtract")
    bits.append("单独把 tau 约定从 exp 换成 Euler，脉冲位置变化 %.2f%%"
                % (100 * out["tau_sensitivity"]))
    bits.append("逐脉冲一致率 %.2f%%" % (100 * (1 - out["euler_matched"]["mismatch"])))
    if out["multi_spike_fraction"] == 0 and out["euler_matched"]["mismatch"] == 0.0:
        out["status"] = "ok"
    else:
        out.update(status="load-only", note="已用 rockpool 自己的神经元跑通；" + "；".join(bits))
print(json.dumps(out))
"""

_NIRTOTCH = r"""
import json, sys
import numpy as np, nir, nirtorch, torch
p = json.load(sys.stdin)
sys.path.insert(0, p["root"])
out = {"nir_version": nir.__version__}
g = nir.read(p["nir"])

# ---- semantic glue (beta, current gain, zero-reset, the bias that nirtorch's
# DEFAULT_MAP drops) lives in one audited place, not in this string.
from snndsl.backends.snntorch_bridge import make_map
net = nirtorch.load(g, make_map(float(p["dt"])))
net.eval()
T, B = int(p["steps"]), int(p["batch"])
x = np.load(p["inputs"]).astype("float32").reshape(B, -1)
cur = torch.as_tensor(x)
acc = None
with torch.no_grad():
    for _ in range(T):
        y = net(cur)
        spk = y[0] if isinstance(y, tuple) else y
        spk = torch.as_tensor(spk).detach().float()
        acc = spk if acc is None else acc + spk
rates = (acc / T).numpy()
ref = np.load(p["reference"]).astype("float32")
out["max_abs_error"] = float(np.abs(rates - ref).max())
out["top1_agreement"] = float((rates.argmax(1) == ref.argmax(1)).mean())
out["spike_rate"] = float(rates.mean())
out["silent"] = bool(not np.any(acc.numpy()))
out["status"] = "ok"
print(json.dumps(out))
"""

_PROBES = {
    "nirtorch": (_NIRTOTCH, "nirtorch + snnTorch", "neuroir"),
    "sinabs": (_SINABS, "sinabs (Speck / DynapCNN)", "SynSense"),
    "rockpool": (_ROCKPOOL, "rockpool (Xylo) frontend", "SynSense"),
    "rockpool-native": (_ROCKPOOL_NATIVE, "rockpool modules (Xylo)", "SynSense"),
}


def _python_for(key: str) -> str:
    """Interpreter to run a vendor probe in.

    Falls back to the current one, which is why a "not-installed" result is
    reported as such rather than as a failure of the file.
    """
    return os.environ.get(VENDOR_PYTHON_ENV) or sys.executable


def probe(nir_path: str,
          inputs: Optional[str] = None,
          reference: Optional[str] = None,
          steps: int = 8,
          batch: int = 4,
          dt: float = 1.0,
          targets: Optional[List[str]] = None,
          trains: Optional[str] = None,
          timeout: int = 600) -> List[TargetResult]:
    """Run every requested foreign frontend against ``nir_path``.

    With ``inputs``/``reference`` (npy files) the probe also executes the graph
    and scores it; without them it only asks whether the file loads.
    ``trains`` is an optional (T, B, N) array of our own per-step spikes; a
    probe that can produce per-step spikes itself uses it to compare fires
    position by position, which is a far stricter test than comparing argmaxes.
    """
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results: List[TargetResult] = []
    for key in (targets or list(_PROBES)):
        if key not in _PROBES:
            continue                     # unknown target names are ignored
        src, label, vendor = _PROBES[key]
        payload = {"nir": os.path.abspath(nir_path), "steps": steps, "batch": batch,
                   "dt": dt, "inputs": inputs, "reference": reference,
                   "trains": trains, "root": root}
        if key in ("nirtorch", "sinabs") and (not inputs or not reference):
            payload.update({"inputs": None, "reference": None})
        try:
            proc = subprocess.run(
                [_python_for(key), "-c", src],
                input=json.dumps(payload), capture_output=True, text=True,
                timeout=timeout, cwd=os.path.dirname(os.path.abspath(nir_path)) or None)
        except subprocess.TimeoutExpired:
            results.append(TargetResult(key, label, vendor, "error",
                                        f"probe exceeded {timeout}s"))
            continue
        except OSError as err:
            # the interpreter itself is missing — a setup problem, not a verdict
            # on the file, so say which one and keep probing the rest
            results.append(TargetResult(key, label, vendor, "not-installed",
                                        f"cannot run {_python_for(key)}: {err.strerror}"))
            continue
        if proc.returncode != 0:
            err = (proc.stderr or "").strip().splitlines()
            tail = err[-1] if err else "no output"
            missing = "ModuleNotFoundError" in (proc.stderr or "")
            results.append(TargetResult(
                key, label, vendor,
                "not-installed" if missing else "upstream-gap",
                _explain(tail),
                _nir_version_from(proc.stdout)))
            continue
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:                    # noqa: BLE001
            results.append(TargetResult(key, label, vendor, "error",
                                        "probe printed no JSON"))
            continue
        status = data.pop("status", "ok")
        nirv = data.pop("nir_version", "")
        if status == "ok" and data.get("silent"):
            # a network that emits nothing agrees with nothing
            status = "load-only"
            data["note"] = "loaded, ran, but emitted no spikes on this input"
        results.append(TargetResult(key, label, vendor, status, "", nirv, data))
    return results


def _nir_version_from(stdout: str) -> str:
    for line in (stdout or "").splitlines():
        line = line.strip()
        if line.startswith("{") and "nir_version" in line:
            try:
                return json.loads(line).get("nir_version", "")
            except Exception:                # noqa: BLE001
                return ""
    return ""


def _explain(tail: str) -> str:
    """Turn a traceback's last line into what a user needs to hear."""
    if "tau_mem" in tail:
        return ("upstream frontend still reads nir < 1.0 field names "
                "(tau_mem); cannot load any nir 1.0 LIF")
    if "unexpected keyword argument" in tail:
        return "the installed nir is older than the one this file was written by"
    if "ModuleNotFoundError" in tail:
        return "not installed in this interpreter"
    return tail[:220]


def format_matrix(results: List[TargetResult]) -> str:
    w = max((len(r.label) for r in results), default=20)
    lines = [f"{'target':<{w}}  {'vendor':<10}  {'status':<14}  detail"]
    lines.append("-" * (w + 60))
    for r in results:
        detail = r.detail
        if r.status == "ok":
            detail = (f"max|Δ|={r.numbers.get('max_abs_error', 0):.2e}  "
                      f"top1={r.numbers.get('top1_agreement', 0):.3f}  "
                      f"nir {r.nir_version}")
        elif r.numbers.get("note"):
            detail = r.numbers["note"]
        elif r.status == "load-only":
            detail = detail or (
                f"loaded and ran ({r.numbers.get('convention', '?')}), but not a "
                f"binary spike stream (peak {r.numbers.get('spike_rate', 0):.1f} "
                f"spikes/neuron/step) - semantics not comparable")
        lines.append(f"{r.label:<{w}}  {r.vendor:<10}  {r.status:<14}  {detail}")
    return "\n".join(lines)
