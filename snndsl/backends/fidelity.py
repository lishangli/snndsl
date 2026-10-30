"""保真报告：同一个 ``.nir``，几种都算合规的实现，跑出几条不同的脉冲。

为什么需要这个（``docs/ECOSYSTEM_GAPS.md`` §5C）：NIR 论文把跨平台差异测了一遍，
然后停在论文里，没有工具把它做成一条命令。这里做的是其中不需要硬件的那几格 ——
而它们恰好是最常被归因错的那几格。「上板掉了三个点，是量化吧」多数时候不是答案，
只是最容易被想到的那个。

基线是这个**文件本身**在 float32 下的语义，不是 PyTorch。理由：文件才是要上板的
东西，PyTorch 只是它的产地。产地到文件的差异归 ``snndsl deploy --verify`` 管，
这里只回答「文件到手之后，还能怎么变」。

两条都合规却给出不同脉冲的机制，这里只覆盖两条：

``int8 / int4 权重量化``
    对称、逐通道。这是厂商做得最多的一格，所以它的数字最容易被拿来当替罪羊。

``tau 的离散化：欧拉 vs 指数``
    NIR 的 LIF 写的是 ``v ← (1 − dt/tau)·v + (dt/tau)·r·i``，即欧拉法。而
    「把 tau 当作连续时间常数」的自然实现是 ``v ← e^(−dt/tau)·v + …``。两者在
    tau ≫ dt 时几乎相同，tau 接近 dt 时差出一个数量级 —— 而**文件里没有字段说明
    该用哪一种**，所以两边都能声称自己读懂了这份文件。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from ..errors import SNNError

__all__ = ["FidelityError", "Report", "Variant", "format_report", "report", "report_file"]

_NEURONS = ("LIF", "LI", "IF", "CubaLIF", "CubaLI")
_WEIGHTED = ("Affine", "Linear", "Conv1d", "Conv2d")


class FidelityError(SNNError):
    """这份图不在「能给出有意义数字」的范围里。"""


@dataclass
class Variant:
    """一种实现选择，以及它把脉冲推到多远。"""

    name: str
    note: str
    bit_flip: float = 0.0          # 逐位置不一致的百分比，跨样本均值
    median: float = 0.0
    worst: float = 0.0
    top1: float = 1.0              # 读出层 argmax 与基线一致的样本比例
    layers: Dict[str, float] = field(default_factory=dict)

    def row(self, width: int = 26) -> str:
        return (f"{self.name:<{width}} {self.bit_flip:6.2f}% {self.median:7.2f}% "
                f"{self.worst:7.2f}% {100 * self.top1:7.1f}%")


@dataclass
class Report:
    path: str
    steps: int
    samples: int
    nodes: int
    dt: float
    variants: List[Variant] = field(default_factory=list)
    first_to_break: List[str] = field(default_factory=list)

    @property
    def baseline_silent(self) -> bool:
        return not self.variants or all(v.bit_flip == 0.0 for v in self.variants)


def _quantise(graph: Any, bits: int, per_channel: bool = True) -> Any:
    """对称量化权重，然后还原成 float，好在同一个解释器里跑。

    量化误差是这一步唯一想引入的东西，所以 scale 之外不动任何字节。
    """
    out = copy.deepcopy(graph)
    levels = float(2 ** (bits - 1) - 1)
    for node in out.nodes.values():
        weight = getattr(node, "weight", None)
        if weight is None:
            continue
        w = np.asarray(weight, dtype=np.float32)
        if per_channel and w.ndim >= 2:
            flat = w.reshape(w.shape[0], -1)
            scale = np.abs(flat).max(axis=1).reshape(-1, *([1] * (w.ndim - 1)))
        else:
            scale = np.abs(w).max()
        scale = np.maximum(np.asarray(scale, dtype=np.float32), 1e-12)
        q = np.clip(np.round(w / scale * levels), -levels, levels)
        node.weight = (q * scale / levels).astype(np.float32)
    return out


def _retime(graph: Any, dt: float) -> Any:
    """把欧拉衰减换成指数衰减，保持同一个连续时间常数。"""
    out = copy.deepcopy(graph)
    for node in out.nodes.values():
        if type(node).__name__ not in _NEURONS:
            continue
        tau = np.asarray(node.tau, dtype=np.float64)
        if not np.all(tau > 0):
            continue
        # 1 − dt/tau2 = e^(−dt/tau)  ⇒  tau2 = dt / (1 − e^(−dt/tau))
        beta = np.exp(-dt / tau)
        tau2 = dt / np.maximum(1.0 - beta, 1e-12)
        node.tau = tau2.astype(np.float32)
        r = getattr(node, "r", None)
        if r is not None and np.allclose(np.asarray(r), tau):
            # r 与 tau 一起出现时是同一个量的两次书写，改一个不改另一个等于改语义
            node.r = tau2.astype(np.float32)
    return out


def _input_shape(graph: Any) -> Optional[tuple]:
    for node in graph.nodes.values():
        if type(node).__name__ != "Input":
            continue
        typ = getattr(node, "input_type", None)
        if isinstance(typ, dict):
            for value in typ.values():
                if value is not None:
                    dims = tuple(int(x) for x in np.asarray(value).reshape(-1))
                    if dims:
                        return dims
    return None


def _spikes(sim: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v) for k, v in sim.get("spikes", {}).items() if np.asarray(v).size}


def report(graph: Any,
           steps: int,
           samples: int = 40,
           seed: int = 0,
           dt: float = 1.0,
           sample: Any = None,
           path: str = "<memory>") -> Report:
    """跑一遍基线，再跑每一种扰动，逐位置比脉冲。"""
    from . import nir_sim

    steps = int(steps)
    if steps <= 0:
        raise FidelityError("steps must be positive")

    shape = _input_shape(graph)
    if shape is None and sample is None:
        raise FidelityError("cannot infer the input shape: the graph has no typed Input node. "
                            "Pass an explicit sample with --input.")
    rng = np.random.default_rng(seed)

    variants = [
        ("int8 权重（逐通道）", "对称量化，权重", _quantise(graph, 8, True)),
        ("int4 权重（逐通道）", "对称量化，权重", _quantise(graph, 4, True)),
        ("tau：欧拉 → 指数", "同一连续时间常数，两种离散化", _retime(graph, dt)),
    ]

    stats: Dict[str, Dict[str, List[float]]] = {
        name: {"bit": [], "top1": [], "layers": {}} for name, _, _ in variants
    }
    for _ in range(int(samples)):
        x = sample if sample is not None else (rng.standard_normal(shape) * 2.0).astype(np.float32)
        base = nir_sim.simulate(graph, x, steps=steps, dt=dt)
        sp0 = _spikes(base)
        # An empty *dict* and a dict of all-zero traces are different failures and
        # both mean the same thing: nothing fired, so every comparison below would
        # be a comparison against silence.
        if not sp0 or sum(float(v.sum()) for v in sp0.values()) == 0.0:
            raise FidelityError("the baseline fires nothing, so every comparison against it "
                                "would be a comparison against silence")
        total = sum(v.size for v in sp0.values())
        out0 = np.asarray(nir_sim.readout(graph, base, mode="spikes")).reshape(-1)

        for name, _, perturbed in variants:
            sim = nir_sim.simulate(perturbed, x, steps=steps, dt=dt)
            sp = _spikes(sim)
            flipped = sum(int((sp[k] != sp0[k]).sum()) for k in sp0 if k in sp)
            stats[name]["bit"].append(100.0 * flipped / total)
            out = np.asarray(nir_sim.readout(perturbed, sim, mode="spikes")).reshape(-1)
            stats[name]["top1"].append(float(out0.argmax() == out.argmax()))
            for k in sp0:
                if k not in sp:
                    continue
                err = 100.0 * float((sp[k] != sp0[k]).sum()) / sp0[k].size
                stats[name]["layers"].setdefault(k, []).append(err)

    out_variants: List[Variant] = []
    for name, note, _ in variants:
        bits = np.asarray(stats[name]["bit"])
        out_variants.append(Variant(
            name=name,
            note=note,
            bit_flip=float(bits.mean()),
            median=float(np.median(bits)),
            worst=float(bits.max()),
            top1=float(np.mean(stats[name]["top1"])),
            layers={k: float(np.mean(v)) for k, v in stats[name]["layers"].items()},
        ))

    worst_variant = max(out_variants, key=lambda v: v.bit_flip)
    order = sorted(worst_variant.layers.items(), key=lambda kv: -kv[1])

    return Report(path=path, steps=steps, samples=int(samples), nodes=len(graph.nodes),
                  dt=dt, variants=out_variants,
                  first_to_break=[k for k, v in order if v > 0.0])


def report_file(path: str, steps: Optional[int] = None, **kwargs: Any) -> Report:
    """从 ``.nir`` 读，若里面记录了 T 就用它。"""
    try:
        import nir
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise FidelityError("reading a .nir file needs the `nir` package: "
                            "pip install 'snndsl[nir]'") from exc

    graph = nir.read(path)
    meta = dict(getattr(graph, "metadata", {}) or {})
    if steps is None:
        recorded = meta.get("snndsl.timesteps")
        if recorded is not None:
            steps = int(float(recorded))
    if steps is None:
        raise FidelityError("this file does not record a step count. Pass --steps, or run "
                            "`snndsl fold` first, which writes metadata['snndsl.timesteps'].")
    return report(graph, steps=steps, path=path, **kwargs)


def format_report(rep: Report) -> str:
    lines = [
        f"fidelity  {rep.path}  ·  {rep.steps} 步  ·  {rep.samples} 组输入  ·  "
        f"{rep.nodes} 节点  ·  dt={rep.dt:g}",
        "基线 = 本文件在 float32 下的语义（不是 PyTorch：产地到文件的差异归 deploy --verify）",
        "",
        f"{'扰动':<20} {'脉冲错位':>9} {'中位':>8} {'最坏':>8} {'top-1 一致':>10}",
    ]
    lines.append("-" * 62)
    for v in rep.variants:
        lines.append(f"{v.name:<20} {v.bit_flip:8.2f}% {v.median:7.2f}% "
                     f"{v.worst:7.2f}% {100 * v.top1:9.1f}%")
    if rep.first_to_break:
        worst = max(rep.variants, key=lambda v: v.bit_flip)
        head = " > ".join(f"{k} ({worst.layers[k]:.2f}%)" for k in rep.first_to_break[:4])
        lines += ["", f"错得最多的一档：{worst.name}", f"  层序（先崩的在前）：{head}"]
    if all(v.bit_flip == 0.0 for v in rep.variants):
        lines += ["", "所有扰动下脉冲逐位置相同 —— 这个网络对这些变化不敏感，"
                      "不能据此说它「适合部署」，只能说这几档没测出区别"]
    return "\n".join(lines)
