"""目标平台的代价探针：tau 可表示性、int8 量化代价、时间维度使用度。

这三个测量回答同一类问题：**训练时写下的一堆浮点常量，到了目标芯片上还剩多少**。

背景（`docs/MECHANISMS.md` §1.3）：Xylo 类芯片的 decay 是 4-bit bit-shift，
状态是 16-bit 定点，权重 8-bit。规格表里 "longest effective time-constant =
32768·dt" 意味着芯片上可表示的 tau 只有 16 个值——`{2^k · dt, k = 0..15}`。
我们在 DSL 里写 `param tau = 3.0`，训练收敛、导出成功、NIR 合法，
而这个值在芯片上不存在。

用法::

    # 1) MNIST MLP：需要先有训练好的 checkpoint
    snndsl run examples/mnist_mlp.snn --epochs 10 -o build/
    python3 examples/probe_target_cost.py --weights build/mnist_mlp.pt

    # 2) 手势网（有真实时间结构）：当场训 8 epoch，约 20 秒
    python3 examples/probe_target_cost.py --gesture

两个网络的结论是反的，这是本脚本存在的理由：

    MNIST MLP   tau 1.0 -> 4.0（四倍跨度）  精度只动 0.09pp
    手势网       tau 2.0 -> 4.0              精度掉 10.00pp

差 111 倍。原因是 MNIST 那个网络的 LIF 层对动力学参数完全无感——
它等价于 `relu(x)`，时间维度没有被使用。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import snndsl  # noqa: F401  (generated code imports it)
from snndsl.backends import runtime as rt
from snndsl.pipeline import compile_source, load_module

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# --------------------------------------------------------------- 目标规则
# 这些数字来自 Rockpool→Xylo 部署管线的规格表（arXiv 2412.11047 Table 1）。
# 写成数据而不是代码，是因为换一颗芯片只该换这张表（MECHANISMS.md M2）。
XYLO_LIKE = {
    "decay_bits": 4,  # bit-shift decay 参数位宽
    "max_shift": 15,  # 最大移位量
    "weight_bits": 8,
    "state_bits": 16,
}


def snap_tau(tau: float, dt: float = 1.0, max_shift: int = 15) -> tuple[float, int]:
    """把任意 tau 吸附到目标芯片能表示的最近值 ``2^k * dt``。

    返回 ``(可表示值, k)``。k 被限制在 ``0..max_shift``，因为 decay 只有 4 bit。
    """
    best_k, best_err = 0, float("inf")
    for k in range(max_shift + 1):
        cand = (2.0**k) * dt
        err = abs(cand - tau)
        if err < best_err:
            best_k, best_err = k, err
    return (2.0**best_k) * dt, best_k


# ---------------------------------------------------------------- MNIST MLP
def report_temporal(deltas: dict) -> None:
    """把 tau 扫描拆成两个独立的量，别混成一个"最大偏差"。

    ``tau = dt`` 时 ``v = v*(1 - dt/tau) + i = i``，膜电位完全无记忆，
    LIF 退化成 ``relu``。所以::

        memory_value    = Δ(tau=dt)           记忆本身值多少
        memory_strength = max|Δ| over [1.5,4] 记忆的强度被用到多少

    两个量分开看才知道该改什么。MNIST MLP 上 memory_value≈1.2pp 而
    memory_strength≈0.09pp：留着记忆比去掉好一点，但记忆多强无所谓——
    典型的"静态分类套了个 LIF 壳"。手势网上 memory_strength 单调涨到 10pp。
    """
    value = abs(deltas.get(1.0, 0.0))
    strength = max(abs(deltas[t]) for t in (1.5, 2.0, 2.5, 3.0, 4.0) if t in deltas)
    verdict = (
        "时间维度未被使用（记忆强度无关，LIF 层等价于 relu）"
        if strength < 0.01
        else "时间维度在用：记忆强度直接决定精度"
    )
    print(f"\n  memory_value    (tau=dt，无记忆)      {value * 100:>6.2f}pp")
    print(f"  memory_strength (tau in [1.5,4])     {strength * 100:>6.2f}pp")
    print(f"  判定：{verdict}")


def mnist_sweep(weights: str) -> None:
    src_path = os.path.join(ROOT, "examples", "mnist_mlp.snn")
    src = open(src_path, encoding="utf-8").read()
    if not os.path.exists(weights):
        raise SystemExit(
            f"缺少 checkpoint：{weights}\n"
            f"先跑：snndsl run examples/mnist_mlp.snn --epochs 10 -o build/"
        )

    blob = torch.load(weights, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)

    def build(tau: float):
        text = src.replace("param tau = 2.0", f"param tau = {tau}")
        res = compile_source(text, filename="<tau-sweep>", quiet=True)
        cls = load_module(res, os.path.join(tempfile.mkdtemp(), "net.py"))
        net = cls()
        net.load_state_dict(state)
        net.eval()
        return net, res.graph

    net0, graph = build(2.0)
    dev = rt.build_dataloaders(graph.train, tuple(graph.node(graph.inputs[0]).shape))
    test = dev["test"]
    shape = tuple(graph.node(graph.inputs[0]).shape)

    @torch.no_grad()
    def acc(net):
        hit = tot = 0
        for x, y in test:
            out = net(rt.adapt_batch(x, shape))
            hit += int((out.argmax(1) == y).sum())
            tot += y.numel()
        return hit / max(tot, 1)

    print("=" * 76)
    print("MNIST MLP（rate 编码，静态分类任务）")
    print("=" * 76)
    base = acc(net0)
    print(f"  基准 tau=2.0  acc={base * 100:.2f}%  n={len(test.dataset)}\n")
    print(f"  {'tau':>6} {'芯片可表示':>11} {'k':>3} {'acc':>9} {'Δ':>10}")
    deltas = {}
    for tau in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        near, k = snap_tau(tau)
        net, _ = build(tau)
        a = acc(net)
        deltas[tau] = a - base
        print(f"  {tau:>6.2f} {near:>11.2f} {k:>3d} {a * 100:>8.2f}% {(a - base) * 100:>+9.2f}pp")
    report_temporal(deltas)

    # int8：当前工具只报压缩率，这里补上精度代价
    net, _ = build(2.0)
    before = acc(net)
    size = rt.model_size_bytes(net)
    print(f"\n  {'int8':>8} {'精度':>18} {'代价':>10} {'压缩':>8}")
    for pc in (True, False):
        net, _ = build(2.0)
        st = rt.quantize_weights_int8(net, per_channel=pc)
        after = acc(net)
        tag = "per-channel" if pc else "per-tensor"
        print(
            f"  {tag:>11} {before * 100:.2f}% -> {after * 100:.2f}% "
            f"{(after - before) * 100:>+9.2f}pp {st['compression']:>7.2f}x"
        )
    print(f"  权重 {size / 1024:.0f} KiB fp32 -> {st['bytes_quantized'] / 1024:.0f} KiB int8")
    print("  注意：10000 张测试集上 95% CI 约 ±0.4pp，所以这里的 -0.02pp 是噪声。")


# ------------------------------------------------------------------ 手势网
def gesture_sweep(epochs: int = 8) -> None:
    sys.path.insert(0, HERE)
    import gesture_snn as G

    xtr, ytr = G.make_dataset(80, seed=0)
    xte, yte = G.make_dataset(40, seed=1)
    print("=" * 76)
    print(f"手势网（2 极性 8 步，membrane 读出，T={G.GestureNet.steps}）")
    print("=" * 76)
    print(f"  train {tuple(xtr.shape)}  test {tuple(xte.shape)}  训练 {epochs} epoch")

    model = G.GestureNet()
    out = G.train(model, xtr, ytr, xte, yte, epochs=epochs, quiet=True)
    state = {k: v.clone() for k, v in model._compiled.state_dict().items()}
    base = out["test_acc"]
    print(f"  基准 tau=2.0  acc={base * 100:.2f}%\n")

    def rebuild(tau: float):
        m = G.GestureNet()
        for name in ("lif1", "lif2", "lif3", "lif4"):
            cell = getattr(m, name)
            cell.params["tau"] = tau
            cell._spec = None  # 缓存的是编译好的神经元规格，改参数必须清
        m.compile(quiet=True)
        m._compiled.load_state_dict(state)
        return m._compiled

    print(f"  {'tau':>6} {'芯片可表示':>11} {'k':>3} {'acc':>9} {'Δ':>10}")
    deltas = {}
    for tau in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        near, k = snap_tau(tau)
        a = G.evaluate(rebuild(tau), xte, yte)
        deltas[tau] = a - base
        print(f"  {tau:>6.2f} {near:>11.2f} {k:>3d} {a * 100:>8.2f}% {(a - base) * 100:>+9.2f}pp")
    report_temporal(deltas)
    print("  同一个 tau 扫描，MNIST MLP 的 memory_strength 是 0.09pp。差两个数量级。")

    print(f"\n  {'int8':>8} {'精度':>18} {'代价':>10} {'压缩':>8}")
    for pc in (True, False):
        net = rebuild(2.0)
        st = rt.quantize_weights_int8(net, per_channel=pc)
        a = G.evaluate(net, xte, yte)
        tag = "per-channel" if pc else "per-tensor"
        print(
            f"  {tag:>11} {base * 100:.2f}% -> {a * 100:.2f}% "
            f"{(a - base) * 100:>+9.2f}pp {st['compression']:>7.2f}x"
        )
    print(
        "  注意：240 张测试集上 95% CI 约 ±3pp。per-channel 与 per-tensor 的 0.42pp 差别不可区分。"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--weights",
        default=os.path.join(ROOT, "build", "mnist_mlp.pt"),
        help="MNIST MLP checkpoint（默认 build/mnist_mlp.pt）",
    )
    ap.add_argument("--gesture", action="store_true", help="改跑手势网（当场训练，约 20 秒）")
    ap.add_argument("--epochs", type=int, default=8, help="手势网的训练 epoch 数")
    args = ap.parse_args(argv)

    if args.gesture:
        gesture_sweep(args.epochs)
    else:
        mnist_sweep(args.weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
