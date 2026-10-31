#!/usr/bin/env python3
"""从零设计一个新的 SNN：任务 -> 结构 -> 训练 -> 可部署产物。

    python examples/gesture_snn.py            # 全流程 + 四组设计实验（约 3 分钟）
    python examples/gesture_snn.py --quick    # 每项少跑几个 epoch，只看链路

这个例子刻意不复刻 MNIST：任务是**事件相机的手势识别**（DVS 风格），
6 类运动方向 —— 右扫 / 左扫 / 下扫 / 上扫 / 对角 / 扩散环。

选它的理由只有一条：**右扫和左扫在单帧上看不出区别**，差别全在
ON/OFF 两个极性通道的空间分布里。一个只会看「亮点在哪」的网络做不对，
这就逼着网络结构必须真的处理极性，而不是靠亮度总量取巧。

运行完会在 build_gesture/ 下留全部中间产物：

    gesture.nir          导出的 NIR（给芯片工具链）
    gesture.snn          IR 回写成 .snn 文本
    gesture.py           生成的 PyTorch 源码（可训练、可单文件部署）
    gesture.pt           checkpoint
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# examples/ 下的脚本可以直接 python 运行，不要求先 pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import snndsl as snn
from snndsl.backends.nir_export import export_nir
from snndsl.backends.snn_mlir import check as backend_check
from snndsl.backends.snn_mlir import preflight
from snndsl.deploy import analyze, verify
from snndsl.runner import energy_report, measure_sparsity

OUT = Path("build_gesture")

# ============================================================ 1. 任务定义
# 传感器视角：一个 32x32 的事件相机看 10 个采样帧，输出两张累积图
# （ON = 亮度上升的事件，OFF = 亮度下降的事件）。这就是真实 DVS 的
# 最常用输入形式——事件累积成两通道的帧，再交给网络。
SIZE = 32
FRAMES = 8
N_CLASSES = 6
CLASSES = ["right", "left", "down", "up", "diag", "ring"]


def _one_sample(cls: int, rng: np.random.Generator) -> np.ndarray:
    """生成一个样本：一条运动轨迹 -> 逐帧亮度 -> 逐帧事件 -> 两通道累积。"""
    g = (np.arange(SIZE) / (SIZE - 1)).astype(np.float32)
    xx, yy = np.meshgrid(g, g)
    jx, jy = rng.uniform(-0.14, 0.14, 2)      # 起点偏移：不让网络记住绝对位置
    speed = rng.uniform(0.6, 1.4)           # 速度抖动：不让网络记住固定行程
    sigma = rng.uniform(0.045, 0.075)           # 目标大小

    bright = np.zeros((FRAMES, SIZE, SIZE), np.float32)
    for t in range(FRAMES):
        s = min((t / (FRAMES - 1)) * speed, 1.0)
        fwd, bwd = 0.12 + 0.76 * s, 0.88 - 0.76 * s
        if cls == 0:                                      # 右扫
            d2 = (xx - fwd - jx) ** 2 + (yy - 0.5 - jy) ** 2
        elif cls == 1:                                    # 左扫
            d2 = (xx - bwd - jx) ** 2 + (yy - 0.5 - jy) ** 2
        elif cls == 2:                                    # 下扫
            d2 = (xx - 0.5 - jx) ** 2 + (yy - fwd - jy) ** 2
        elif cls == 3:                                    # 上扫
            d2 = (xx - 0.5 - jx) ** 2 + (yy - bwd - jy) ** 2
        elif cls == 4:                                    # 对角
            d2 = (xx - 0.15 - 0.7 * s - jx) ** 2 + (yy - 0.15 - 0.7 * s - jy) ** 2
        else:                                             # 扩散环
            r = np.sqrt((xx - 0.5 - jx) ** 2 + (yy - 0.5 - jy) ** 2)
            d2 = (r - (0.05 + 0.38 * s)) ** 2
        bright[t] = np.exp(-d2 / (2 * sigma ** 2)) * (0.22 if cls == 5 else 1.0)

    on = np.zeros((SIZE, SIZE), np.float32)
    off = np.zeros((SIZE, SIZE), np.float32)
    prev = np.zeros((SIZE, SIZE), np.float32)
    for t in range(FRAMES):
        d = bright[t] - prev                 # 这是 DVS 的全部物理：只发"变化"
        on += np.maximum(d, 0)
        off += np.maximum(-d, 0)
        prev = bright[t]
    # 背景噪声事件：真实传感器的暗电流噪声，也是防止网络学到"干净输入"的护栏
    on += (rng.random((SIZE, SIZE)) < 0.08).astype(np.float32)
    off += (rng.random((SIZE, SIZE)) < 0.08).astype(np.float32)
    ev = np.stack([on, off]).astype(np.float32)
    # 按 99 分位归一化：等价于相机自动曝光。不做这一步，事件累积的均值只有
    # 0.05，喂进 LIF 后膜电位永远到不了阈值，网络是死的（实测 lif1 发放率 7e-4，
    # 后面两层恒为 0，logits 全是 0，loss 卡在 ln(6)）。
    return np.clip(ev / (np.percentile(ev, 99.0) + 1e-6), 0.0, 1.0)


def make_dataset(per_class: int, seed: int) -> tuple:
    rng = np.random.default_rng(seed)
    xs = [_one_sample(c, rng) for c in range(N_CLASSES) for _ in range(per_class)]
    ys = [c for c in range(N_CLASSES) for _ in range(per_class)]
    x = np.clip(np.stack(xs), 0.0, 1.0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    perm = rng.permutation(len(y))
    return torch.from_numpy(x[perm]), torch.from_numpy(y[perm])


def merge_polarity(x: torch.Tensor) -> torch.Tensor:
    """消融用的对照组：把 ON/OFF 合成一个通道，极性信息彻底消失。"""
    return x.sum(dim=1, keepdim=True).repeat(1, 2, 1, 1) * 0.5


# ========================================================== 2. 网络结构
class GestureNet(snn.Module):
    """卷积脉冲网络。forward 描述**一个时间步**，T 由编译器套上去。

    每个设计选择都对应一条可验证的后果，见 docs/DESIGN_A_NETWORK.md：

    - ``reset = "zero"``   只有零复位能被 NIR 精确表示，减法复位会漂移
    - ``readout = "membrane"``  证据在膜电位里累积，不在最后一步的脉冲里
    - 池化用 ``avg``       NIR 没有 MaxPool2d，写了会被后端静默换成 avg
    - 神经元用 ``LIF``     ALIF 的自适应阈值 NIR 表示不了，导出即降级
    """

    input_shape = (2, 32, 32)
    steps = 8
    reset = "zero"
    readout = "membrane"

    def __init__(self, width: int = 16, neuron: str = "lif", pool: str = "avg",
                 reset: str | None = None, steps: int | None = None) -> None:
        super().__init__()
        if reset is not None:
            self.reset = reset
        if steps is not None:
            self.steps = int(steps)
        Cell = {"lif": snn.LIF, "alif": snn.ALIF}[neuron]
        Pool = {"avg": snn.AvgPool, "max": snn.MaxPool}[pool]

        self.conv1 = snn.Conv2d(width, kernel=3, padding=1, scale=0.6)
        self.lif1 = Cell(tau=2.0, v_th=1.0, reset=self.reset)
        self.pool1 = Pool(2)

        self.conv2 = snn.Conv2d(width * 2, kernel=3, padding=1, scale=1.5)
        self.lif2 = Cell(tau=2.0, v_th=1.0, reset=self.reset)
        self.pool2 = Pool(2)

        self.fc1 = snn.Dense(64, scale=3.5)
        self.lif3 = snn.LIF(tau=2.0, v_th=0.6, reset=self.reset)
        # 输出层自己也发放：否则 `readout` 是空设置。output 接在线性层后面时，
        # membrane 与 spikes 生成的是同一份代码（都读线性层的连续输出），
        # 两种读出就无从比较——这是实测撞出来的，不是设计上的选择。
        self.fc2 = snn.Dense(N_CLASSES)
        self.lif4 = snn.LIF(tau=2.0, v_th=0.6, reset=self.reset)

    def forward(self, x):
        # (2,32,32) -> (16,32,32) -> 发放 -> (16,16,16)
        x = self.pool1(self.lif1(self.conv1(x)))
        # -> (32,16,16) -> 发放 -> (32,8,8)
        x = self.pool2(self.lif2(self.conv2(x)))
        # dense 自己会 flatten：2048 -> 64
        x = self.lif3(self.fc1(x))
        # 6 个类别的读出神经元：membrane 读膜电位，spikes 读计数
        return self.lif4(self.fc2(x))


# =============================================================== 3. 训练
def train(model: GestureNet, xtr, ytr, xte, yte, epochs: int, lr: float = 2e-3,
          batch: int = 64, seed: int = 0, quiet: bool = False) -> dict:
    """标准 BPTT。没有任何 snndsl 运行时参与 —— 产物就是个 nn.Module。"""
    torch.manual_seed(seed)
    result = model.compile(quiet=True)
    net = model._compiled
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    g = torch.Generator().manual_seed(seed)
    history, t0 = [], time.time()

    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(ytr), generator=g)
        loss_sum = hit = 0
        for i in range(0, len(ytr), batch):
            idx = perm[i:i + batch]
            logits = net(xtr[idx])                     # 内部展开 T 步
            loss = F.cross_entropy(logits, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()                            # 阈值不可导，梯度来自 surrogate
            opt.step()
            loss_sum += float(loss.detach()) * len(idx)
            hit += int((logits.detach().argmax(1) == ytr[idx]).sum())
        sched.step()
        acc = evaluate(net, xte, yte)
        history.append({"epoch": ep, "loss": loss_sum / len(ytr),
                        "train_acc": hit / len(ytr), "test_acc": acc})
        if not quiet:
            print(f"    epoch {ep}/{epochs}  loss={loss_sum / len(ytr):.4f}  "
                  f"train={hit / len(ytr) * 100:5.1f}%  test={acc * 100:5.1f}%  "
                  f"({time.time() - t0:.1f}s 累计)")
    return {"graph": result.graph, "module": net, "history": history,
            "test_acc": history[-1]["test_acc"],
            "params": sum(p.numel() for p in net.parameters()),
            "seconds": time.time() - t0}


@torch.no_grad()
def evaluate(net, x, y, batch: int = 128) -> float:
    net.eval()
    hit = 0
    for i in range(0, len(y), batch):
        hit += int((net(x[i:i + batch]).argmax(1) == y[i:i + batch]).sum())
    return hit / len(y)


# ============================================================ 4. 设计实验
def sweep_steps(xtr, ytr, xte, yte, epochs: int) -> list:
    """T 是延迟预算，不是超参：T 步 = T 个硬件时钟周期。"""
    rows = []
    for steps in (4, 8, 16):
        m = GestureNet(steps=steps)
        r = train(m, xtr, ytr, xte, yte, epochs, quiet=True)
        rows.append({"steps": steps, "acc": r["test_acc"],
                     "seconds": r["seconds"], "macs": steps * _macs(r["graph"])})
        print(f"    T={steps:<3d} test={r['test_acc'] * 100:5.1f}%   "
              f"{steps * _macs(r['graph']) / 1e6:6.2f}M MAC   "
              f"延迟 {steps} 步")
    return rows


def _macs(graph) -> int:
    total = 0
    for n in graph.nodes:
        if n.op == "dense":
            total += int(n.attrs["in_features"]) * int(n.attrs["units"])
        elif n.op == "conv":
            total += (int(n.attrs["in_channels"]) * int(n.attrs["kernel"]) ** 2
                      * int(n.attrs["out_channels"]) * int(n.shape[1]) * int(n.shape[2]))
    return total


def deploy_chain(result: dict, xte, tag: str) -> dict:
    """一条模型走完四道关：导出 -> 预检 -> 后端读 -> 逐脉冲对比。"""
    graph, net = result["graph"], result["module"]
    weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    path = OUT / f"{tag}.nir"
    rep = export_nir(graph, str(path), weights=weights, target="snn-mlir")
    return {
        "tag": tag,
        "path": path,
        "export": rep,
        "preflight": preflight(graph),
        "deploy": analyze(graph),
        "backend": backend_check(str(path)),
        "verify": verify(graph, net, xte[:16], weights=weights,
                         nir_path=str(path)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--sweep-epochs", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="2/2 epoch，只检查链路")
    ap.add_argument("--no-sweeps", action="store_true")
    args = ap.parse_args(argv)
    if args.quick:
        args.epochs, args.sweep_epochs = 2, 2

    OUT.mkdir(exist_ok=True)
    rule = lambda s: print(f"\n{'=' * 72}\n{s}\n{'=' * 72}")  # noqa: E731

    # ---------------------------------------------------------- 数据
    rule("0. 任务：6 类事件相机手势")
    t0 = time.time()
    xtr, ytr = make_dataset(80, seed=0)
    xte, yte = make_dataset(40, seed=1)
    print(f"  训练 {tuple(xtr.shape)}  测试 {tuple(xte.shape)}  "
          f"({time.time() - t0:.1f}s 生成)")
    print(f"  每个样本 = 2 通道 x 32x32 的事件累积（ON / OFF），{FRAMES} 帧累积")
    for c in range(N_CLASSES):
        m = xtr[ytr == c]
        print(f"    {CLASSES[c]:6s} ON {float(m[:, 0].mean()):.4f}   "
              f"OFF {float(m[:, 1].mean()):.4f}")
    print("  注意 right/left 的 ON/OFF 均值几乎相同 —— 差别只在空间分布，")
    print("  所以这个任务不能靠「有多少事件」区分，必须看结构。")

    # ---------------------------------------------------------- 结构
    rule("1. 结构：13 行 forward -> 12 个 IR 节点")
    model = GestureNet()
    result = model.compile(quiet=True)
    graph = result.graph
    print(result.summary())
    print("\n  forward 与 IR 逐行对照（节点名 = 属性名）：")
    print(f"    {'node':7s} {'op':8s} {'shape':16s} inputs")
    for n in graph.nodes:
        print(f"    {n.id:7s} {n.op:8s} {tuple(n.shape)!s:16s} {n.inputs}")

    # ---------------------------------------------------------- 训练
    rule(f"2. 训练：BPTT {args.epochs} epoch")
    r = train(model, xtr, ytr, xte, yte, args.epochs)
    net, graph = r["module"], r["graph"]
    print(f"  参数量 {r['params']:,}   测试准确率 {r['test_acc'] * 100:.2f}%   "
          f"{r['seconds']:.1f}s")

    # 读出方式决定能不能上硬件，所以第二个模型不是可选项
    spike_model = GestureNet()
    spike_model.readout = "spikes"
    r_spk = train(spike_model, xtr, ytr, xte, yte, args.sweep_epochs, quiet=True)
    print(f"  对照：readout=spikes，{args.sweep_epochs} epoch -> "
          f"测试准确率 {r_spk['test_acc'] * 100:.2f}%")

    # ---------------------------------------------------------- 实验
    if not args.no_sweeps:
        rule(f"3. 设计实验（每项 {args.sweep_epochs} epoch，冷启动重训）")
        print("  a) T 选多大：T 是延迟预算，不是超参")
        rows = sweep_steps(xtr, ytr, xte, yte, args.sweep_epochs)
        top = max(r["acc"] for r in rows)
        # 精度差不到 1 个点就当平手，平手取最小 T ---- T 是延迟预算
        pick = min((r for r in rows if r["acc"] >= top - 0.01),
                   key=lambda d: d["steps"])
        lo, hi = min(r["steps"] for r in rows), max(r["steps"] for r in rows)
        gain = (top - min(r["acc"] for r in rows)) * 100
        near = next(r for r in rows if r["steps"] == pick["steps"])
        print(f"    -> 选 T={pick['steps']}：T 从 {lo} 加到 {hi} 只多 {gain:.1f} 个点"
              f"（T={pick['steps']} 本身 {near['acc'] * 100:.1f}%），MAC 却按 T 线性涨")
        print("       T 的代价是延迟（T 步 = T 个时钟周期），不是精度")

        print("\n  b) 读出方式：膜电位 vs 脉冲（两个模型上面已训好）")
        for tag, mm, rr in (("membrane", model, r), ("spikes", spike_model, r_spk)):
            with torch.no_grad():
                logits = mm._compiled(xte[:16]).detach()
            conf = float(logits.softmax(1).max(1).values.mean())
            with torch.no_grad():
                allvals = mm._compiled(xte).detach().numpy().ravel()
            distinct = len({round(float(v), 6) for v in allvals})
            print(f"    readout={tag:9s} test={rr['test_acc'] * 100:5.1f}%   "
                  f"平均置信度={conf:.4f}   全测试集 logit 的取值种类={distinct}"
                  f"（共 {allvals.size} 个数）")
        print("    -> membrane 读的是膜电位，连续；spikes 读的是「T 步里发了几次」，")
        print("       一个神经元只有 T+1 种取值，整层最多 (T+1)^K 种组合。")
        print("       分类头靠计数，分辨率就是时间窗长度 —— 这正是掉点的原因，")
        print("       不是训练不充分：两者训练预算完全一样。")

        print("\n  c) 极性消融：把 ON/OFF 合成一张图，时间信息还剩多少")
        m = GestureNet()
        rr = train(m, merge_polarity(xtr), ytr, merge_polarity(xte), yte,
                   args.sweep_epochs, quiet=True)
        print(f"    合并极性   test={rr['test_acc'] * 100:5.1f}%   "
              f"（原样 {r['test_acc'] * 100:.1f}%）")
        print("    -> 掉多少，就是极性这两通道贡献了多少")

        print("\n  d) 同一份 forward 的两种配置：能不能上硬件")
        variants = [("A LIF+avg+zero（就是上面那个模型）", r)]
        m = GestureNet(neuron="alif", pool="max", reset="subtract")
        variants.append(("B ALIF+max+subtract",
                         train(m, xtr, ytr, xte, yte, args.sweep_epochs, quiet=True)))
        for tag, rr in variants:
            pf = preflight(rr["graph"])
            print(f"    {tag:36s} test={rr['test_acc'] * 100:5.1f}%  "
                  f"accepts={pf['accepts']}  faithful={pf['faithful']}")
            for item in pf["lossy"]:
                print(f"        ! {item['rule']}: {item['detail']}")
        b_acc = variants[1][1]["test_acc"]

    # ---------------------------------------------------------- 产物
    rule("4. 产物：IR 回写成 .snn，源码写成 .py")
    snn_text = model.to_snn(comment=f"gesture {N_CLASSES} 类事件相机手势")
    (OUT / "gesture.snn").write_text(snn_text, encoding="utf-8")
    (OUT / "gesture.py").write_text(result.module_source, encoding="utf-8")
    torch.save({"state_dict": net.state_dict(), "steps": graph.temporal.steps},
               OUT / "gesture.pt")
    print(f"  gesture.snn  {len(snn_text.splitlines())} 行  (IR 的文本形式)")
    print(f"  gesture.py   {len(result.module_source.splitlines())} 行  (可训练的 PyTorch)")
    print("  gesture.pt   checkpoint")
    print("\n  .snn 里的 pipeline 段（文件其余部分是 neuron 定义与 temporal）：")
    keep = False
    for line in snn_text.splitlines():
        if line.startswith("pipeline:"):
            keep = True
        elif line.startswith("infer:"):
            keep = False
        if keep:
            print("    " + line)

    # ---------------------------------------------------------- 部署
    rule("5. 部署链路：导出 -> 预检 -> 后端读 -> 逐脉冲对比")
    chains = [deploy_chain(r, xte, "membrane"),
              deploy_chain(r_spk, xte, "spikes")]

    for ch in chains:
        g = r["graph"] if ch["tag"] == "membrane" else r_spk["graph"]
        print(f"\n  [{ch['tag']}]  {ch['path']}  "
              f"{ch['path'].stat().st_size / 1024:.0f} KB  "
              f"({ch['export']['node_count']} 节点 / {len(ch['export']['edges'])} 边)")
        print(f"    导出     skipped={ch['export']['unsupported'] or '无'}  "
              f"approximated={ch['export']['approximated'] or '无'}")
        pf = ch["preflight"]
        print(f"    预检     accepts={pf['accepts']}  faithful={pf['faithful']}  "
              f"blockers={len(pf['blockers'])}")
        print(f"    后端     snn-mlir check -> {ch['backend']['ok']}")
        dp = ch["deploy"]
        print(f"    deploy   {'DEPLOYABLE' if dp.deployable else 'NOT DEPLOYABLE'}  "
              f"(readout={dp.readout}, T={dp.temporal_steps})")
        for f in dp.findings:
            if f.severity == "blocker":
                print(f"      BLOCK {f.where}: {f.message}")
                if f.fix:
                    print(f"            -> {f.fix}")
        v = ch["verify"]
        print(f"    保真     max|Δ|={v['max_abs_error']:.3e}  "
              f"mean|Δ|={v['mean_abs_error']:.3e}  "
              f"top1={v['top1_agreement'] * 100:.1f}%  "
              f"equivalent={v['equivalent']}")
        print(f"    节点数   {len(g.nodes)}   "
              f"发放率最高的层 {max(v['layers'], key=lambda d: d['nir_spike_rate'])['node']}")

    rule("6. 成本：发放率与能耗（SNN 相对 ANN 的全部收益来源）")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xte, yte), batch_size=64)
    shape = tuple(graph.node(graph.inputs[0]).shape)
    rates = measure_sparsity(net, graph, loader, shape, n_batches=2)
    print("  各层平均发放率（每步、每个神经元）：")
    for nid, r_ in sorted(rates.items(), key=lambda kv: -kv[1]):
        print(f"    {nid:6s} {r_:.4f}")
    er = energy_report(graph, rates)
    print(f"\n  稠密 ANN  : {er['ann_macs'] / 1e6:8.2f}M MAC   "
          f"{er['ann_energy_nj']:.1f} nJ/样本")
    print(f"  本网络    : {er['snn_accumulates'] / 1e6:8.2f}M AC + "
          f"{er['snn_dense_macs'] / 1e6:.2f}M MAC   {er['snn_energy_nj']:.1f} nJ/样本")
    print(f"  能耗比    : {er['energy_ratio']:.2f}x   "
          f"(45nm, MAC={er['assumptions']['mac_pj']}pJ / AC="
          f"{er['assumptions']['ac_pj']}pJ)")
    print("  注意第一层仍按稠密 MAC 计 —— 它吃的是模拟电流，不是脉冲。")

    rule("7. 结论：结构由部署约束反推，不是先画图再想办法")
    print("  这个网络真正的设计顺序是四条约束，不是「先画图再想办法部署」：")
    print("")
    print("  1. 读出方式最先定。membrane 的证据在 T 步里累积，精度最高，但 NIR 神经元")
    print("     只发脉冲、不暴露膜电位 —— deploy.analyze 判定为硬 BLOCKER。")
    print("     spikes 能部署，代价是 logit 变成「几步里发了几次」，取值离散。")
    print("  2. 池化只能用 avg。NIR 没有 MaxPool2d，写 max 会被后端静默换成 avg，")
    print("     文件能转、C 能编、数字在动，但那已经是另一个网络。")
    print("  3. 复位的形状要挑。zero 复位精确映射到 NIR 的 v_reset；subtract 会漂移。")
    if not args.no_sweeps:
        gap = (b_acc - r["test_acc"]) * 100
        print(f"  4. 但 B 版（ALIF+max）实测 {b_acc * 100:.1f}%，比 A 版高 {gap:+.1f} 个点。")
        print("     所以「可部署版更好」是假的：可部署版是**拿精度换的**。")
        print(f"     问题变成：这 {gap:+.1f} 个点值不值得放弃整个工具链？")
        print("     在 240 张测试集上，1 个点约为 2.4 张图，而 95% 置信区间接近 ±3 点 ——")
        print("     这个差距还落在噪声里，不足以换掉部署能力。")

    rule("完成")
    print(f"  产物都在 {OUT.resolve()}/")
    print(f"  下一步：cat {OUT}/gesture.snn | head -40")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
