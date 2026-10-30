#!/usr/bin/env python3
"""一个 .snn 从编译到部署的全过程，全部用 Python 写，不用命令行。

    python examples/train_deploy.py                     # 默认 examples/mnist_hw.snn
    python examples/train_deploy.py path/to/yours.snn --epochs 3

六个阶段，每个阶段都是可以直接抄进你自己项目的几行代码：

    1. compile   源码 -> IR -> 生成的 PyTorch 源码文件
    2. train     自己写 BPTT（不调本仓库的训练循环），证明产物就是个 nn.Module
    3. export    IR + 权重 -> .nir，然后把文件从磁盘读回来看里面到底是什么
    4. verify    同一批输入，PyTorch 一遍、NIR 解释器一遍，逐位置比
    5. measure   稀疏度 / 能耗 / 量化误差
    6. infer     脱离编译器，只用生成的 .py + checkpoint

跑完 build_code/ 下就是全部中间产物，可以逐个打开看。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from snndsl import compile_source, load_module
from snndsl.backends import runtime as rt
from snndsl.backends.nir_export import export_nir
from snndsl.deploy import analyze, format_report, verify
from snndsl.runner import energy_report


# ---------------------------------------------------------------- 1. 编译
def stage_compile(src_path: Path, out: Path) -> tuple:
    """源码 -> IR -> 中间产物。这是整个工具的核心，一共三行。"""
    source = src_path.read_text(encoding="utf-8")

    result = compile_source(source, filename=str(src_path))   # 前端 + pass 流水线
    graph = result.graph                                      # 优化后的图，普通 Python 对象

    print(result.summary())          # 编译器眼里这个网络长什么样

    # IR 就是数据，可以直接遍历：节点 id、算子、形状、上游
    print("\nIR 节点（pass 之后的最终形态）")
    for n in graph.nodes:
        print(f"  {n.id:6s} {n.op:7s} shape={tuple(n.shape)!s:14s} <- {n.inputs}")

    # 三个产物落盘：生成的源码、静态 IR(JSON)、以及上面的 .mlir
    out.mkdir(parents=True, exist_ok=True)
    name = graph.name
    (out / f"{name}.py").write_text(result.module_source, encoding="utf-8")
    (out / f"{name}.ir.json").write_text(
        json.dumps(result.report_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n产物: {out}/{name}.py  ({len(result.module_source.splitlines())} 行，可读 PyTorch)")
    print(f"      {out}/{name}.ir.json")
    return result, graph


# ---------------------------------------------------------------- 2. 训练
def stage_train(result, graph, out: Path, epochs: int, batch: int, lr: float) -> torch.nn.Module:
    """自己写训练循环。

    生成的 .py 是一个普通 nn.Module：forward 里已经包含 T 步时间展开和
    代理梯度，所以训练就是一个标准 BPTT —— 不需要任何 snndsl 的运行时。
    """
    Net = load_module(result, str(out / f"{graph.name}.py"))   # import 那个文件
    net = Net()

    shape = tuple(graph.node(graph.inputs[0]).shape)
    loaders = rt.build_dataloaders(graph.train, shape, batch_size=batch)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    for ep in range(1, epochs + 1):
        net.train()
        t0, seen, correct, total_loss = time.time(), 0, 0, 0.0
        for x, y in loaders["train"]:
            x = rt.adapt_batch(x, shape)          # (B,1,28,28) -> IR 声明的形状
            logits = net(x)                       # <- 这里跑了 T=8 步，反向走的是代理梯度
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()                       # 阈值比较是硬的，梯度来自 surrogate
            opt.step()

            seen += len(y)
            correct += int((logits.argmax(1) == y).sum())
            total_loss += float(loss.detach()) * len(y)
        sched.step()

        net.eval()
        hit = n = 0
        with torch.no_grad():
            for x, y in loaders["test"]:
                x = rt.adapt_batch(x, shape)
                hit += int((net(x).argmax(1) == y).sum())
                n += len(y)
        print(f"  epoch {ep}/{epochs}  loss={total_loss / seen:.4f}  "
              f"train_acc={correct / seen * 100:.2f}%  test_acc={hit / n * 100:.2f}%  "
              f"({time.time() - t0:.1f}s)")

    ckpt = out / f"{graph.name}.pt"
    torch.save({"state_dict": net.state_dict(), "program": graph.name,
                "steps": graph.temporal.steps}, ckpt)
    print(f"  checkpoint: {ckpt}")
    return net


# ---------------------------------------------------------------- 3. 导出
def stage_export(graph, net: torch.nn.Module, out: Path) -> Path:
    """IR + 训练好的权重 -> .nir，然后读回来检查里面到底有什么。

    没有任何 C++ / 汇编 / 二进制 kernel：这一层的产物是一个 HDF5 文件，
    里面是「节点 + 边 + 每层的参数数组」，下游的芯片工具链接手。
    """
    import nir

    weights = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    path = out / f"{graph.name}.nir"
    report = export_nir(graph, str(path), weights=weights)

    print(f"  {path}  ({path.stat().st_size / 1024:.0f} KB)")
    for nid, kind in report["nodes"].items():
        print(f"    {nid:6s} -> {kind}")

    g = nir.read(str(path))            # 从磁盘读回，不是内存里那份
    print(f"  节点 {len(g.nodes)} / 边 {len(g.edges)} / metadata {dict(g.metadata)}")
    return path


# ---------------------------------------------------------------- 4. 验证
def stage_verify(graph, net: torch.nn.Module, nir_path: Path, out: Path) -> None:
    """同一批真实输入：PyTorch 跑一遍、NIR 解释器跑一遍、逐位置比。"""
    shape = tuple(graph.node(graph.inputs[0]).shape)
    loaders = rt.build_dataloaders(graph.train, shape, batch_size=64)
    x, _ = next(iter(loaders["test"]))
    sample = rt.adapt_batch(x, shape)

    rep = verify(graph, net, sample, nir_path=str(nir_path))
    print(f"  max |error| : {rep['max_abs_error']:.3e}")
    print(f"  mean|error| : {rep['mean_abs_error']:.3e}")
    print(f"  top-1 一致  : {rep['top1_agreement']:.3f}")
    print(f"  等价        : {rep['equivalent']}")
    for layer in rep["layers"]:
        print(f"    {layer['node']:6s} 发放率 {layer['nir_spike_rate']:.4f}")


# ---------------------------------------------------------------- 5. 测量
def stage_measure(graph, net: torch.nn.Module, out: Path) -> None:
    """稀疏度、事件驱动能耗、int8 量化的代价 —— 都是跑出来的数，不是估的。"""
    shape = tuple(graph.node(graph.inputs[0]).shape)
    loaders = rt.build_dataloaders(graph.train, shape, batch_size=256)
    x, _ = next(iter(loaders["test"]))
    x = rt.adapt_batch(x, shape)

    net.collect = True
    with torch.no_grad():
        net(x)
    rates = {nid: acc / graph.temporal.steps for nid, acc in net.spike_trace.items()}
    net.collect = False

    print("  每层发放率：" + "  ".join(
        f"{graph.node(k).name}={v * 100:.1f}%" for k, v in rates.items()))

    # 事件驱动成本模型：只有输入真的是稀疏脉冲的层才按 AC 计费，
    # 首层吃 rate 编码的模拟电流，必须按 MAC 计费（所以本网络只有 1.0x）
    er = energy_report(graph, rates)
    print(f"  ANN {er['ann_macs']:,} MAC -> {er['ann_energy_nj']:.1f} nJ")
    print(f"  SNN {int(er['snn_accumulates']):,} AC + {er['snn_dense_macs']:,} MAC "
          f"-> {er['snn_energy_nj']:.1f} nJ  ({er['energy_ratio']:.2f}x)")
    for ly in er["layers"]:
        tag = "event" if ly["event_driven"] else "dense"
        print(f"    {ly['name']:4s} {ly['op']:5s} {tag:5s} "
              f"in_rate={ly['input_rate']:.3f}  {ly['synapses']:,} synapses")

    q = rt.quantize_weights_int8(net)          # 原地 fake-quant，weight-only
    print(f"  int8: {q['bytes_fp32'] / 1024:.0f} KB -> {q['bytes_quantized'] / 1024:.0f} KB "
          f"({q['compression']:.2f}x)")

    net.load_state_dict(torch.load(out / f"{graph.name}.pt")["state_dict"])  # 还原 fp32
    net.eval()


# ---------------------------------------------------------------- 6. 推理
def stage_infer(graph, net: torch.nn.Module, out: Path) -> None:
    """部署侧：不再需要编译器，只要 .py + checkpoint。

    DeployedModel 只是把「编译 + 载权重 + 统计」包了一层的便利类；
    下面注释里那三行是它等价的手写版本。
    """
    from snndsl.infer import DeployedModel

    shape = tuple(graph.node(graph.inputs[0]).shape)
    loaders = rt.build_dataloaders(graph.train, shape, batch_size=4)
    x, y = next(iter(loaders["test"]))
    x = rt.adapt_batch(x, shape)

    deployed = DeployedModel.load(str(Path(graph.filename)), weights=str(out / f"{graph.name}.pt"),
                                  out_dir=str(out))
    # 手写版本：
    #   Net = load_module(result, "build_code/mnist_hw.py"); net = Net()
    #   net.load_state_dict(torch.load("build_code/mnist_hw.pt")["state_dict"])
    #   logits = net(x); probs = logits.softmax(1)
    r = deployed.predict(x, topk=3)

    hit = sum(int(p == t) for p, t in zip(r["pred"], y.tolist(), strict=True))
    print(f"  {len(y)} 个样本，预测正确 {hit}（{hit / len(y) * 100:.2f}%）")
    for i in range(min(6, len(y))):
        print(f"    真实 {int(y[i])}  预测 {r['pred'][i]}  置信度 {r['confidence'][i]:.4f}")
    print(f"  批量 top-3（只给第 0 个样本）: "
          f"{[(d['class'], round(d['score'], 4)) for d in r['topk']]}")
    print(f"  每层脉冲 {r['spikes_per_layer']}")
    print(f"  单样本延迟 {r['latency_ms_per_sample']:.3f} ms")
    # 注意置信度恒为 0.232/0.0853/0.0853：spike readout 下输出层一旦过阈值就
    # 整个 T 全发放，logits 是 one-hot，softmax 必然给出这三个数。argmax 是对的，
    # 置信度没有信息量。要可用的置信度用 readout=membrane（但那个不能无损部署）。


def main() -> int:
    ap = argparse.ArgumentParser(description="compile -> train -> export -> verify -> infer")
    ap.add_argument("file", nargs="?", default="examples/mnist_hw.snn")
    ap.add_argument("-o", "--out", default="build_code")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    src, out = Path(args.file), Path(args.out)

    print("=" * 70, "\n1. 编译\n", "=" * 70, sep="")
    result, graph = stage_compile(src, out)

    print("\n静态部署检查（不训练、不要数据，走着 IR 就能给结论）")
    print(format_report(analyze(graph)))

    print("\n" + "=" * 70 + "\n2. 训练\n" + "=" * 70)
    net = stage_train(result, graph, out, args.epochs, args.batch, args.lr)

    print("\n" + "=" * 70 + "\n3. 导出 NIR\n" + "=" * 70)
    nir_path = stage_export(graph, net, out)

    print("\n" + "=" * 70 + "\n4. 验证导出前后是同一个网络\n" + "=" * 70)
    stage_verify(graph, net, nir_path, out)

    print("\n" + "=" * 70 + "\n5. 测量\n" + "=" * 70)
    stage_measure(graph, net, out)

    print("\n" + "=" * 70 + "\n6. 推理\n" + "=" * 70)
    stage_infer(graph, net, out)

    print(f"\n全部产物在 {out}/，可以逐个打开：")
    for p in sorted(out.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size / 1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
