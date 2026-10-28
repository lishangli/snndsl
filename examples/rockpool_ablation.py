"""用 SynSense 自己的神经元，量出 NIR 规范里三处"没写死"的约定各值多少。

    SNNDSL_VENDOR_PYTHON=/root/.venvs/vendor/bin/python \
        python3 examples/rockpool_ablation.py

为什么需要这个脚本：`deploy --targets` 里的 `nirtorch + snnTorch` 那条给的是
`max|Δ|=0.0`，很漂亮，但它不能回答"我换个厂商还好使吗"。rockpool 的 NIR 前端
读不了 nir 1.0（还在找 `tau_mem`），可它的**神经元模块**是好的——于是绕开前端，
用 rockpool 的 `LinearTorch` + `LIFNeuronTorch` 重建同一张图，喂同样的电流。

这里做的是对照实验，不是"再跑一遍看结果对不对"：

  对照 A：rockpool 的神经元  vs  我们的训练实现（两者都是 subtract reset）
          → 只在"tau 怎么变成衰减因子"这一件事上有差别，其余全部对齐。
            用来证明"换算方式"能造成多大差异。

  对照 B：rockpool 的神经元  vs  我们的 NIR 导出文件（zero reset）
          → 除 tau 外还差一个 reset 形式，用来证明"标准表达不了 subtract"值多少。

输入强度分四档：低档不饱和（v 从不超过 2θ），高档就是训练网络真实的电流量级。
饱和档里 rockpool 的 `floor(v/θ)` 会一步发多枚脉冲，而我们的 LIF 只发一枚——
这本身是第三个未定义点，脚本会把它算成 multi-spike 比例单独报出来，
不混进 tau 的数字里。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCALES = (0.15, 0.2, 0.5, 1.0)

# 在厂商解释器里跑的那一段。它拿到的是权重、输入、参考脉冲串，返回逐脉冲差异。
_VENDOR = r"""
import json, sys
import numpy as np, torch
from rockpool.nn.modules import LinearTorch
from rockpool.nn.modules.torch.lif_neuron_torch import LIFNeuronTorch

p = json.load(sys.stdin)
first = lambda o: o[0] if isinstance(o, tuple) else o
dt, tau, vth, T = p["dt"], p["tau"], p["vth"], p["steps"]
alpha = 1.0 - dt / tau
tau_direct = tau
tau_matched = -dt / np.log(alpha)


def run(tau_mem, x):
    mods = []
    for layer in p["layers"]:
        w = torch.as_tensor(np.array(layer["w"], dtype="float32"))
        b = torch.as_tensor(np.array(layer["b"], dtype="float32"))
        lin = LinearTorch(shape=(w.shape[1], w.shape[0]), weight=w.T, has_bias=True)
        with torch.no_grad():
            lin.bias.copy_(b)
        mods.append((lin, LIFNeuronTorch(shape=(w.shape[0],), tau_mem=tau_mem,
                                         threshold=vth, dt=dt, bias=0.0)))
    cur = torch.as_tensor(x).unsqueeze(1).expand(-1, T, -1).contiguous()
    outs = []
    with torch.no_grad():
        for lin, neu in mods:
            cur = first(neu(first(lin(cur))))
            outs.append(cur.numpy().transpose(1, 0, 2))     # (T,B,N)
    return outs


out = {"tau_direct": tau_direct, "tau_matched": float(tau_matched),
       "alpha": alpha, "rows": []}
for case in p["cases"]:
    x = np.load(case["inputs"]).reshape(int(p["batch"]), -1)
    ref_out = np.load(case["trains"]).astype("float32")
    ref_hid = np.load(case["trains_hidden"]).astype("float32")
    row = {"scale": case["scale"]}
    for name, tm in (("direct", tau_direct), ("matched", tau_matched)):
        trains = run(tm, x)
        got_out = trains[-1]
        got_hid = trains[0]
        row[name] = float((got_out != ref_out.reshape(got_out.shape)).mean())
        row[name + "_hidden"] = float((got_hid != ref_hid.reshape(got_hid.shape)).mean())
        row["multi_spike"] = float((got_out > 1.0).mean())
    out["rows"].append(row)
print(json.dumps(out))
"""


def _capture_torch_spikes(res, ckpt, xb, steps, module, load_module):
    """抓训练实现的逐时间步脉冲。

    生成的网络是一个大 forward，没有可挂钩的子模块，所以替换掉软阶跃函数本身：
    每次被调用就是"一层神经元在这一步算出脉冲"，按调用顺序交替取即可。
    必须在 ``load_module`` **之前** 打补丁——生成代码在 import 时就把这个名字
    绑进了自己的全局命名空间，事后再改 runtime 里的那份已经无效。
    """
    calls = []
    orig = module.surrogate_fast_sigmoid

    def patched(x, alpha=1.0):
        out = orig(x, alpha)
        calls.append(out.detach().clone())
        return out

    module.surrogate_fast_sigmoid = patched
    try:
        net = load_module(res, os.path.join(tempfile.mkdtemp(), "net.py"))()
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        with torch.no_grad():
            net(xb)
    finally:
        module.surrogate_fast_sigmoid = orig
    if len(calls) != 2 * steps:
        raise RuntimeError(f"expected {2 * steps} neuron calls, saw {len(calls)}")
    return (torch.stack(calls[0::2]).numpy(),      # 隐藏层：(T,B,N)
            torch.stack(calls[1::2]).numpy())      # 输出层：(T,B,N)


def main() -> int:
    vendor = os.environ.get("SNNDSL_VENDOR_PYTHON")
    if not vendor or not os.path.exists(vendor):
        print("需要 SNNDSL_VENDOR_PYTHON 指向装了 rockpool 的解释器", file=sys.stderr)
        return 2

    import snndsl.backends.runtime as rt
    from snndsl import compile_source, load_module
    from snndsl.backends import nir_sim
    from snndsl.backends.nir_export import build_nir_graph

    res = compile_source(open("examples/mnist_mlp.snn").read())
    graph = res.graph

    loaders = rt.build_dataloaders({"data": "mnist", "seed": 0},
                                   tuple(graph.node(graph.inputs[0]).shape),
                                   batch_size=64, root=None)
    test = loaders["test"] if isinstance(loaders, dict) else loaders[-1]
    xb, _ = next(iter(test))
    steps = int(graph.temporal.steps)
    tau = float(graph.neurons["LIF"].params["tau"])

    ckpt = "build/mnist_mlp.pt"
    net = load_module(res, os.path.join(tempfile.mkdtemp(), "net.py"))()
    net.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"])
    weights = {k: v.cpu().numpy() for k, v in net.state_dict().items()}

    work = tempfile.mkdtemp()
    cases = []
    bundle_layers = [{"w": weights["w_h1"].tolist(), "b": weights["b_h1"].tolist()},
                     {"w": weights["w_y"].tolist(), "b": weights["b_y"].tolist()}]

    nir_graph, _ = build_nir_graph(graph, weights)

    for scale in SCALES:
        x_path = os.path.join(work, f"x_{scale}.npy")
        xb_s = xb * scale
        np.save(x_path, xb_s.numpy())
        # 两套参考都必须用**同一档电流**跑，否则比的是两个不同的输入
        torch_hidden, torch_trains = _capture_torch_spikes(res, ckpt, xb_s, steps, rt, load_module)
        sim = nir_sim.simulate(nir_graph, xb_s, steps=steps,
                               dt=float(graph.temporal.dt))
        nir_trains = sim["spikes"]["y_1"].numpy()
        nir_hidden = sim["spikes"]["h1_1"].numpy()
        for tag, trains, hidden in (("torch", torch_trains, torch_hidden),
                                    ("nir", nir_trains, nir_hidden)):
            p_out = os.path.join(work, f"ref_{tag}_{scale}.npy")
            p_hid = os.path.join(work, f"refh_{tag}_{scale}.npy")
            np.save(p_out, trains)
            np.save(p_hid, hidden)
            cases.append({"scale": scale, "inputs": x_path, "trains": p_out,
                          "trains_hidden": p_hid, "tag": tag})

    payload = {"layers": bundle_layers, "tau": tau, "vth": 1.0,
               "dt": float(graph.temporal.dt), "steps": steps,
               "batch": int(xb.shape[0]), "cases": cases}
    proc = subprocess.run([vendor, "-c", _VENDOR], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        return 1
    data = json.loads(proc.stdout.strip().splitlines()[-1])

    rows = {(r["scale"], c["tag"]): r for r, c in zip(data["rows"], cases)}
    print(f"tau={tau}  dt={graph.temporal.dt}  "
          f"直传 tau_mem={data['tau_direct']} (alpha={np.exp(-graph.temporal.dt / tau):.4f})  "
          f"换算后 tau_mem={data['tau_matched']:.4f} (alpha={data['alpha']:.4f})")
    print()
    print("| 输入强度 | 对照 | 层 | tau 直传 | 按 exp 换算 | multi-spike |")
    print("|---|---|---|---|---|---|")
    for scale in SCALES:
        for tag, note in (("torch", "vs 训练实现"), ("nir", "vs 我们的 NIR 文件")):
            r = rows[(scale, tag)]
            for layer, key, lbl in (("hidden", "_hidden", "隐藏层"), ("out", "", "输出层")):
                print(f"| {scale} | {note} | {lbl} | {100 * r['direct' + key]:.2f}% | "
                      f"{100 * r['matched' + key]:.2f}% | {100 * r['multi_spike']:.2f}% |")
    print()
    print("读法：'vs 训练实现' 两列只差 tau 一件事（reset 与脉冲规则都对齐），"
          "所以差值就是 tau 约定本身的影响；")
    print("      'vs 我们的 NIR 文件' 还多一个 zero/subtract 之差，"
          "那是标准表达能力的缺口，不是换算问题。")

    out = os.path.join("docs", "rockpool_ablation.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"tau": tau, "vendor": data["tau_matched"], "rows": [
            {"scale": s, "tag": t, **rows[(s, t)]} for s, t in rows]}, fh, indent=2)
    print(f"\n数字已写入 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
