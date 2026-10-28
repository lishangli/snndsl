"""端到端证据：真实测试集上，训练实现与 NIR 参考实现是否给出相同预测。

    python3 examples/verify_deployment.py

这条脚本回答的是投资人/客户最关心的那个问题：**我把模型交给硬件之后，
它还是同一个模型吗？** 答案是逐样本比对预测，不是看文件能不能打开。

前置：先训练出 build_hw/mnist_hw.pt
    python3 -m snndsl run examples/mnist_hw.snn --out build_hw --epochs 4
"""
import json

import torch
from snndsl import compile_source, load_module
from snndsl.backends import nir_sim
from snndsl.backends.nir_export import build_nir_graph
from snndsl.backends.runtime import build_dataloaders

with open("examples/mnist_hw.snn", encoding="utf-8") as fh:
    res = compile_source(fh.read())
g = res.graph
Net = load_module(res, "build_hw/_verify_net.py")
net = Net()
blob = torch.load("build_hw/mnist_hw.pt", map_location="cpu", weights_only=False)
net.load_state_dict(blob.get("state_dict", blob))
net.eval()
w = {k: v.cpu().numpy() for k, v in net.state_dict().items()}

cfg = {"data": "mnist", "seed": 0}
loaders = build_dataloaders(cfg, tuple(g.node(g.inputs[0]).shape), batch_size=100)
test = loaders["test"] if isinstance(loaders, dict) else loaders[-1]

ng, _ = build_nir_graph(g, w)
T = g.temporal.steps
agree = total = correct_t = correct_n = 0
for xb, yb in test:
    with torch.no_grad():
        out_t = net(xb)
    sim = nir_sim.simulate(ng, xb, steps=T, dt=float(g.temporal.dt))
    out_n = sim["spikes"]["y_1"].mean(0)
    pt, pn = out_t.argmax(1), out_n.argmax(1)
    agree += (pt == pn).sum().item()
    correct_t += (pt == yb).sum().item()
    correct_n += (pn == yb).sum().item()
    total += yb.numel()
    if total >= 1000:
        break
print(json.dumps({"samples": total,
                  "pytorch_acc": round(correct_t/total,4),
                  "nir_acc": round(correct_n/total,4),
                  "prediction_agreement": round(agree/total,4)}, indent=1))
