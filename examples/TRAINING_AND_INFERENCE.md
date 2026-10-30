# 训练与推理示例（全部为本仓库真实运行输出）

三个可跑通的例子，从 25 行 DSL 到卷积网。所有数字都来自 32 核 CPU（无 GPU）上的实际执行，命令可直接复现。

| 示例 | 规模 | 用途 | 实测 |
|---|---|---|---|
| `examples/quickstart.snn` | 101,770 参数 / T=6 | 最小可读示例 | 2 epoch → 93.83% |
| `examples/mnist_mlp.snn` | 203,530 参数 / T=8 | spike readout 参考实现 | 10 epoch → 96.52% |
| `examples/mnist_membrane.snn` | 203,530 参数 / T=8 | membrane readout（推荐） | 4 epoch → **97.21%** |

---

## 1. 一个完整的 DSL 程序

`examples/quickstart.snn` —— 去掉注释只有 25 行有效代码：

```snn
program quickstart

temporal:
    steps = 6

# 浮点型神经元：v 是膜电位，i 是上一层注入的突触电流，s 是输出脉冲
neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v - s * v_th
    surrogate = fast_sigmoid(alpha=4.0)

pipeline:
    input  x  : 1, 28, 28
    encode x  -> rate with gain=1.0
    dense  h  : 128 <- x  with scale=2.0
    neuron h  -> LIF
    dense  y  : 10  <- h  with scale=2.0
    neuron y  -> LIF
    output out <- y

train:
    data = mnist
    epochs = 2
    batch = 128
    lr = 0.001
    loss = ce
```

四段结构：

- `temporal` 定全局时间语义（步数、dt、reset 方式），后端据此决定展开还是循环。
- `neuron` 是**微分方程的源码**，不是选一个预定义模型。`v > v_th` 这行写的是硬阈值比较，编译器自动把它降级成 `surrogate_fast_sigmoid(v - v_th, 4.0)`——前向硬 0/1，反向有梯度。加第二、第三个 `state` 就得到 ALIF（见 `examples/adaptive_lif.snn`）。
- `pipeline` 里名字可以重绑定：`dense h` 之后 `neuron h` 覆盖同名，编译器内部转 SSA，IR 保持干净 DAG。
- `train` / `infer` / `budget` 是声明式配置，后端直接读，不写在代码里。

---

## 2. 训练

```bash
snndsl run examples/quickstart.snn --out build_quick
```

真实输出：

```
program   : quickstart   (quickstart.snn)
time      : T=6  dt=1.0  schedule=looped
graph     : 7 nodes, 101,770 params, 609,792 MAC (T x step)
input     : [1, 28, 28]
output    : [[10]]
passes    :
    dce          no change
    fuse         no change
  * memplan      activation memory 7376 -> 6272 B (-15.0%) using 2 slot(s)
  * schedule     T=6 > 4 -> looped (keeps code size at 4 layers instead of 24)
artifacts : quickstart.py, quickstart.ir.json, quickstart.nir
compile   : 1.6 ms

training  : mnist  epochs=2  batch=128  lr=0.001
  epoch 1/2  loss=1.5782  train_acc=86.56%  test_acc=92.69%  (91.2s)
  epoch 2/2  loss=1.5128  train_acc=93.19%  test_acc=93.83%  (168.6s)
checkpoint: build_quick/quickstart.pt

--- sparsity (avg firing rate over T) ---
  h             34.7%
  y             11.4%
  mean          23.1%

--- event-driven energy (per sample) ---
  ANN dense :      609,792 MAC  ->  2805.04 nJ
  SNN event :        2,664 AC +    602,112 MAC ->  2772.11 nJ
  ratio     : 1.01x lower energy than the dense ANN
  spikes/sample: 273
```

三个值得注意的点：

**编译只要 1.6 ms**，比一次前向还快。这是静态图 + 无 autograd 追踪的收益。

**`memplan` 是真实生效的**：7376 → 6272 字节，靠活跃区间着色复用 buffer。`schedule` 选了 looped（T=6 > 4），代码量从 24 层降到 4 层。

**能耗只有 1.01×**，这是结构决定的负数结论，不是实现 bug。首层 `808→128` 吃了 98% 的 MAC，而它的输入是 rate 编码的**模拟电流**，必须按 MAC 计费。第 4 节有敏感性分析。

训练循环是怎么来的：DSL 的 `train:` 块被解析成 dict，`runtime.train_module` 按 `opt`/`sched`/`loss`/`reg` 字段组装，你不需要写 optimizer 代码，也不能写——这是编译器的边界。

要更快的 BPTT，加 `--compile`（`torch.compile(mode="reduce-overhead")`，实测 25.25 → 8.56 ms/batch，2.95×，warmup 约 22 s）。

---

## 3. 推理

推理是独立的 `infer` 子命令，从 checkpoint 恢复，不需要训练。

### 3.1 单样本：看它怎么决策

```bash
snndsl infer examples/mnist_membrane.snn \
    --weights build_mem/mnist_membrane.pt --samples 1 --ascii
```

```
--- input as the encoder receives it ---
        -#*+:.
        %@@@@@########*:
        :=:=*%@%@@@@%@@+
              : :::: @@=
                    -@%
                    @@-
                   +@@.
                  :@@:
                  +@#
                  %@:
                 =@#
                :@@:
                %@*
               #@%.
              .@@-
             .%@=
             +@@:
            :@@@:
            =@@%.
            =@%
  (28x28, mean intensity 9%)

--- decision ---
  class  7   99.95%  ████████████████████████████████████
  class  3    0.02%  █
  class  2    0.02%  █
  -> predict 7   (true label 7)

--- spikes fired while deciding ---
  h1           rate  29.1%        595 spikes
  y            rate  28.8%         23 spikes
  total                            618 spikes

--- cost of this decision (1 sample, 8 time steps) ---
  SNN event-driven : 7391.262 nJ  (5,950 AC + 1,605,632 MAC)
  dense ANN        : 7480.115 nJ  (1,626,112 MAC)
  ratio            : 1.01x
  latency          : 1.51 ms
```

这段输出把推理链路摊开了：**输入**（编码器真正收到的像素）、**决策**（top-k 概率）、**代价**（哪个层发了多少脉冲、多少 nJ、多少毫秒）。稀疏不是口头声明，是 618 个实际非零事件。

`--image your.png` 可以直接喂文件，走同一条路径。

### 3.2 全量测试集 + 延迟

```bash
snndsl infer examples/mnist_membrane.snn \
    --weights build_mem/mnist_membrane.pt --samples 10000 --repeats 10
```

```
--- held-out accuracy ---
  97.21% on 10000 samples

--- sparsity ---
  h1           rate  34.9%
  y            rate  24.5%

--- latency (10000 samples/batch, 10 repeats) ---
  mean  389.09 ms   p50  398.74 ms   p95  480.67 ms
  throughput 25,701 samples/s
```

25,701 samples/s 是 CPU 上 fp32 的 PyTorch 执行。真机部署走 NIR（`snndsl export`），这里测的是功能正确性，不是硬件峰值。

### 3.3 Python API

命令行只是壳，`DeployedModel` 可以直接用：

```python
from snndsl.infer import DeployedModel

net = DeployedModel.load("examples/mnist_membrane.snn",
                         weights="build_mem/mnist_membrane.pt",
                         out_dir="build_mem")

import torch
from torchvision import datasets, transforms

ds = datasets.MNIST("./data", train=False, transform=transforms.ToTensor())
x = torch.stack([ds[i][0] for i in range(8)])      # (8, 1, 28, 28)

r = net.predict(x, topk=3)
print(r["pred"], [round(c, 4) for c in r["confidence"]])
print(r["topk"])                                    # 第 0 个样本的 top-3
print(r["spikes_per_layer"])                        # 每层发放率与脉冲数
print(r["latency_ms_per_sample"])

# 延迟分布
print(net.benchmark(x, repeats=50))
```

`predict` 返回的字段：`pred` / `confidence` / `topk` / `scores` / `spikes_per_layer` / `spikes_per_sample` / `latency_ms`。批量调用返回每一行的结果，不是只有第一行。

### 3.4 Python 前端同样可训可推

`examples/mnist_py.py` 用 Python 构建网络，编译、训练、推理、NIR 导出全部走同一条链路：

```python
import snndsl as snn

LIF = snn.LIF(tau=2.0, v_th=1.0)
net = snn.Net("mnist_py", steps=8, readout="membrane")
x = net.encode(net.input(1, 28, 28), "rate", gain=1.0)

h = x
for i in range(3):                       # 循环展开 3 个 block
    h = net.neuron(net.dense(h, 256, scale=2.0, name=f"h{i+1}"), LIF, name=f"a{i+1}")

net.output(net.neuron(net.dense(h, 10, name="y"), LIF, name="ay"))
net.train(data="mnist", epochs=4, lr=1e-3, loss="ce", sched="cosine")

metrics = net.run(out_dir="build_py")    # 等价于 snndsl run
```

```bash
python examples/mnist_py.py --dry-run       # 只看图
python examples/mnist_py.py --layers 3      # 循环生成 3 层并训练
```

实测：Python 前端编译出的模块，加载 `.snn` 训练出的 checkpoint，
`missing=[] / unexpected=[]`，MNIST 全量 10000 张 **97.21%**——与 `.snn` 前端逐样本一致。
两个前端产出同一个 IR（op 序列、参数量、MAC、生成代码行数全部相同）。

---

## 4. 一个真实的坑：readout 选错会毁掉置信度

`mnist_mlp.snn` 用 `readout=spikes`（输出层脉冲计数除以 T）。同一个网络换成 `mnist_membrane.snn` 的 `readout=membrane`（发放前膜电位），差别巨大：

| | spike readout | membrane readout |
|---|---|---|
| 训练到 96.5% 以上所需 | 10 epoch / 243 s | **4 epoch / 96 s** |
| 测试精度（10000 张） | 96.52% | **97.21%** |
| int8 PTQ 后 | 96.50% | 97.19% |
| 单样本 top-1 置信度 | 23.20% | **99.95%** |
| 输出层发放率 | 10.3% | 24.5% |

**为什么 spike readout 的置信度是假的**：输出层 LIF 的输入乘了 `scale=2.0`，一旦超过阈值，膜电位每步都被推到阈值以上，于是该神经元在全部 8 步都发放（rate=1.0），其余 9 个全程沉默（rate=0）。logits 变成严格的 one-hot：

```
row0 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
```

softmax 一个 one-hot 向量，得到的永远是 23.20% / 8.53% / 8.53%——**与网络、与输入完全无关**。

两个证据：

1. 我第一次跑推理时，两个独立训练的模型（`mnist_mlp` 和 `quickstart`，参数不同）给出**完全相同**的 top-3 分布。
2. 同一模型换 4 个不同类别的样本，`pred` 是 `[7, 2, 1, 0]`（全对），但 `confidence` 全是 `0.232`：

```json
{"pred": [7, 2, 1, 0], "conf": [0.232, 0.232, 0.232, 0.232]}
```

argmax 仍然对（所以准确率 96.5% 是真的），但"置信度"没有任何信息量，也丢掉了类间边际，白白浪费了 4 个 epoch。

改 `readout=membrane` 后，logits 是发放前的连续膜电位，保留了证据强度，置信度恢复正常，精度更高、训练更快。**这是 SNN 分类头的标准做法，写在这里免得你踩同一个坑。**

---

## 5. 能耗：诚实的数字与敏感性

`mnist_mlp.snn` 报告 1.01× 节能，也就是基本没省。原因在结构：

```
ANN dense :    1,626,112 MAC  ->  7480.12 nJ
SNN event :        6,173 AC +  1,605,632 MAC ->  7391.46 nJ
```

`784→256` 这一层占 98.7% 的 MAC，而它的输入是 rate 编码的**模拟电流**（`encode x -> rate`），不是脉冲，按规则必须按 MAC 计费。剩下 0.1% 的 AC 优惠救不了全局。

能效完全取决于首层能否吃到稀疏脉冲。报告里的敏感性扫描：

| 首层输入发放率 | SNN 能耗 | 相对 dense ANN |
|---|---|---|
| 100%（模拟电流） | 1450.9 nJ | 5.16× |
| 50% | 728.4 nJ | 10.27× |
| 30% | 439.4 nJ | 17.04× |
| 13%（≈MNIST 平均像素强度） | 193.7 nJ | 38.61× |
| 5% | 78.3 nJ | 95.75× |

结论：把 `encode x -> rate` 换成 `-> poisson`，或接事件相机/脉冲传感器直连，才能落进高倍率区间。稠密图像 + 恒流编码省不下来。**这个结论比一个漂亮倍数有用。**

---

## 6. 完整流程

```bash
# 编译 -> 训练 -> 量化 -> 稀疏度/能耗 -> 产物
snndsl run examples/mnist_membrane.snn --out build_mem

# 推理（单样本，带输入预览）
snndsl infer examples/mnist_membrane.snn \
    --weights build_mem/mnist_membrane.pt --samples 1 --ascii

# 推理（全量 + 延迟）
snndsl infer examples/mnist_membrane.snn \
    --weights build_mem/mnist_membrane.pt --samples 10000 --repeats 10

# 导到神经形态硬件生态（Lava / SpiNNaker2 / Speck / Xylo 都读 NIR）
snndsl export examples/mnist_membrane.snn \
    --weights build_mem/mnist_membrane.pt -o build_mem/mnist.nir

# 单文件 HTML 编译报告
snndsl report examples/mnist_membrane.snn \
    -o build_mem/report.html --weights build_mem/mnist_membrane.pt
```

`run` 产出三样东西，都可独立使用：`build_mem/mnist_membrane.py`（可读 PyTorch 源码，1.4 ms 生成）、`.ir.json`（静态 IR，供其他后端消费）、`.nir`（硬件中间表示）。

---

## 7. 边界

- 无循环突触与轴突延迟，所以不支持时间递归 SNN（EventProp 那类）。
- `quant=int8` 是 weight-only PTQ（fake quant）：权重真被量化，体积与精度损失是实测的，但没生成 int8 kernel，实际加速取决于部署框架。
- 事件驱动能耗是成本模型 + 实测稀疏度，不是位精确稀疏 kernel 的执行。
- 硬件链路止步于 NIR 导出，未上真机验证。
- 精度是未调参的 baseline。BatchNorm 折叠、更多 epoch、direct coding 都还没做，97.21% 不是上限。
