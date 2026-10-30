# 从零到部署：一条命令链

这一页只回答一个问题：**我想训练一个 SNN 并把它部署出去，从头到尾要敲哪些命令。**
按顺序复制即可，每一步都给出在这台机器上真实跑出来的输出。

全流程 6 步，3～5 分钟（3 epoch 训练 77 秒，其余都是秒级）：

```
写 .snn ──► 问能不能上硬件 ──► 训练 ──► 导出 .nir ──► 验证还是同一个网络 ──► 交给硬件生态
   │             │              │           │                │                      │
 inspect       deploy         run        export        deploy --verify        deploy --targets
                                                    + 第三方运行时复跑         + report.html
```

下面所有例子都用 `examples/mnist_hw.snn`——它是本仓库里**能无损部署**的那个网络
（`readout=spikes` + zero reset）。先跑通它，再换成你自己的。

---

## 第 0 步：装，然后问这台机器能干什么

```bash
git clone https://github.com/lishangli/snndsl && cd snndsl
pip install -e ".[full]"
snndsl doctor
```

```console
component      status                       required
------------------------------------------------------
python         3.11.1                       yes
snndsl         0.3.0                        yes
numpy          2.3.5                        yes
torch          2.10.0+cu128                 no
torchvision    0.25.0+cu128                 no
nir            1.0.8                        no
nirtorch       2.6                          no
matplotlib     3.10.8                       no

what works right now
  compile to Python  : yes
  train / infer      : yes
  export NIR         : yes
  verify against NIR : yes
  third-party probes : yes
```

**为什么要先跑这一步**：`doctor` 会告诉你哪几个子命令在这台机器上真的会工作。
只有 numpy 也能编译（产出 `.py` + `.ir.json`），只是不能训练、不能导 NIR。
先知道边界，后面就不会把「缺依赖」误读成「我的网络有问题」。

---

## 第 1 步：写网络，然后在训练之前就问「它能上硬件吗」

`examples/mnist_hw.snn` 全文（去掉空行 30 行）：

```
program mnist_hw
const T_STEPS = 8
const HIDDEN  = 256

temporal:                  # 时间语义：8 步，dt=1，发放后清零
    steps = T_STEPS
    dt = 1.0
    reset = zero

neuron LIF:                # 神经元 = 有序的更新语句，不是选一个预置模型
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v * (1 - s)
    surrogate = fast_sigmoid(alpha=4.0)

pipeline:                  # 数据流
    input  x  : 1, 28, 28
    encode x  -> rate with gain=1.0
    dense  h1 : HIDDEN <- x  with scale=2.0
    neuron h1 -> LIF
    dense  y  : 10 <- h1     with scale=1.0
    neuron y  -> LIF
    output out <- y

train:
    data = mnist
    epochs = 10
    batch = 128
    lr = 0.001

infer:
    mode = event
    readout = spikes

budget:
    spikes_per_sample = 40000
```

现在**不要训练**，先花一秒检查它能不能导出：

```bash
snndsl deploy examples/mnist_hw.snn
```

```console
[DEPLOYABLE] readout=spikes  steps=8
  info  x$1   encode has no NIR equivalent and is dropped; the target receives raw currents instead
  info  h1$1  zero reset maps exactly onto NIR's v_reset
  info  h1$1  input current is scaled by r=2 in NIR terms (tau/dt); the exporter carries this factor
  info  h1$1  LIF maps to NIR LIF (tau=2, v_th=1)
  info  y$1   zero reset maps exactly onto NIR's v_reset
  ...
  info  out   spike readout maps 1:1 onto a NIR spiking layer
```

这条命令**不需要数据、不需要权重、不需要 GPU**，2 秒内给出结论。对同一个仓库里
另外四个网络，它给出的是五种不同答案（`BLOCK` / `WARN` / 全 `info`），
矩阵见 [`P0_DEPLOY_MATRIX.md`](P0_DEPLOY_MATRIX.md)。

**这是本项目最省时间的一步**：先知道「这个设计上不了硬件」，比训练 3 小时再发现要便宜。
两个真实会撞到的 `BLOCK`：

```
BLOCK out   the classification head reads the membrane potential *before* reset,
            but a NIR spiking neuron emits spikes, not potentials
         -> either train a spike-readout head, or add a non-spiking readout layer
BLOCK h1$1  the temporal header declares reset='zero' but the neuron equations
            implement reset='subtract'; the header is documentation, the body is what runs
```

想看清楚它是怎么被理解的：

```bash
snndsl inspect examples/mnist_hw.snn      # IR、形状、每个 pass 的 before/after
```

---

## 第 2 步：训练

```bash
snndsl run examples/mnist_hw.snn --epochs 3 --out build_e2e
```

```console
program   : mnist_hw   (mnist_hw.snn)
time      : T=8  dt=1.0  schedule=looped
graph     : 7 nodes, 203,530 params, 1,626,112 MAC (T x step)
input     : [1, 28, 28]
output    : [[10]]
passes    :
    dce          no change
    fuse         no change
  * memplan      activation memory 8400 -> 6272 B (-25.3%) using 2 slot(s)
  * schedule     T=8 > 4 -> looped (keeps code size at 4 layers instead of 32)
artifacts : mnist_hw.py, mnist_hw.ir.json, mnist_hw.nir
compile   : 1.4 ms

training  : mnist  epochs=3  batch=128  lr=0.001
  epoch 1/3  loss=1.5717  train_acc=87.00%  test_acc=91.98%  (25.4s)
  epoch 2/3  loss=1.5086  train_acc=93.44%  test_acc=94.36%  (50.7s)
  epoch 3/3  loss=1.4986  train_acc=94.63%  test_acc=94.47%  (76.8s)
checkpoint: build_e2e/mnist_hw.pt
nir       : 4 tensors embedded, round-trip OK

--- sparsity (avg firing rate over T) ---
  h1            35.3%
  y             11.0%
  mean          23.2%

--- event-driven energy (per sample) ---
  ANN dense :    1,626,112 MAC  ->  7480.12 nJ
  SNN event :        7,235 AC +  1,605,632 MAC ->  7392.42 nJ
  ratio     : 1.01x lower energy than the dense ANN
  spikes/sample: 732
  budget spikes_per_sample: 732 / 40000 OK
```

32 核 CPU、无 GPU，3 epoch 80 秒，MNIST 全量 **94.47%**（10 epoch 是 96.52%）。

三件事值得看一眼，因为它们是编译器做的、不是你做的：

| 输出里的行 | 含义 |
|---|---|
| `compile : 1.4 ms` | 静态图，没有 autograd 追踪，编译比一次前向还快 |
| `memplan 8400 -> 6272 B` | 按活跃区间着色复用 buffer，少 25% 峰值激活内存 |
| `schedule T=8 > 4 -> looped` | T=8 时生成 4 层的循环，而不是摊平成 32 层 |

训练循环本身由 `train:` 块驱动（优化器、调度器、损失都在里面），你不用写，也不能写。

---

## 第 3 步：换成你自己的数据

`data =` 认四种写法，**其他任何名字都会直接报错**——不会悄悄退回内置合成数据集：

| 写法 | 数据放哪 |
|---|---|
| `data = mnist` / `fashion_mnist` / `cifar10` | torchvision 自动下载 |
| `data = "./mydata"` | 目录，`mydata/<类别名>/*.png`，也可再分 `train/` `test/` 两层 |
| `data = "./mydata.npz"` | `x_train`/`y_train`（及可选 `x_test`/`y_test`），uint8 或 float |
| `data = synthetic` | 只有冒烟测试用 |

```
mydata/
├── cat/   000.png 001.png ...
├── dog/   000.png 001.png ...
└── bird/  000.png ...
```

图片会自动 resize 到 `pipeline` 里 `input` 声明的 `(C, H, W)`，单通道图按灰度读。
整段脚本就是换一行：

```
train:
    data = "./mydata"
    epochs = 12
```

实测（120 张 MNIST 风格的 3 分类 png、隐藏层 64、T=6、12 epoch、1.0 秒）：

```console
training  : ./mydata  epochs=12  batch=16  lr=0.002
  epoch 1/12   loss=1.0103  train_acc=51.04%  test_acc=66.67%  (0.0s)
  epoch 3/12   loss=0.7369  train_acc=78.12%  test_acc=95.83%  (0.2s)
  epoch 12/12  loss=0.5514  train_acc=100.00%  test_acc=95.83%  (1.0s)
```

数据集名字写错时的行为（值得看一次，因为它**不会**给你一个假的精度）：

```console
$ snndsl run bad.snn
bad.snn: error[backend]: unknown dataset 'my_dataset'
    hint: use mnist | fashion_mnist | cifar10 | synthetic, or a path to a folder of
          <class>/*.png images, or a .npz with x_train/y_train
```

---

## 第 4 步：导出，然后证明导出去还是同一个网络

```bash
snndsl export examples/mnist_hw.snn --weights build_e2e/mnist_hw.pt -o build_e2e/hw.nir
```

```console
weights   : 4 tensors from build_e2e/mnist_hw.pt
NIR graph : build_e2e/hw.nir
  nodes   : 6
  edges   : 5
  mapping : {'x': 'Input', 'x_1': 'skipped', 'h1': 'Affine', 'h1_1': 'LIF',
             'y': 'Affine', 'y_1': 'LIF', 'out': 'Output'}
```

文件合法不等于网络还是那个网络。加 `--verify`：把文件**从磁盘读回来**，
用独立解释器跑一遍，跟 PyTorch 逐样本比。

```bash
snndsl deploy examples/mnist_hw.snn --verify --weights build_e2e/mnist_hw.pt --data mnist
```

```console
--- NIR equivalence check ---
  checked artifact   : deploy_check.nir (read back from disk)
  batch              : mnist test split, 256 samples
  steps              : 8
  mean |error|       : 0.000e+00
  max  |error|       : 0.000e+00
  top-1 agreement    : 1.000
  verdict            : PRESERVED

--- foreign frontends (same file, same currents) ---
target                     vendor      status          detail
-------------------------------------------------------------------------------------
nirtorch + snnTorch        neuroir     ok              max|Δ|=0.00e+00  top1=1.000
sinabs (Speck / DynapCNN)  SynSense    load-only       loaded and ran, but not a binary spike
                                                        stream (peak 1.2 spikes/neuron/step)
rockpool (Xylo) frontend   SynSense    upstream-gap    reads nir < 1.0 field names (tau_mem)
rockpool modules (Xylo)    SynSense    load-only      87.54% per-spike agreement; tau convention
                                                        (Euler vs exp) moves 5.86% of spikes
```

**第二张表是「不是我们自己说的」那一种证据**：`nirtorch` + `snnTorch` 都是别人的实现，
它们从同一份文件、同一批输入出发，逐位复现出我们的脉冲（`max|Δ| = 0`）。
后面两行是坏消息，也是有用的坏消息——SynSense 的两条链路各自缺一环，缺在哪一句写清了。

---

## 第 5 步：交给硬件生态，以及现在还交不到哪

导出的 `.nir` 是生态里的通用货币，Lava / SpiNNaker2 / Speck / Xylo 都读它：

```bash
snndsl export examples/mnist_hw.snn --weights build_e2e/mnist_hw.pt -o build_e2e/hw.nir
```

想给不在终端前的同事看，生成一份单文件 HTML（内联 SVG，无脚本无外链）：

```bash
snndsl deploy examples/mnist_hw.snn --verify --weights build_e2e/mnist_hw.pt \
    --data mnist --report build_e2e/report.html
```

**必须说清楚的一步之遥**：本仓库没有在 Xylo / Loihi / Speck 真机上跑过。
上表测的是「厂商的 Python 神经元能不能读、能不能复现」，不是「芯片的定点行为」。
定点截断、位宽、片上噪声属于未测项，本地工作改变不了。这条登记在
[`GAPS.md`](GAPS.md) 第 14 条，验证命令也写在那里。

顺带两个只有导出之后才会想知道的问题：

```bash
snndsl fold build_e2e/hw.nir          # 时间维：这份文件怎么表达"跑 8 次"
snndsl fidelity build_e2e/hw.nir      # 量化 / 换 tau 离散化后，脉冲错多少
```

```console
扰动                        脉冲错位       中位       最坏   top-1 一致
--------------------------------------------------------------
int8 权重（逐通道）             0.48%    0.42%    1.17%     100.0%
int4 权重（逐通道）             4.63%    4.32%    7.61%      87.5%
tau：欧拉 → 指数              0.53%    0.52%    1.69%     100.0%

错得最多的一档：int4 权重（逐通道）
  层序（先崩的在前）：h1_1 (4.76%) > y_1 (1.09%)
```

这就是「把 int8 交给硬件之前，先知道会掉多少」的那个数字，并且告诉你先崩的是哪一层。

---

## 第 6 步：推理，以及那个 23.20% 的坑

```bash
snndsl infer examples/mnist_hw.snn --weights build_e2e/mnist_hw.pt --samples 1 --ascii
```

```console
--- decision ---
  class  7   23.20%  ████████
  class  8    8.53%  ███
  class  6    8.53%  ███
  -> predict 7   (true label 7)

--- spikes fired while deciding ---
  h1           rate  36.4%        745 spikes
  y            rate  10.0%          8 spikes

--- cost of this decision (1 sample, 8 time steps) ---
  SNN event-driven : 7392.612 nJ  (7,450 AC + 1,605,632 MAC)
  dense ANN        : 7480.115 nJ  (1,626,112 MAC)
  ratio            : 1.01x
  latency          : 1.55 ms
```

预测对了，但置信度 23.20% 看着像坏了。**这不是 bug，是 spike readout 的固有性质**：
输出层一旦越过阈值就在全部 8 步发放（rate=1.0），其余 9 个全程沉默，
logits 变成严格 one-hot，softmax 一个 one-hot 永远得到 23.20% / 8.53% / 8.53%。
准确率是真的，置信度没有信息量。

想要有意义的置信度就把读出换成膜电位（`readout=membrane`，`examples/mnist_membrane.snn`）：
4 epoch 到 97.21%、置信度 99.95%。**但它部署不了**——硬件给的是脉冲，不是 reset 前的膜电位。
这就是这个工具想让你在训练前就做的那个选择：

| 你要什么 | 选什么 | 代价 |
|---|---|---|
| 上硬件 | `readout=spikes` | 置信度是 one-hot，别拿它做拒识/阈值 |
| 精度和置信度 | `readout=membrane` | `deploy` 会 `BLOCK out`，`.nir` 里那个 head 必须换成不发放的读出层 |

全量测试集与延迟：

```bash
snndsl infer examples/mnist_hw.snn --weights build_e2e/mnist_hw.pt --samples 10000 --repeats 5
```

```console
--- held-out accuracy ---
  94.75% on 10000 samples
--- latency (10000 samples/batch, 5 repeats) ---
  mean  362.78 ms   p50  392.57 ms   p95  396.79 ms
  throughput 27,565 samples/s
```

### 把它放进你自己的程序（软件侧部署）

不想走命令行，三行就够。权重从 checkpoint 加载，返回逐样本预测、每层脉冲数、延迟：

```python
import torch
from snndsl.infer import DeployedModel

net = DeployedModel.load("examples/mnist_hw.snn",
                         weights="build_e2e/mnist_hw.pt",
                         out_dir="build_e2e")
r = net.predict(batch_of_images, topk=3)

r["pred"]                 # [7, 2, 1]
r["topk"]                 # [{'class': '7', 'score': 0.2319}, ...]
r["spikes_per_layer"]     # {'h1': {'rate': 0.33, 'spikes_per_sample': 681.0}, 'y': {...}}
r["latency_ms_per_sample"]  # 0.55
```

（真实输出；那 0.2319 就是上一节说的 one-hot 置信度，别拿它当概率用。）

编译产物 `build_e2e/mnist_hw.py` 也是可直接 import 的普通 `nn.Module`——
它只 import `torch` 和 `snndsl.backends.runtime`（后者只依赖 numpy），
所以交付时带上 `snndsl` 这个包就够，不需要编译器本身：

```python
import sys; sys.path.insert(0, "build_e2e")
import torch
from mnist_hw import SnnNet

net = SnnNet()
ckpt = torch.load("build_e2e/mnist_hw.pt")      # {"state_dict": ..., "program": ..., ...}
net.load_state_dict(ckpt["state_dict"])         # missing=[], unexpected=[]
net(torch.zeros(2, 1, 28, 28))                  # -> (2, 10)
```

---

## 两条捷径：命令行版 和 代码版

命令行：

```bash
bash examples/e2e.sh examples/mnist_hw.snn     # 第 1～6 步，产物在 build_e2e/
```

代码（**推荐先读这个**，它把每一段的 Python 写出来了）：

```bash
python examples/train_deploy.py --epochs 2     # 产物在 build_code/
```

`examples/train_deploy.py` 一个文件跑完六个阶段，每段都是能抄走的一小段：

| 阶段 | 干什么 | 用到的 API |
|---|---|---|
| 1 编译 | `.snn` → IR → 生成的 `.py` | `compile_source` / `load_module` / `result.report_dict()` |
| 2 训练 | 自己写 BPTT，不调本仓库的训练循环 | `rt.build_dataloaders` / `rt.adapt_batch` + 标准 `loss.backward()` |
| 3 导出 | IR + 权重 → `.nir`，再读回来验证 | `export_nir(graph, path, weights=...)` / `nir.read` |
| 4 验证 | PyTorch 一遍、NIR 解释器一遍，逐位置比 | `deploy.verify(graph, net, x, nir_path=...)` |
| 5 测量 | 发放率、事件驱动能耗、int8 体积 | `runner.energy_report` / `rt.quantize_weights_int8` |
| 6 推理 | 脱离编译器，只要 `.py` + `.pt` | `DeployedModel.load` 或直接 `net(x)` |

第 2 段值得单独说：**你不写训练代码，是因为产物就是一个普通 `nn.Module`**，
`forward` 里已经把 T 步时间展开和代理梯度写死了，训练就是标准 BPTT——

```python
Net = load_module(result, "build_code/mnist_hw.py")
net = Net()
for x, y in loader:
    logits = net(rt.adapt_batch(x, (1, 28, 28)))   # 内部跑 T=8 步
    loss = F.cross_entropy(logits, y)
    loss.backward()                                 # 阈值是硬的，梯度来自 surrogate
    opt.step()
```

---

## 我该改哪几行

| 我想做的事 | 改哪里 |
|---|---|
| 换输入尺寸 | `pipeline: input x : C, H, W`，图片会自动 resize |
| 换层数/宽度 | `dense h1 : HIDDEN <- x`，多写几行 `dense` + `neuron` |
| 换成卷积 | `conv c1 : 8 <- x with k=3, p=1` + `pool p1 : 2 <- c1 with kind=avg` |
| 换时间步 | `temporal: steps = T`（>4 自动转循环，不影响语义） |
| 换神经元 | `neuron` 块里加 `state`，任意多条更新语句（见 `examples/adaptive_lif.snn`） |
| 换数据集 | `train: data = ...`（第 3 步） |
| 换优化器/调度 | `train:` 里 `opt = adamw` / `sched = cosine` / `lr = ...` |
| 加分支/残差 | `add z : a, b` / `concat z : a, b`（见 `examples/residual_snn.snn`） |
| 改部署约束 | `budget:` 里 `spikes_per_sample`、`latency_ms`，超了编译后标红 |

---

## 会遇到的四个错误，和它们的真正原因

| 报错 | 真正的原因 | 怎么办 |
|---|---|---|
| `BLOCK out ... reads the membrane potential` | 头读膜电位，硬件只给脉冲 | 换 `readout=spikes`，或加不发放的读出层 |
| `BLOCK ... NIR has no MaxPool2d` | NIR 1.0.x 就没有这个算子 | `pool ... with kind=avg` |
| `error[backend]: unknown dataset` | 名字既不是内置数据集也不是路径 | 用引号写路径：`data = "./mydata"` |
| `WARN ... reset is 'subtract'` | 减式复位在 NIR 里没有对应，只能退化成固定 `v_reset` | 神经元体改成 `v = v * (1 - s)` |

---

## 边界（别把它当承诺）

- 支持前馈与残差类 DAG；**不支持循环突触与轴突延迟**，所以 EventProp 那类时间递归网络不行。
- `quant=int8` 是 weight-only PTQ（fake quant）：权重真的被量化了，体积和精度损失是实测的，
  但没有生成 int8 kernel，实际加速取决于部署框架。
- 能耗是「事件驱动成本模型 + 实测发放率」，不是位精确稀疏 kernel 的执行。
- 硬件链路止步于 `.nir`，未上真机（见第 5 步）。
- 精度是未调参的 baseline，97.21%（membrane）/ 96.52%（spike）都不是上限。

完整的缺口登记在 [`GAPS.md`](GAPS.md)，每条都标了归属、状态和验证命令。
