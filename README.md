# snndsl

[![CI](https://github.com/lishangli/snndsl/actions/workflows/ci.yml/badge.svg)](https://github.com/lishangli/snndsl/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![version](https://img.shields.io/badge/version-0.3.0-informational)](CHANGELOG.md)

用一份 `.snn` 文本或几行 Python 描述脉冲神经网络，编译成四样东西：
**可读的 PyTorch 源码**、**带数字的推理成本**、**能交给神经形态硬件的 NIR**、
**能被 MLIR 工具链读的 `.mlir`**。
外加一个别人不做的事——回答「这样导出去之后，还是同一个网络吗」，逐脉冲给证据。

## 它补的是哪一段

`snnTorch`、`SpikingJelly`、`Norse` 已经能训练 SNN，`NIR` 已经能把 SNN 交给芯片。
缺的是中间那层：把网络当**程序**来检查、优化、量化成本，并在导出前后对一次账。
所以这里没有新的神经元模型、没有新的仿真器，只有编译层。

中间层现在有别人了：SNN-MLIR 和 SNAX 用 MLIR/xdsl 做 NIR 往下的 lowering，做得比本项目
深。空着的是另一半——**从源码往上**，以及在训练开始前就说清「这个网络上不了硬件」。
两者的边界、以及为什么不硬接他们的后端，写在 [docs/ECOSYSTEM.md](docs/ECOSYSTEM.md) §7。

| 你的处境 | snndsl 给什么 |
|---|---|
| 写完网络，不知道能不能上硬件 | `snndsl deploy` 走一遍 IR，列出每个过不去的东西和改法，不需要数据 |
| 导出成功，但不知道还是不是同一个网络 | `deploy --verify` 把文件从磁盘读回来，跑独立解释器，逐样本比 |
| 想知道稀疏到底省了多少能量 | 事件驱动成本模型 + 每层实际发放率，附首层稀疏度敏感性 |
| 编译产物是个黑盒 | 生成人类可读的 Python 源码：可 diff、可 audit、可手改 |
| 换个目标为什么结果不一样 | `deploy --targets` 把同一份文件喂给厂商工具链，把差异量化出来 |

## 安装

尚未发布到 PyPI（`0.2.x` 阶段），从源码装：

```bash
git clone https://github.com/lishangli/snndsl && cd snndsl
pip install -e ".[full]"        # 训练 + 硬件导出 + 报告（torch / nir / nirtorch）
snndsl doctor                   # 这台机器上哪些部分真的能用，逐项列出
```

只要编译器核心（不训练、不导硬件）就用 `pip install -e .`——它只依赖 numpy。

`snndsl doctor` 是排查第一步：它说清楚缺了哪个后端、哪些子命令因此不可用，
而不是让你在几分钟后收到一个 traceback。只有 numpy 也能 `build`：产物是
`.py` + `.ir.json`，`.nir` 会提示跳过而不是留个空文件。

## 30 秒

`examples/quickstart.snn` 是 25 行、能直接跑的最小网络：

```
program quickstart
temporal:
    steps = 6
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
```

编译它，看编译器怎么理解这个网络（节点数、优化前后、神经元方程被降级成什么）：

```console
$ snndsl inspect examples/quickstart.snn
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

id                    op        shape             inputs
-----------------------------------------------------------------
x                     input     [1, 28, 28]
x$1                   encode    [1, 28, 28]       x
h                     dense     [128]             x$1
h$1                   neuron    [128]             h
y                     dense     [10]              h$1
y$1                   neuron    [10]              y
out                   output    [10]              y$1

neurons  :
  LIF: params={'tau': 2.0, 'v_th': 1.0} states={'v': 0.0} surrogate=fast_sigmoid
      v = ((v * (1 - (dt / tau))) + i)
      s = surrogate_fast_sigmoid((v) - (v_th), 4.0)
      v = (v - (s * v_th))
```

训练 + 评测（首次运行会下载 MNIST；2 epoch 到 93.8%，CPU 上约 40 秒）：

```console
$ snndsl run examples/quickstart.snn --epochs 2
  epoch 1/2  loss=1.5782  train_acc=86.56%  test_acc=92.69%  (19.3s)
  epoch 2/2  loss=1.5128  train_acc=93.19%  test_acc=93.83%  (37.3s)

--- sparsity (avg firing rate over T) ---
  h             34.7%
  y             11.4%
--- event-driven energy (per sample) ---
  ANN dense :      609,792 MAC  ->  2805.04 nJ
  SNN event :        2,664 AC +    602,112 MAC ->  2772.11 nJ
  ratio     : 1.01x lower energy than the dense ANN
```

那 1.01× 不是失败，是 MLP 的真实结论——原因和怎么改见 [`docs/RESULTS.md`](docs/RESULTS.md)。

## 部署检查：导出去还是同一个网络吗

编译出 `.nir` 只证明文件合法，不证明**网络还是同一个网络**。`deploy` 回答后者：

```console
$ snndsl deploy examples/mnist_hw.snn --verify
[DEPLOYABLE] readout=spikes  steps=8
  info  x$1   encode has no NIR equivalent and is dropped; the target receives raw currents instead
  info  h1$1  zero reset maps exactly onto NIR's v_reset
  info  h1$1  input current is scaled by r=2 in NIR terms (tau/dt); the exporter carries this factor
  info  h1$1  LIF maps to NIR LIF (tau=2, v_th=1)
  info  out   spike readout maps 1:1 onto a NIR spiking layer

--- NIR equivalence check ---
  checked artifact   : deploy_check.nir (read back from disk)
  steps              : 8
  max  |error|       : 0.000e+00
  top-1 agreement    : 1.000
  WARNING            : 未指定 --weights，验的是随机初始化网络；下面的数字不属于任何训练好的模型
  verdict            : PRESERVED · 随机权重
```

`BLOCK`（不改就上不了）、`WARN`（能上，但行为会变）、`info`（记录发生了什么）
三种严重度，每条都带修法。`deploy` 需要 `[full]`（编译靠 `torch`，等价检查靠 `nir`）；
`inspect` / `build` 只要 numpy。两个真实会遇到的 `BLOCK`：

```
BLOCK h1$1  the temporal header declares reset='zero' but the neuron equations
            implement reset='subtract'; the header is documentation, the body is
            what runs
BLOCK out   the classification head reads the membrane potential before reset,
            but a NIR spiking neuron emits spikes, not potentials
         -> either train a spike-readout head, or add a non-spiking readout layer
```

四块能力：

| 命令 | 做什么 | 要数据吗 |
|---|---|---|
| `deploy <file>` | 静态走 IR，列出每一个活不过去的东西 | 不要 |
| `deploy <file> --verify` | 导出 → 独立参考解释器 → 与 PyTorch 逐样本比 | 不要 |
| `deploy <file> --report r.html` | 单文件 HTML 报告，可直接转发（内联 SVG，无脚本无外链） | 建议 `--data mnist` |
| `deploy <file> --targets` | 同一份文件喂给**别人的**框架，看谁读得进、谁跑得对 | 建议同上 |

在 1024 张真实 MNIST 上的两种证据，`max |Δ|` 都是 `0.000e+00`、预测一致率 100%：
`examples/verify_deployment.py` 用本仓库的 NIR 解释器，`examples/cross_validate.py`
把每一环换成别人的实现（`nir` + `nirtorch` + `snnTorch` 的神经元）。
后者是「不是我们自己说的」那一种证据。

## 语言

一个完整程序（`examples/mnist_mlp.snn`，注释略）：

```
program mnist_mlp

const T_STEPS = 8
const HIDDEN  = 256

temporal:                  # 时间语义
    steps = T_STEPS
    dt = 1.0
    reset = subtract

neuron LIF:                # 神经元动力学 = 有序的更新语句
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i     # i 是上游注入的突触电流
    spike s = v > v_th             # 阈值比较 → 自动降级为代理梯度
    v = v - s * v_th               # 复位
    surrogate = fast_sigmoid(alpha=4.0)

encoder rate:              # 输入编码：恒流正比于像素强度
    gain = 1.0

pipeline:                  # 数据流，名字可重绑定（SSA）
    input  x  : 1, 28, 28
    encode x  -> rate with gain=1.0
    dense  h1 : HIDDEN <- x  with scale=2.0
    neuron h1 -> LIF
    dense  y  : 10 <- h1     with scale=2.0
    neuron y  -> LIF
    output out <- y with readout=spikes

train:                     # 训练配方，编译器据此生成训练循环
    data = mnist
    epochs = 10
    batch = 128
    lr = 0.001
    opt = adam
    loss = ce
    seed = 0
    sched = cosine

infer:                     # 部署意图
    mode = event
    readout = spikes       # spikes | membrane
    quant = int8

budget:                    # 硬约束，编译后校验，超了直接标红
    spikes_per_sample = 40000
    latency_ms = 5.0
```

| 概念 | 写法 | 说明 |
|---|---|---|
| 常量 | `const K = 2 * 3 + 1` | 编译期折叠，可参与形状与超参 |
| 神经元 | `neuron NAME:` + `param/state/spike` | 语句**顺序执行**，任意多状态变量 |
| 自定义阈值 | `spike s = v > v_th + beta * a` | 任意布尔表达式，形如 `LHS > RHS` 即自动接代理梯度 |
| 代理梯度 | `surrogate = fast_sigmoid(alpha=4.0)` | `fast_sigmoid / atan / triangle / super_spike / rectangular` |
| 编码 | `encode x -> rate\|poisson\|latency` | 恒流 / 伯努利采样 / 首脉冲时刻 |
| 层 | `dense / conv / pool / flatten / dropout` | 空间算子用 `(C,H,W)`，向量算子用 `(F,)` |
| 合并 | `add z : a, b` / `concat z : a, b` | 构造 DAG，不是纯链式 |
| 属性 | `with k=3, p=1, scale=2.0` | 逐层超参，无全局隐式默认 |

`pipeline` 里的名字是**变量**，允许重绑定：`dense h1` 定义它，`neuron h1` 覆盖为脉冲版本，
编译器内部转 SSA，IR 始终是干净 DAG。

不想用文本？同一个网络也能用 Python 写，走同一条命令链：

```python
import snndsl as snn

LIF = snn.LIF(tau=2.0, v_th=1.0)
net = snn.Net("mnist_py", steps=8, readout="membrane")
x = net.encode(net.input(1, 28, 28), "rate", gain=1.0)
h = x
for i in range(3):
    h = net.neuron(net.dense(h, 256, scale=2.0, name=f"h{i+1}"), LIF, name=f"a{i+1}")
net.output(net.neuron(net.dense(h, 10, name="y"), LIF, name="ay"))
net.train(data="mnist", epochs=4, lr=1e-3, loss="ce", sched="cosine")
res = net.compile()                        # 与 .snn 完全相同的 pass 与 codegen
```

两个前端产出同一个 IR，可以核对（`examples/mnist_py.py` 与 `examples/mnist_mlp.snn`）：
op 序列一致、参数量都是 203,530、MAC 都是 1,626,112、生成代码都是 119 行。
checkpoint 也通用：`.snn` 训出来的权重直接加载进 Python 前端编译的模块，
`missing=[] / unexpected=[]`，MNIST 全量同一精度。

## 命令

| 命令 | 作用 |
|---|---|
| `snndsl inspect <file>` | 编译并打印 IR、形状、每个 pass 的 before/after |
| `snndsl build <file>` | 编译出 `.py`（可读 PyTorch 源码）+ `.ir.json` + `.nir` + `.mlir` |
| `snndsl run <file>` | 编译 → 训练 → 量化 → 测稀疏度/能耗/预算 |
| `snndsl infer <file> --weights w.pt` | 从 checkpoint 推理：top-k、逐层脉冲数、nJ、延迟 |
| `snndsl export <file> -o x.nir` | 只导出 NIR，可带 `--weights` |
| `snndsl report <file> -o r.html` | 单文件 HTML 编译报告 |
| `snndsl deploy <file> [--verify] [--report] [--targets]` | 部署检查 |
| `snndsl fold <file.nir> [-o out.nir]` | 把按时间展开的 `.nir` 收回成一个时间步，并把 T 记进 metadata |
| `snndsl fidelity <file.nir>` | 同一份文件在量化 / 另一种 tau 离散化下，脉冲逐位置差多少 |
| `snndsl doctor` | 这台机器上哪些能力可用 |

作为库用：

```python
from snndsl import compile_source, load_module

res = compile_source(open("examples/mnist_membrane.snn").read())
Net = load_module(res, "build/net.py")     # 生成的是真实 Python 源码，可断点可 diff
net = Net()
```

```python
from snndsl.infer import DeployedModel

net = DeployedModel.load("examples/mnist_membrane.snn",
                         weights="build_mem/mnist_membrane.pt")
out = net.predict(batch_of_images, topk=3)
out["spikes_per_layer"]                     # 每层发放率与脉冲数
```

## 示例

| 文件 | 展示 |
|---|---|
| `examples/quickstart.snn` | 25 行有效代码的最小程序，2 epoch 到 93.8% |
| `examples/mnist_mlp.snn` | 完整 MLP-SNN：rate 编码、int8 PTQ、预算检查（spike readout） |
| `examples/mnist_membrane.snn` | 同网络换 membrane readout，4 epoch 到 97.21% |
| `examples/mnist_conv.snn` | **两层卷积，训练到 98.06%**，自己仿真与第三方运行时都 0 误差 |
| `examples/cifar_hw_avg.snn` | 卷积 + avg pool，全链路逐位一致（`max\|Δ\| = 0.0`） |
| `examples/cifar_hw.snn` | 同上换 max pool，**不可部署**（NIR 无 `MaxPool2d`），用来看 blocker 长什么样 |
| `examples/cifar_cnn.snn` | conv / max-pool / flatten、per-layer scale、flatten 融合 |
| `examples/adaptive_lif.snn` | 自定义神经元：双状态变量 + 自适应阈值 + atan 代理梯度 |
| `examples/residual_snn.snn` | DAG 分支 + `add` 合并 + `direct_input` pass |
| `examples/mnist_py.py` | Python 前端，与 `.snn` 版本 IR 等价 |

`examples/TRAINING_AND_INFERENCE.md` 是训练与推理的完整可复现记录，含真实终端输出。

## 效果

全部在 32 核 CPU（无 GPU）上跑出来，没有超参搜索：

| 网络 | 精度 | 备注 |
|---|---|---|
| MNIST MLP-SNN（spike readout） | 96.52% | 10 epoch / 243 s，int8 PTQ 后 96.50%、体积 3.90× 更小 |
| 同网络（membrane readout） | **97.21%** | 4 epoch / 96 s，收敛更快、置信度有意义 |
| MNIST 两层卷积 | **98.06%** | 3 分钟，int8 PTQ 97.94%，第三方运行时逐位一致 |

一个反直觉但有用的结论：MLP 上事件驱动的能耗优势只有 **1.01×**，
因为首层（784→256）占了 98.7% 的 MAC，而它的输入是 rate 编码的模拟电流，必须按 MAC 计费。
首层输入发放率降到 13%（≈MNIST 平均像素强度）时是 38.6×。**能效优势完全取决于首层能否吃到稀疏脉冲。**

完整数字、能耗敏感性分析、readout 对分类头的影响：[`docs/RESULTS.md`](docs/RESULTS.md)。

## 现状与边界

`0.2.1`，Apache-2.0。CI 覆盖 Python 3.10–3.13，81 个测试（77 通过 / 4 跳过，
跳过的是需要独立解释器的厂商工具链探针）。

能做的事：

- 前馈与残差类 DAG 的 SNN，全连接与卷积都行，`avg` 池化逐位无损导出 NIR
- NIR 导出后与本仓库的参考解释器、以及第三方运行时（nirtorch + snnTorch）逐脉冲对齐
- int8 weight-only PTQ、事件驱动成本模型、spike 预算校验、单文件 HTML 报告

做不到，或者只做到一半（完整登记表见 [`docs/GAPS.md`](docs/GAPS.md)，每条标了归属和验证命令）：

| 边界 | 现状 | 现在怎么办 |
|---|---|---|
| 循环突触 / 延迟 | 不支持，只有前馈 + DAG | 用 `examples/residual_snn.snn` 那类结构；`nir.Delay` 已在导出层预留 |
| max pooling | **NIR 1.0.x 没有 `MaxPool2d`**，导出时直接 `BLOCK` | 换 `kind=avg`（`examples/cifar_hw_avg.snn` 因此全链路 0 误差） |
| int8 量化 | 是 weight-only PTQ（fake quant），不生成 int8 kernel | 体积与精度损失是实测的；加速取决于部署框架 |
| `readout=membrane` | 无法无损部署（硬件给脉冲，不给 reset 前膜电位） | 上硬件用 spike readout，或单独加不发放的读出层 |
| 真实芯片 | 未在 Xylo / Loihi / Speck 上验证，只对过厂商的 Python 神经元 | 定点与位宽行为属于未测项，本地工作改变不了 |
| 数据集 | 只有 MNIST 系与 CIFAR 的示例 | 训练路径不绑定 MNIST，但别的数据没跑过 |

## 文档

| 文件 | 内容 |
|---|---|
| [`docs/WHY.md`](docs/WHY.md) | 为什么需要编译层，与已有项目的差异 |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | 编译器内部：前端 → IR → pass → codegen |
| [`docs/RESULTS.md`](docs/RESULTS.md) | 实测数字与能耗敏感性分析 |
| [`docs/P0_DEPLOY_MATRIX.md`](docs/P0_DEPLOY_MATRIX.md) | 七个网络的部署结论（五种不同答案） |
| [`docs/SEMANTICS.md`](docs/SEMANTICS.md) | 同一份文件换个目标为什么不一样：三处规范缺口 + 实测 |
| [`docs/GAPS.md`](docs/GAPS.md) | 缺口登记表：归属、状态、验证命令 |
| [`docs/UPSTREAM_ISSUES.md`](docs/UPSTREAM_ISSUES.md) | 四条可提交的上游 issue |
| [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md) | 生态位，含真实 GitHub 数据 |
| [`docs/ECOSYSTEM_GAPS.md`](docs/ECOSYSTEM_GAPS.md) | 训练/交换/编译/硬件五段缺陷清单，含外部模型的实测数字 |
| [`docs/PYTHON_FRONTEND.md`](docs/PYTHON_FRONTEND.md) | 从 PyTorch 往下：哪一段能复用，分水岭是时间循环 |
| [`docs/deploy_report_mnist.html`](docs/deploy_report_mnist.html) | 一份可以直接发给别人的部署报告 |

## 依赖与致谢

必需依赖只有 `numpy`。训练后端是 PyTorch；硬件后端是
[NIR](https://github.com/neuromorphs/NIR)（Neuromorphic Intermediate Representation）；
逐脉冲对照用的是 [nirtorch](https://github.com/neuromorphs/nirtorch) 与
[snnTorch](https://github.com/jeshraghian/snntorch)。
代理梯度公式取自 Neftci et al. 2019，能耗常数取自 Horowitz, ISSCC 2014。
与 SpikingJelly / Sango / cuSpike 等项目的定位差异见 [`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md)。

## 参与

贡献前读 [`CONTRIBUTING.md`](CONTRIBUTING.md)——那里有一条这个项目的基本规则：
**没有复现命令的结论不写进来**。bug 报告请附 `snndsl doctor` 输出和最小的 `.snn` 文件。
安全相关问题走 [`SECURITY.md`](SECURITY.md)。

引用的工作请用 [`CITATION.cff`](CITATION.cff)。许可 [Apache-2.0](LICENSE)。
