# snndsl — 脉冲神经网络的领域特定语言与编译器

[![CI](https://github.com/lishangli/snndsl/actions/workflows/ci.yml/badge.svg)](https://github.com/lishangli/snndsl/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![tests](https://img.shields.io/badge/tests-70%20passed%20%2F%204%20skipped-brightgreen)](tests/)
[![license](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![version](https://img.shields.io/badge/version-0.2.0-informational)](CHANGELOG.md)

用两种前端描述脉冲神经网络（拓扑、神经元动力学、时间语义、训练配方、部署预算）：
一份声明式文本 `.snn`，或直接写 Python。两者编译成**同一个静态 SNN IR**，
经过优化后降级为可执行的 PyTorch 模块，并能导出 NIR 直接对接神经形态硬件
（Loihi / SpiNNaker2 / Speck / Xylo 等）。

不是又一个 SNN 运行时库——`snnTorch`、`SpikingJelly`、`Norse` 已经做得很好。
这里补的是**编译层**：把网络当程序来写、来检查、来优化、来量化成本。

```
 .snn 文本 ─┐
            ├─→ 语义分析/形状推断 ─→ SNN IR ─→ 常量折叠·DCE·融合·内存规划·时间调度
 Python 嵌入 ┘                          ↓
              ┌───────────────────┬──────────────────────┬────────────────┐
        PyTorch 模块源码      事件驱动稀疏推理        NIR（神经形态硬件）
        （BPTT 训练）        （能耗/发放率分析）      （Lava/Speck/Xylo）
```

为什么值得做、相比已有项目的差异：**[`docs/WHY.md`](docs/WHY.md)**。
在生态里的位置（含真实 GitHub 数据）：**[`docs/ECOSYSTEM.md`](docs/ECOSYSTEM.md)**。
接下来怎么走：**[`docs/ROADMAP.md`](docs/ROADMAP.md)**。
六个网络的部署结论（五种不同答案，含一个卷积网逐位一致的正面案例）：**[`docs/P0_DEPLOY_MATRIX.md`](docs/P0_DEPLOY_MATRIX.md)**。
同一份文件换个目标为什么不一样（三处规范缺口 + 实测）：**[`docs/SEMANTICS.md`](docs/SEMANTICS.md)**。
用户访谈怎么问、什么结果算失败：**[`docs/INTERVIEW_KIT.md`](docs/INTERVIEW_KIT.md)**。
一份能直接发给别人的部署报告：[`docs/deploy_report_mnist.html`](docs/deploy_report_mnist.html)。
**还差什么、算谁的账**：[`docs/GAPS.md`](docs/GAPS.md)——缺口登记表，每条附验证命令。
改了什么、撤回了什么：[`CHANGELOG.md`](CHANGELOG.md)。怎么贡献：[`CONTRIBUTING.md`](CONTRIBUTING.md)。

## 1. 快速开始

```bash
pip install -e ".[full]"                        # 只写编译器？`pip install snndsl` 就够（numpy）
snndsl doctor                                     # 这台机器上哪些部分真的能用

snndsl inspect examples/mnist_membrane.snn        # 看 IR、形状、优化报告
snndsl build   examples/mnist_membrane.snn        # 编译出 .py + .ir.json + .nir
snndsl run     examples/mnist_membrane.snn        # 编译→训练→量化→测稀疏度/能耗
snndsl infer   examples/mnist_membrane.snn \
               --weights build/mnist_membrane.pt --samples 1 --ascii   # 推理
snndsl report  examples/mnist_membrane.snn -o r.html  # 单文件 HTML 编译报告
snndsl export  examples/cifar_cnn.snn -o c.nir   # 导出 NIR
snndsl deploy  examples/mnist_hw.snn --verify    # 能上硬件吗？等价吗？
```

训练与推理的完整可复现记录（含真实终端输出）见
**[`examples/TRAINING_AND_INFERENCE.md`](examples/TRAINING_AND_INFERENCE.md)**。

作为库使用：

```python
from snndsl import compile_source, load_module

res  = compile_source(open("examples/mnist_membrane.snn").read())
Net  = load_module(res, "build/net.py")   # 生成的是真实 Python 源码
net  = Net()
```

推理侧：

```python
from snndsl.infer import DeployedModel

net = DeployedModel.load("examples/mnist_membrane.snn",
                         weights="build_mem/mnist_membrane.pt")
out = net.predict(batch_of_images, topk=3)
out["spikes_per_layer"]   # 每层发放率与脉冲数
```

## 2. Python 嵌入前端

同一个网络，用 Python 写。适合需要循环、条件、函数复用的真实工程。
**两个前端走同一条命令**：文件是 `.py` 时 CLI 会导入它、取其中的网络，
后面所有子命令（`inspect` / `build` / `run` / `deploy` / `report`）完全一样。

```bash
snndsl inspect examples/mnist_py.py     # 与 inspect examples/*.snn 等价
```

模块需要暴露一个零参数的 `build()` 或模块级 `net`；两者都没有时报错会直接说清楚。

```python
import snndsl as snn

LIF = snn.LIF(tau=2.0, v_th=1.0)            # 神经元方程复用一份

net = snn.Net("mnist_py", steps=8, readout="membrane")
x = net.encode(net.input(1, 28, 28), "rate", gain=1.0)

h = x
for i in range(3):                          # 循环展开 3 个 block
    h = net.neuron(net.dense(h, 256, scale=2.0, name=f"h{i+1}"), LIF, name=f"a{i+1}")

net.output(net.neuron(net.dense(h, 10, name="y"), LIF, name="ay"))
net.train(data="mnist", epochs=4, lr=1e-3, loss="ce", sched="cosine")

res = net.compile()                         # 与 .snn 完全相同的 pass 与 codegen
```

可用的神经元：`snn.LIF` / `snn.IF` / `snn.ALIF`，以及 `snn.custom(name, params, state, body)`
自写方程（`body` 用 DSL 的数学语法，同样会被校验）。

**两个前端产出同一个 IR**，可以验证：

| 检查项 | `.snn` | Python | 一致 |
|---|---|---|---|
| op 序列 | input,encode,dense,neuron,dense,neuron,output | 同左 | ✅ |
| 参数量 | 203,530 | 203,530 | ✅ |
| MAC | 1,626,112 | 1,626,112 | ✅ |
| 生成代码行数 | 119 | 119 | ✅ |

checkpoint 也跨前端通用：`.snn` 训出来的权重，直接加载进 Python 前端编译的模块，
`missing=[] / unexpected=[]`，MNIST 全量 10000 张同样是 **97.21%**。

文本 DSL 的循环做不到这件事（`pipeline:` 是平铺的），这是 Python 前端唯一的**表达力**优势，
其余好处是编辑体验（补全、断点、类型检查）。

## 3. DSL 速览

一个完整程序（`examples/mnist_mlp.snn`）：

```
program mnist_mlp

const T_STEPS = 8
const HIDDEN  = 256

temporal:                  # 全局时间语义
    steps = T_STEPS
    dt = 1.0
    reset = subtract

neuron LIF:                # 神经元动力学 = 有序的更新语句
    param tau = 2.0        # 参数（编译期折叠为常量）
    param v_th = 1.0
    state v = 0.0          # 状态（运行期每个时间步携带）
    v = v * (1 - dt / tau) + i     # i 是上游注入的突触电流
    spike s = v > v_th             # 阈值比较 → 自动降级为代理梯度
    v = v - s * v_th               # 复位
    surrogate = fast_sigmoid(alpha=4.0)

encoder rate:
    gain = 1.0

pipeline:                  # 数据流，名字可重绑定（SSA）
    input  x  : 1, 28, 28
    encode x  -> rate with gain=1.0
    dense  h1 : HIDDEN <- x  with scale=2.0
    neuron h1 -> LIF
    dense  y  : 10 <- h1     with scale=2.0
    neuron y  -> LIF
    output out <- y

train:                     # 训练配方，编译器据此生成训练循环
    data = mnist
    epochs = 3
    batch = 128
    lr = 0.001
    opt = adam
    loss = ce
    sched = cosine

infer:                     # 部署意图
    mode = event
    quant = int8

budget:                    # 硬约束，编译后校验
    spikes_per_sample = 40000
```

### 3.1 语法要点

| 概念 | 写法 | 说明 |
|---|---|---|
| 常量 | `const K = 2 * 3 + 1` | 编译期折叠，可参与形状与超参 |
| 神经元 | `neuron NAME:` + `param/state/spike` | 语句**顺序执行**，支持任意多状态变量 |
| 自定义阈值 | `spike s = v > v_th + beta * a` | 任意布尔表达式，只要形如 `LHS > RHS` 即自动接代理梯度 |
| 代理梯度 | `surrogate = fast_sigmoid(alpha=4.0)` | 支持 `fast_sigmoid / atan / triangle / super_spike / rectangular` |
| 编码 | `encode x -> rate\|poisson\|latency` | `rate` 恒流、`poisson` 伯努利采样、`latency` 首脉冲时刻 |
| 层 | `dense/conv/pool/flatten/dropout` | 空间算子用 `(C,H,W)`，向量算子用 `(F,)` |
| 合并 | `add z : a, b` / `concat z : a, b` | 构造 DAG，不是纯链式 |
| 神经元挂载 | `neuron h -> LIF` | 对同名值就地施加神经元，产生脉冲版本 |
| 属性 | `with k=3, p=1, scale=2.0` | 逐层超参，无全局隐式默认 |

`pipeline` 中的名字是**变量**，允许重绑定：`dense h1` 定义它，`neuron h1` 覆盖为脉冲版本。
编译器内部转换为 SSA，因此 IR 始终是干净的 DAG（见测试 `test_rebinding_creates_distinct_values`）。

### 3.2 诊断

所有错误带 `文件:行:列` 与源码定位：

```
examples/cifar_cnn.snn:35:37: error[sema]: unknown identifier 'max' in pool p1
        pool   p1 : 2 <- c1   with kind=max
                                        ^
```

## 4. 编译器

### 4.1 前端 → IR

两个前端，一个后端。它们唯一的共同出口是 `pipeline.finalize()`，
所以优化 pass 和代码生成不可能出现两套行为。

| 前端 | 入口 | 图怎么产生 |
|---|---|---|
| 文本 `.snn` | `compile_source` / `compile_file` | lexer → 递归下降 parser → AST → 语义分析 |
| Python 嵌入 | `snn.Net(...)` | `_add()` 直接构造 IRNode，形状随手校验 |

文本前端内部：
- **词法**：缩进敏感（INDENT/DEDENT），Python 风格；括号内忽略换行
- **语法**：递归下降 + Pratt 表达式解析（运算符优先级对齐 Python，ODE 可直接写）
- **语义**：名字解析、常量折叠、形状推断、神经元方程校验（符号作用域检查）
- **IR**：SSA 数据流图，形状全静态，`(C,H,W)` 或 `(F,)`，batch/time 由后端补齐

神经元方程被降级为 Python 源码内联进生成模块，`v > v_th` 变成
`surrogate_fast_sigmoid((v) - (v_th), 4.0)`——前向硬 0/1，反向有梯度。
这条降级逻辑在 `sema.build_neuron_spec()`，两个前端共用，无法分叉。

### 4.2 优化 pass（每一项都有可测的 before/after）

| pass | 做什么 | 实测效果（examples） |
|---|---|---|
| `dce` | 从 output 反向求可达，删除死分支 | 死分支节点数 → 0 |
| `fuse` | flatten 折进 dense；推理期删除 dropout | CIFAR: 激活内存 534,608 → 262,144 B (**−51%**) |
| `memplan` | 按活跃区间做区间图着色，复用激活 buffer | MNIST: 8,400 → 6,272 B (**−25%**)，反向传播型网络 −33% ~ −42% |
| `schedule` | T ≤ 4 展开，T > 4 循环 | T=8 时不生成 32 份代码而只保留 4 层 |
| `direct_input`（可选） | 首层用模拟电流替代脉冲编码 | 去掉 1 个 encode 节点与其全尺寸中间 buffer |

buffer 复用是真实计算：对每个值求 `[定义位置, 最后使用位置]` 区间，贪心着色分配物理槽位，
同时给出「不复用总量 / 理论峰值 / 实际分配」三个数（`live_peak_bytes`，有单测覆盖）。

### 4.3 代码生成

输出**人类可读的 PyTorch 源码**，而不是 pickle 一个对象：

```python
# neuron h1 model=LIF
_st = st["h1$1"]
dt = self.DT
t = t
tau = 2.0
v_th = 1.0
i = v_h1
v = _st["v"]
v = ((v * (1 - (dt / tau))) + i)
s = surrogate_fast_sigmoid((v) - (v_th), 4.0)
v = (v - (s * v_th))
_st["v"] = v
v_h1_1 = s
```

好处：可 diff、可 audit、可手改、可被任何 PyTorch 工具链消费。

## 5. 训练与推理

### 5.1 训练：BPTT + 代理梯度

`train:` 块直接驱动训练循环（优化器/loss/调度/正则/梯度裁剪），
编译器生成 `spike_penalty()` 供 `spike_l2` 稀疏正则使用。

### 5.2 推理：事件驱动成本模型

推理后端统计每层发放率，用 45 nm CMOS 常数（Horowitz ISSCC 2014：MAC=4.6 pJ，AC=0.9 pJ）
计算「dense ANN vs 事件驱动 SNN」的能耗：

- ANN：每个突触一次 MAC
- SNN：仅当输入有脉冲时累加，`AC = 发放率 × fan-in × fan-out`

预算块（`budget: spikes_per_sample = ...`）在编译产物上校验，超标直接标红。

部署态推理走 `snndsl infer` 或 `DeployedModel`，从 checkpoint 恢复、不需要训练：

```bash
snndsl infer examples/mnist_membrane.snn \
    --weights build_mem/mnist_membrane.pt --samples 1 --ascii
```

输出包括输入预览（ASCII）、top-k 概率、每层实际脉冲数、单次决策的 nJ 与延迟。
完整示例见 [`examples/TRAINING_AND_INFERENCE.md`](examples/TRAINING_AND_INFERENCE.md)。

`infer: readout` 决定分类头语义：`spikes`（脉冲计数，输出层易饱和成 one-hot）
或 `membrane`（发放前膜电位，**推荐**，实测 4 epoch 97.21% vs 10 epoch 96.52%）。

### 5.3 神经形态硬件：NIR

导出 [NIR](https://github.com/neuromorphs/NIR)（Neuromorphic Intermediate Representation，
Nature Communications 2024）。节点映射：

| SNN IR | NIR primitive |
|---|---|
| `dense` + bias | `Affine` |
| `dense` 无 bias | `Linear` |
| `conv` | `Conv2d` |
| `avg/max pool` | `AvgPool2d` / `SumPool2d` |
| `flatten` | `Flatten` |
| `neuron`（标准 LIF 方程） | `LIF`（参数按 per-neuron 数组） |
| `neuron`（非标准方程） | `LI` 近似 + 报告中标注 |
| `encode` / `decode` | 无对应原语，按电流直连并在报告中说明 |
| `add` / `concat` | NIR 无合并原语，报告中标注需目标框架处理 |

导出后可被 Lava/Loihi、SpiNNaker2、Speck/Xylo、Sinabs、snnTorch、Norse、Spyx、
BrainScaleS-2 读取。带训练权重导出：`snndsl export x.snn --weights build/x.pt`。

## 5.5 部署检查（`deploy`）

编译出 `.nir` 只证明文件合法，不证明**网络还是同一个网络**。`deploy` 回答后者。

```bash
$ snndsl deploy examples/mnist_hw.snn --verify
[DEPLOYABLE] readout=spikes  steps=8
  info  h1$1   zero reset maps exactly onto NIR's v_reset
  info  out    spike readout maps 1:1 onto a NIR spiking layer

--- NIR equivalence check ---
  mean |error|     : 0.000e+00
  max  |error|     : 0.000e+00
  top-1 agreement  : 1.000
  verdict          : PRESERVED
```

四块能力：

| 命令 | 做什么 | 需要数据吗 |
|---|---|---|
| `deploy <file>` | 静态走 IR，列出每一个活不过去的东西，给出修法 | 不需要 |
| `deploy <file> --verify` | 导出 NIR → 跑独立参考解释器 → 和 PyTorch 逐样本比 | 不需要 |
| `deploy <file> --report r.html` | 出一份单文件 HTML，能直接发给没见过本项目的人 | 建议 `--data mnist` |
| `deploy <file> --targets` | 把同一份文件喂给**别人的**框架，看谁读得进去、谁跑得对 | 建议同上 |

三种严重度：`BLOCK`（不改就上不了）、`WARN`（能上，但行为会变）、`info`。

### `--report`：一份能转发的文件

```bash
$ snndsl deploy examples/mnist_hw.snn --weights build_hw/mnist_hw.pt \
        --data mnist --batch 32 --report deploy_report.html
```

单文件、无外链、无 `<script>`、图表是内联 SVG——这类文件会被邮件转发、
在权限受限的机器上打开，需要联网才能渲染的报告在最需要它的时候是白屏。

它按"读者先问什么"排：结论横幅 → 数字卡 → 静态检查 → 数值等价 →
别人的工具链能不能读 → 成本 → 复现命令 → **这份报告没证明什么**。
最后一节是刻意的：报告越像结论，越容易被当成保证。

| 伪阳性 | 怎么处理 |
|---|---|
| 全零权重的文件 | `strict=True` 拒绝写盘，报告标 `EMPTY` |
| 验的是内存图不是文件 | 先写盘再 `nir.read` 读回，`nir_source=file` |
| 噪声输入下网络不发放，"0.0 等于 0.0" | 判 `vacuous`，报告写"这次比较不携带信息" |

### `--targets`：厂商工具链到底吃不吃

```bash
$ snndsl deploy examples/mnist_hw.snn --weights build_hw/mnist_hw.pt --data mnist
--- foreign frontends (same file, same currents) ---
target                     vendor      status          detail
nirtorch + snnTorch        neuroir     ok              max|Δ|=0.00e+00  top1=1.000
sinabs (Speck / DynapCNN)  SynSense    load-only       能加载能跑，非二值脉冲流，时间模型不同
rockpool (Xylo) frontend   SynSense    upstream-gap    上游前端仍在读 nir<1.0 字段名（tau_mem）
rockpool modules (Xylo)    SynSense    load-only       已用 rockpool 自己的神经元跑通；11.7% 的
                                                      (神经元,时间步) 一步发多枚脉冲；文件声明
                                                      zero reset、目标只有 subtract；单独换 tau
                                                      约定，脉冲位置变化 6.45%
```

最后一行是重点：rockpool 的前端读不了 nir 1.0，但它的**神经元**是可用的，于是绕开
前端、用它自己的 `LinearTorch` + `LIFNeuronTorch` 重建同一张图。算术是 SynSense 的。
量下来的结论（完整表在 [`docs/SEMANTICS.md`](docs/SEMANTICS.md)）：

> **把 `tau` 的换算方式对齐之后，两家实现在非饱和区间逐脉冲完全一致**
> （8 步 × 256 样本 × 256 神经元 = 524,288 个位置，0 处不同）。
> 而 `tau` 直传（大多数人的做法）会让 1.19%–16.50% 的脉冲位置出错。
> 真实网络的差异主要来自"一步该发几枚脉冲"——标准对此没有规定，
> rockpool 发 `⌊v/θ⌋` 枚，我们的 LIF 发 1 枚。

**这三件事（tau 的离散化、每步脉冲数、reset 形式）在 NIR 1.0 里都没写死。**
所以同一份文件在不同目标上就是不同的模型，而用户往往到集成期才知道。

这一步抓出了两个**上游**问题（详见 [`docs/UPSTREAM_ISSUES.md`](docs/UPSTREAM_ISSUES.md)）：

- `nir` 1.0.x 的补丁版本之间文件不兼容：1.0.8 写的文件，1.0.4 打开报
  `unexpected keyword argument 'v_reset'`。厂商都钉 1.0.4。
  现在 `v_reset` 为 0 时不写该字段（nir 自己也是零填充兜底，语义不变），
  整条 1.0.x 都能读。
- `rockpool` 3.1 的 NIR 前端还在读 `tau_mem`（nir 1.0 已改名 `tau`），
  任何 nir 1.0 模型都进不去 Xylo 路径。三个工具链还把 `nir` 钉在不同版本，
  装在一起会把我们的降级——所以所有探测都在**独立子进程**里跑，
  用 `--vendor-python` 指到另一套环境。

### 它会抓到什么

这三个例子都是真实发现的、而不是编出来的场景：

**① 声明的 reset 和方程的 reset 不一致**

```snn
temporal:
    reset = zero                 # 头里这么写
neuron LIF:
    v = v - s * v_th             # 方程里这么做
```

`temporal` 的 `reset` 只是文档，真正跑的是 neuron body。头改了、body 没改，
网络行为一点没变。检查器实测神经元在 `v=3, s=1` 后的落点来判定真实语义：
落在 2 是 subtract，落在 0 是 zero。

```
BLOCK h1$1  the temporal header declares reset='zero' but the neuron equations
            implement reset='subtract'; the header is documentation, the body is
            what runs
```

**② 电流增益被漏掉（真实的量纲 bug）**

DSL 写 `v = v*(1 - dt/tau) + i`，电流是**未缩放**直接加进膜电位的。
翻译成 NIR 的 `tau*dv/dt = (v_leak - v) + r*i` 时，`r` 必须等于 `tau/dt`。
早期版本写死 `r = 1.0`，于是导出的网络驱动少了一半。

现在用数值探测从方程里反解系数（不靠模式匹配文本）：

```
a = 1 - dt/tau  =>  tau = dt/(1-a)
c = (dt/tau)*v_leak
b = (dt/tau)*r  =>  r = b*tau/dt
```

**③ 分类头用的是膜电位，硬件只能给脉冲**

`readout=membrane` 取的是 reset **前**的膜电位；NIR 的脉冲神经元输出的是**脉冲**。
这两个东西不能互换，所以这是 blocker 而不是警告：

```
BLOCK out  the classification head reads the membrane potential before reset,
           but a NIR spiking neuron emits spikes, not potentials
        -> either train a spike-readout head, or add a non-spiking readout layer
```

### 端到端证据

`examples/verify_deployment.py` 在真实 MNIST 测试集上逐样本比对两个实现的预测：

```bash
$ python3 examples/verify_deployment.py
{
 "samples": 1024,
 "pytorch_acc": 0.9434,
 "nir_acc": 0.9434,
 "prediction_agreement": 1.0
}
```

**1024 张样本，两个实现给出完全相同的预测。** 这是"训练的网络就是部署的网络"
这句话目前唯一不自证的证据形式。

配套的 `backend/nir_sim.py` 是一个按 NIR 规范写的参考解释器（LIF/LI/IF/Linear/
Affine/Conv/Pool/Flatten）。遇到它不建模的节点（如 `Delay`）会直接抛错而不是静默跳过——
静默跳过比报错危险得多。

## 5.6 第三方交叉验证（`cross_validate.py`）

`deploy --verify` 用的是本仓库自己的 NIR 解释器。对不信任这套代码的人，那仍然是自证。
这个脚本把链条上每一环都换成别人的实现：

| 环节 | 由谁实现 |
|---|---|
| 读 `.nir` | `nir` 1.0.8（neuromorphs 官方） |
| 图 → 可执行模块 | `nirtorch.load` 2.6 |
| Linear / Affine / Conv / Pool / Flatten | nirtorch 的 `DEFAULT_MAP` |
| 神经元的积分、阈值、复位 | `snnTorch` 的 `snn.Leaky` 1.0.0 |
| 参数换算 | 本项目（`beta = 1 - dt/tau`、驱动增益 `dt/tau*r`） |

```bash
$ python3 examples/cross_validate.py
--- third-party cross-validation ---
  graph engine  : nirtorch.load
  neuron        : snnTorch snn.Leaky
  parameter mapping (ours -> snnTorch):
    h1_1       beta=0.5  th=1.0  gain=1.0  offset=0.0
    y_1        beta=0.5  th=1.0  gain=1.0  offset=0.0
  accuracy ours        : 94.34%
  accuracy third-party : 94.34%
  prediction agreement : 100.00%
  max |output diff|    : 0.000e+00
  verdict              : CONSISTENT
```

1024 张真实 MNIST，逐元素相等（不是"误差很小"）。这份映射随结果一起打印，
可以逐项核对。详见 **[`docs/P0_CROSS_VALIDATION.md`](docs/P0_CROSS_VALIDATION.md)**。

## 6. 复用的开源项目

调研了现有生态（2025）后的定位：

| 项目 | 类型 | 与本项目的关系 |
|---|---|---|
| [NIR](https://github.com/neuromorphs/NIR) | 硬件中间表示 | **直接依赖**，作为硬件后端 |
| [snnTorch](https://github.com/jeshraghian/snntorch) | PyTorch SNN 库 | 代理梯度公式一致（Neftci et al. 2019），语义可对齐 |
| [SpikingJelly](https://github.com/fangwei123456/spikingjelly) | PyTorch SNN 库 | 参考其多步传播模式；本项目用生成代码而非模块组合 |
| [Sango](https://github.com/sandialabs/Sango) | Python 内嵌组合 DSL | 同为 SNN DSL，但它是 Python 内嵌 + NumPy 仿真；本项目是独立文本语言 + 编译到 torch/硬件 |
| [cuSpike](https://github.com/kocatepedogu/cuspike) | CSM DSL → CUDA | 目标是最小化仿真器；本项目目标是可训练网络的编译与成本分析 |
| [snn-compilation-infra](https://github.com/wwwwanhonghuang/snn-compilation-infrustrature) | IR + TVM 路线 | 同方向探索；本项目选择 torch 代码生成作为可立即训练的后端 |
| PyTorch / torchvision | 张量 + 数据 | 训练后端 |

依赖只有 `numpy`（必需）、`torch`/`torchvision`（训练）、`nir`（硬件导出）、
`matplotlib`（报告），全部可选降级。

## 7. 实测结果

以下数字来自本仓库在 32 核 CPU（无 GPU）上的真实运行，命令与产物都可复现。

### 7.1 MNIST MLP-SNN（`examples/mnist_mlp.snn`，spike readout）

| 指标 | 数值 | 说明 |
|---|---|---|
| 参数量 | 203,530 | 784→256→10 |
| 训练 | 10 epoch / 243 s | BPTT + fast_sigmoid 代理梯度，Adam + cosine |
| 测试精度 | **96.52%** | 未做超参搜索的 baseline |
| 编译耗时 | 1.4 ms | parse → sema → 4 pass → codegen |
| 生成代码 | 118 行 | 可读 PyTorch 源码 |
| 激活内存 | 8,400 → 6,272 B（−25.3%） | `memplan` buffer 复用，2 个槽位 |
| 发放率 | h1 31.8%，y 10.3% | 平均每步非零比例 |
| 脉冲数 | 660 / 样本 | 预算 40,000，PASS |
| int8 量化 | 795 KiB → 203.6 KiB（**3.90×**） | per-channel 对称，精度 96.52% → 96.50%（−0.02 pp） |
| 能耗（事件驱动） | 7,391 nJ vs ANN 7,480 nJ | **仅 1.01×**，原因见下 |
| `torch.compile` | 25.25 → 8.56 ms/batch（**2.95×**） | `--compile`，warmup 约 22 s |

**为什么能耗只有 1.01×？** 因为第一层（784→256）占了 98.7% 的 MAC，
而它的输入是 rate 编码的**模拟电流**，不是脉冲，必须按 MAC 计费。
这是 MLP 结构决定的，不是实现问题。报告里的敏感性分析给出结论：

| 首层输入发放率 | SNN 能耗 | 相对 dense ANN |
|---|---|---|
| 100%（模拟电流） | 1450.9 nJ | 5.16× |
| 50% | 728.4 nJ | 10.27× |
| 30% | 439.4 nJ | 17.04× |
| 13%（≈MNIST 平均像素强度） | 193.7 nJ | 38.61× |
| 5% | 78.3 nJ | 95.75× |

也就是说：**SNN 的能效优势完全取决于首层能否吃到稀疏脉冲**。
把 `encode x -> rate` 换成 `-> poisson`，或使用事件相机/脉冲传感器直连，
就能落进高倍率区间；用稠密图像 + 恒流编码，省不下来。这个结论比一个漂亮的
倍数更有用。

### 7.2 readout 的选择决定分类头质量（`examples/mnist_membrane.snn`）

同一个网络（203,530 参数、T=8、同一套 LIF），只改 `infer: readout`，差别很大：

| | `readout=spikes` | `readout=membrane` |
|---|---|---|
| 达到 96.5% 以上所需 | 10 epoch / 243 s | **4 epoch / 96 s** |
| 测试精度（10000 张全量） | 96.52% | **97.21%** |
| int8 PTQ 后 | 96.50% | 97.19% |
| 单样本 top-1 置信度 | 23.20%（假） | **99.95%** |
| 输出层发放率 | 10.3% | 24.5% |
| 推理吞吐（CPU） | 25,019 samples/s | 25,701 samples/s |

**`spikes` readout 的置信度是假的。** 输出层 `scale=2.0` 让膜电位每步都被推到阈值以上，
于是每个样本的输出都是严格 one-hot（一个类 rate=1.0，其余 0.0）。softmax 一个 one-hot
向量永远得到 23.20% / 8.53% / 8.53%，**与网络、与输入都无关**——两个独立训练的模型给出
完全相同的 top-3，就是这个原因。argmax 仍然正确（96.5% 是真的），但类间边际信息全丢了，
白训练 6 个 epoch。

`membrane` readout 用发放前的膜电位当 logits，保留证据强度，精度更高、收敛更快、置信度有
意义。这是 SNN 分类头的标准做法。复现：

```bash
snndsl run   examples/mnist_membrane.snn --out build_mem
snndsl infer examples/mnist_membrane.snn --weights build_mem/mnist_membrane.pt --samples 1 --ascii
```

### 7.3 CIFAR-10 CNN-SNN（`examples/cifar_cnn.snn`，仅编译侧验证）

| 指标 | 数值 |
|---|---|
| 参数量 | 2,193,226 |
| 每步 MAC | 74,529,792 |
| 节点数 | 15 → 14（`fuse` 折叠 flatten） |
| 激活内存 | 534,608 → 262,144 B（**−51.0%**） |
| 编译耗时 | ~4 ms |

未在本仓库训练到收敛（CPU 上代价过高），仅用于验证卷积/池化/展平算子链与融合 pass。

## 8. 示例

| 文件 | 展示 |
|---|---|
| `examples/quickstart.snn` | 12 行有效代码的最小程序，2 epoch 到 93.8% |
| `examples/mnist_mlp.snn` | 完整 MLP-SNN：rate 编码、int8 PTQ、预算检查（spike readout） |
| `examples/mnist_membrane.snn` | 同网络换 membrane readout，4 epoch 到 97.21% |
| `examples/cifar_cnn.snn` | conv / max-pool / flatten、per-layer scale、flatten 融合 |
| `examples/cifar_hw.snn` | 同上换 zero reset，**max pool 导致不可部署**（NIR 无 `MaxPool2d`） |
| `examples/cifar_hw_avg.snn` | 只把池化换成 avg，**卷积网全链路逐位一致**（`max\|Δ\|=0.0`） |
| `examples/mnist_conv.snn` | **两层卷积、训练后 98.06%**，自己仿真与第三方都 0 误差 |
| `examples/adaptive_lif.snn` | 自定义神经元：双状态变量 + 自适应阈值 + atan 代理梯度 |
| `examples/residual_snn.snn` | DAG 分支 + `add` 合并 + direct_input pass |
| `examples/mnist_py.py` | **Python 前端**：`net.dense(...)` 构建同一网络，`--layers N` 用循环展开 |

`examples/TRAINING_AND_INFERENCE.md` 是训练与推理的完整可复现记录，含真实终端输出；
`docs/WHY.md` 是与已有项目的差异分析和价值判断。

## 9. 测试

```bash
python -m pytest -q                       # 77 passed, 4 skipped
python -m pytest --cov=snndsl             # 覆盖率
python -m ruff check snndsl tests examples
```

只有 numpy 的机器上（不装 torch、不装 nir）：`31 passed, 7 skipped`，而且
`snndsl build` 照样产出 `.py` + `.ir.json` —— 编译器核心不依赖 torch 不是一句
口号，`tests/test_artifacts.py` 在子进程里屏蔽掉 torch 的 import 机制来证明它，
因为一个装了 torch 的环境没法自证“不需要 torch”。缺后端时命令给的是
`error[backend]: this command needs PyTorch...` 加怎么装，不是 traceback。

覆盖：词法/语法错误定位、常量折叠、形状推断、SSA 重绑定、DCE、buffer 复用、
flatten 融合、时间步调度、direct_input、生成模块可前向可反传、状态跨调用隔离、
NIR 往返、能耗模型单调性、readout 语义、**无脉冲神经元拒绝**、循环连接诊断、
**双前端 IR 等价**（op 序列/参数量/生成代码行数）、Python 循环展开、自定义神经元。

部署相关（`tests/test_deploy.py`，11 个）：电流增益 `r=tau/dt` 的推导、
ALIF 不被误判为 LIF、membrane readout 必须报 blocker、subtract reset 必须报警告、
**header 与 body 不一致必须报 blocker**、zero reset 下与 NIR 参考实现**逐位一致**、
subtract 下确实发散（预期行为）、解释器与手写递推一致、
**batch 大小绝不被误读成时间步**、未建模节点必须抛错。

导出完整性（`tests/test_export_integrity.py`，9 个）：没权重时 strict 必须拒绝导出、
round-trip 必须能发现丢失的字段、**校验器读的必须是磁盘文件而不是内存图**、
零权重文件必须被判"空壳"而不是"一致"、**恒定输出必须判无效比较**、
第三方后端必须复现我们每一个脉冲。

NIR → snnTorch 桥（`tests/test_bridge.py`，5 个）：每条都锁一个"能加载、能跑、
数字是错的"的缺陷——未映射算子必须落到 nirtorch 的 `DEFAULT_MAP`、
未知算子仍须抛错、**平均池化不能是求和**、多维神经元参数必须报错而不是被平均、
逐样本探测必须清零膜电位。

4 个跳过的是厂商探针（sinabs / rockpool），它们把 `nir` 钉在 1.0.4，
必须装进独立解释器再 `export SNNDSL_VENDOR_PYTHON=...`，做法见 `CONTRIBUTING.md`。

## 10. 局限（明确写下）

> 这一节是结论，不是清单。**持续维护的缺口登记表在 [`docs/GAPS.md`](docs/GAPS.md)**：
> 每条缺口都标了归属（我们的 bug / 上游的 bug / 标准的空白）和一条能当场跑的命令。
> 归属很重要——这个项目已经三次把自己的缺陷写成别人的问题，每次看上去都很专业。

- 只实现了前馈 + 残差类 DAG：**无循环突触/延迟**，因此不支持真正的时间递归 SNN
  （EventProp 类方法）。`nir.Delay` 已在导出层预留位置。
- `quant = int8` 实现的是 **weight-only PTQ（fake quant）**：权重真正被量化并反量化，
  体积与精度损失是实测的；但没有生成 int8 推理 kernel，实际加速取决于部署框架。
- 精度是未调参的 baseline（MNIST MLP 96.5%）。加 BatchNorm 折叠、更多 epoch、
  direct coding 与更好初始化应该能到 97–98%，这些没有做。
- 事件驱动推理是**成本模型 + 稀疏度统计**，不是位精确的稀疏 kernel 实现；
  能耗数字用于比较设计选择，不是芯片实测。
- 硬件后端止步于 NIR 导出与**参考解释器**，未在任何真实芯片上验证。
  `deploy --verify` 与 `cross_validate.py` 证明的是"导出的 NIR 在规范语义下与训练实现一致"，
  不等于"在 Xylo/Loihi 上跑出来一致"——后者还取决于目标编译器的实现质量。
- **卷积 SNN 可以导出**（这条早前写错了，已更正）。原话是"NIR 的神经元层由一维
  逐神经元参数数组决定，吃不下 `(C,H,W)`，所以 `conv → neuron → conv` 在标准内
  无法表达"。实测不成立：NIR 的神经元从**参数数组自身的形状**推导 `input_type`
  （`LIF.__post_init__` 就是 `input_type = {"input": np.array(r.shape)}`），
  所以 `(C,H,W)` 的逐元素参数完全合法，元素个数与展平形式**一模一样**，只是 rank
  不同——也就是说，不用付出任何体积代价。`examples/cifar_hw_avg.snn` 的完整卷积
  链路 `max|error| = 0.0`、top-1 一致率 1.000。当初的报错来自我们自己把参数写成了
  一维向量，再由 `deploy` 直接以"多维激活"为由 `BLOCK`，把实现问题记在了标准头上。
- **真正挡住卷积的是 max pooling**：NIR 1.0.x 只定义 `AvgPool2d` 和 `SumPool2d`，
  没有 `MaxPool2d`。`examples/cifar_hw.snn` 用了 `kind=max`，所以它仍是
  `NOT DEPLOYABLE`。这条是标准的能力边界，改成 `kind=avg` 立刻通过。早前这个
  池化在导出时被**静默写成 SumPool2d**（2×2 求和把驱动放大四倍，首层发放率
  0.02 → 0.37），现在会报 `BLOCK` 并写明原因。
- **`readout=membrane` 无法无损部署**：NIR 的脉冲神经元输出脉冲，不输出 reset 前的膜电位。
  这是标准层面的约束，不是实现缺陷。需要膜电位分类头时，只能在目标侧额外读状态。
- **`subtract` reset 无法精确表达**：NIR 只有固定 `v_reset`。实测差异在大驱动下可达
  17% 的脉冲率偏差。zero reset 是逐位精确的，所以 `examples/mnist_hw.snn` 用它。
  代价是 4 epoch 精度 94.81%，低于 subtract 版的 97.21%（后者不可无损部署）。
- 第三方交叉验证只用了一个后端组合（NIRTorch + snnTorch），它现在同时覆盖全连接与
  卷积两类网络（都是 `max|Δ| = 0.00e+00`），但**仍然只是一个后端**。再加一个
  （Nengo 或 Lava）会更强。
- **跨后端的对照只做到 Python 神经元这一层**：`nirtorch` 自己的 `AvgPool2d` 映射
  把平均池化实现成了求和（2×2 窗口恒差 4 倍），我们只能在自己的桥接层绕开它。
  这类"映射表里有、语义不对"的问题，只有当有人逐层对过脉冲才会暴露——
  本项目的 `deploy --targets` 就是为此存在的。

## 11. 目录

```
snndsl/
├── errors.py            诊断与源码定位
├── lexer.py             缩进敏感词法
├── ast.py               语法树
├── parser.py            递归下降 + Pratt
├── sema.py              名字解析/常量折叠/形状推断/神经元降级
├── ir.py                SNN IR（SSA 图 + 时间语义）
├── passes/              优化 pass（每项输出 before/after 指标）
├── pipeline.py          编译驱动
├── backends/
│   ├── runtime.py       代理梯度/编码/数据/训练循环/能耗常数
│   ├── torch_codegen.py IR → PyTorch 源码
│   └── nir_export.py    IR → NIR
├── runner.py            训练+稀疏度+能耗+预算校验
├── infer.py             部署后端：predict / 逐层发放率 / 延迟基准
├── deploy.py            部署分析：静态阻断 + 数值等价 + 外部工具链
├── deploy_html.py       单文件 HTML 部署报告（内联 SVG，无脚本无外链）
├── report.py            单文件 HTML 编译报告
└── cli.py               命令行入口（build/run/inspect/infer/export/report/deploy/doctor）
```

`backends/` 里还有四个文件是部署侧的核心，读代码从它们开始：

| 文件 | 作用 |
|---|---|
| `nir_export.py` | IR → NIR：写盘后回读自证、方言降级、`strict` 拒绝空壳 |
| `nir_sim.py` | 独立的 NIR 参考解释器：逐位复现、readout、输出脉冲流 |
| `targets.py` | 厂商探针：nirtorch / sinabs / rockpool，各自跑在独立子进程 |
| `snntorch_bridge.py` | NIR → snnTorch 的映射，三处上游陷阱的绕行都在这里 |

文档：

| 文件 | 内容 |
|---|---|
| `docs/WHY.md` | 为什么需要编译层，与已有项目的差异 |
| `docs/ECOSYSTEM.md` | 生态位，含真实 GitHub 数据 |
| `docs/GAPS.md` | 缺口登记表：归属、状态、验证命令 |
| `docs/SEMANTICS.md` | 三处规范缺口 + 实测 |
| `docs/P0_DEPLOY_MATRIX.md` | 七个网络的部署结论 |
| `docs/UPSTREAM_ISSUES.md` | 四条可提交的上游 issue（含一条撤回记录） |
| `docs/ROADMAP.md` | 阶段、验收标准、证伪条件 |
