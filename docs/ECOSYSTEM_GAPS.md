# SNN 生态缺陷清单：哪一段是空的，为什么空，我们补哪一格

> 这份文档回答一个问题：一个已经把 SNN 训练出来的人，要把模型跑到硬件上，
> 现有的东西够不够用；如果不够，缺的是哪一格，那一格为什么到现在还空着。
>
> 判断"缺"的标准有三层，缺一层就不算缺：**格式规范里没有**（不是没人写代码）、
> **实测跑不通**（不是文档没写）、**上游自己承认**（不是我们理解错了）。
> 每条结论下面标出用的是哪一层证据。

---

## 0. 一句话

整条链上，**"一个网络跑 T 步"这件事没有归宿**：训练框架把它展开成 T 份图，
交换格式里没有时间维，硬件上它又是一段重复执行的回路。三段各自都自洽，
**交接处丢失的东西没人负责**。这不是编译器优化问题，是表示问题。

---

## 1. 链上有五个交接面，不是一个流水线

```
    人写的 PyTorch
         │  ① 图捕获
    FX / torch.export 图
         │  ② 语义归一（beta→tau/r、reset 类型、编码方式）
       .nir
         │  ③ 编译（时间维、量化、映射）
    MLIR / C / RISC-V
         │  ④ 上硬件
   Loihi 2 / Speck / Xylo / SpiNNaker2
         │  ⑤ 保真核对（跑出来的脉冲是不是训练的那条）
     报告 / 回归基线
```

每一段都有人，但**每一段都假设上一段已经把它要的东西准备好了**。下面逐段看。

---

## 2. 逐段：谁在、覆盖到哪、交接处丢了什么

### ① 训练：显存是 O(L×T)，解法在论文里，不在库里

| 已有 | 做了什么 | 证据 |
|---|---|---|
| snnTorch / SpikingJelly / Norse / Rockpool / Sinabs | 替代梯度训练，PyTorch 原生 | 均为成熟库，SpikingJelly 自称训练加速 11× |
| 梯度检查点 + 脉冲无损压缩（ICLR 2026） | 峰值显存压到 **0.12×–0.47×**，精度无损，慢 <20% | 论文给出的分解：T=4 时中间特征占峰值显存 **>96%** |
| OTTT / SLTT / OSTL / DECOLLE | 在线或空间解耦，显存 O(1) 或降 **70%+** | 各自牺牲时间梯度或精度 |

**缺口不在算法，在包装。** 上面每一条都是"改你的训练代码"。那篇 ICLR 论文自己
承认的关键点：现有省显存路线"大多要手改网络结构或重写训练代码，又繁琐又容易出错"，
所以它把自己做成一个 `memory_optimization(net)` 的 pass——这个方向是对的，但它
只解决显存，不管导出。

> 这一格我们不动。它是训练侧，和"模型能不能上硬件"是两件事，且已经有活跃的
> 学术供给。列在这里是为了说明：**训练侧的痛点是真的，但已经被认领了。**

### ② 图捕获：时间循环在这一步就没了

这是整条链上**唯一一个已被上游部分解决、而且解法正好是分水岭**的地方。

实测（torch 2.10，两层 LIF，`torch.export` 之后 FX 图节点数）：

| 步数 T | `for t in range(T)` 写法 | `hop.scan` 写法 |
|---:|---:|---:|
| 8 | 136 | **13** |
| 32 | 520 | **13** |
| 128 | 2,056 | **13** |

`for` 版本线性增长，`scan` 版本与 T 无关。SNN 常用 T=100–1000，`for` 版本会给出
1,600–16,000 个节点。原因不是谁写得差：**`torch.export` 把 Python 循环当普通控制流，
它没有理由知道"这个循环是时间"**；而在硬件上时间是一段重复执行的时序回路，
不是 T 份并联电路。

`hop.scan`（`torch._higher_order_ops.scan`）是官方高阶算子，export 会保留它。
但它下划线开头、是私有 API，会随版本漂移——这是这条路唯一的系统性风险。

### ③ 交换格式：NIR 是"单步"格式，而没人把这个前提写进文件

NIR 已经是事实标准：7 个仿真器 + 4 个硬件平台（Loihi 2、Speck、SpiNNaker 2、Xylo）。
它的定位很清醒——**只描述"网络是什么"，故意不管"怎么跑"**。NIR 论文原文：
节点只是*声明*动力学，不负责*执行*它们。

问题是这个清醒的边界有一个没人写下来的推论：

> **文件里没有 T。** 于是"这个图跑一次 = 一步"还是"跑 T 次"，只能靠生产者
> 和消费者之间的口头约定。

实测 NIR 1.0.8 的节点表（`dir(nir)`，不是文档）：

```
有  Affine Linear Conv1d Conv2d SumPool2d AvgPool2d Flatten
    LIF LI IF I CubaLIF CubaLI Delay Scale Threshold Input Output
没有 Elementwise / Concatenate          ← 残差 add 没有原语
没有 MaxPool2d                          ← 只能退化成 Avg/Sum
没有 AdaptiveThreshold / ALIF           ← 自适应阈值没有对应
没有 Loop / Scan / Time                 ← 时间维没有对应
```

这五行"没有"就是本仓库 `deploy` 那套拦截清单的**上游成因**：10 个示例模型 5 个
被拦，拦的理由全部命中上面某一行。

### ④ 编译：这一格已经被人占了，而且做得比我们好

必须说清楚，否则后面所有话都不可信。

| 工作 | 从哪进 | 出到哪 | 覆盖 |
|---|---|---|---|
| **SNN-MLIR**（INTERA，arXiv 2606.09213，Apache-2.0+LLVM，已上 PyPI） | `.nir` | MLIR → linalg/arith → 自包含 C11 | Linear/Affine/Conv1d/Conv2d/SumPool/AvgPool/Flatten/LIF/LI/IF/CubaLIF/CubaLI，float 与 i8/i32 同一套 op，自动插入量化 scale 重标定 |
| **SNAX-MLIR**（KULeuven-MICAS） | xdsl 方言 | RISC-V 集群 + verilator 模型 | 面向 GEMM，不为时序回路设计 |
| IREE / Lava / 各厂 SDK | 各自格式 | 各自硬件 | 不互通 |

SNN-MLIR 的动机段落和我们想说的一字不差：*"NIR 只解决交换，不解决怎么跑；
每个后端都在从零重写推理循环。"* 它把这句话变成了一个方言加一条 lowering。

所以：**"SNN 缺一个编译层"这个判断在 2026 年已经过期。** 缺口往后移了一格。

### ⑤ 硬件与保真：数字有，工具没有

| 平台 | 谁在维护 | 实测数字（来自公开论文，非本仓库） |
|---|---|---|
| SpiNNaker 2 | TU Dresden / `py-spinnaker2` | NIR 部署 DVS-Gesture **94.13%**（QAT），PTQ 95.07→94.0，权重 0.17MB→0.04MB，**0.765 mJ/帧** |
| Loihi 2 | Intel Lava | 需 INRC 资格 |
| Speck / Xylo | SynSense（Rockpool/Sinabs） | 商用，工具链私有 |

这一格最诚实的一句话来自 NIR 论文自己：他们的第三个实验（循环 SNN）跨平台
"**显著不一致**"，并把原因归为三类——神经元定义不同、量化、异步硬件的非确定性。
论文的结论是 NIR 不只是一种表示，**也是一个测量工具**。

但那个测量是**论文里做过一次**（3 个模型、7 个仿真器），不是一个谁都能跑的命令。

---

## 3. 实测：一个真实 snnTorch 模型在交接处发生了什么

`snnTorch 1.0.0` + `nirtorch 2.6` + `nir 1.0.8` + `torch 2.10`，两层 LIF，
`forward` 里写 `for t in range(4)`（snnTorch 教程的形状），全部数字由
`tests/test_nir_fold.py` 复现：

| 观察 | 数字 |
|---|---|
| 内存中的 `.nir` 节点数 | **18** = 1 Input + 4×(2 Affine + 2 LIF) + 1 Output |
| 四个副本之间的边 | **0** —— 膜电位在 Python 变量里，NIR 没有边能表达它 |
| 四个副本的参数 | **逐位相同**（tau/r/v_th/v_leak 全等） |
| 写成文件再读回 | **21 个节点、4 个 Output** —— `nir.write` 给每个没有后继的节点补了一个 Output |
| 用这个文件跑一次（= 一步） | 与 PyTorch 的末步**不一致** |
| 用这个文件跑 4 次（= T 步） | 只有最后一个副本对，另外 3 个算完丢弃 |
| 计算量 | 副本数 × 步数 = **T²** 次块求值，而这个网络的成本应该是 T |

最后一行才是要害。T=4 时是 4 倍；T=100 时是 **100 倍**。对于一个全部卖点
就是"每帧毫焦"的格式，这个系数把结论反过来。

而且**没有任何一个字节说明 T 是多少**。消费者只有两个选择：跑一次（错），
或者猜 T 然后跑 T 次（对，但贵 T 倍）。现在的 `nir_sim` 连第二个都做不到——
它会报 `expected exactly one Output node, found 4`。

### 顺带暴露的两个上游缺陷（都已在本仓库做了兼容，不是修上游）

1. **nirtorch 的导出物无法序列化**：`Output.input_type = {'input': None}`，
   `nir.write` 在 h5py 里抛 `TypeError: One of data, shape or dtype must be specified`。
   生产者存不下自己的输出。
2. **nirtorch 的导出物过不了 NIR 自己的类型检查**：神经元参数写成 size-1 数组，
   而 `nir.LIF.__post_init__` 用参数形状推 `input_type`，于是一个接在 8 宽
   `Affine` 后面的 LIF 声明 `input_type=[1]`，`nir.read` 抛
   `type mismatch: fc1.output: [[8]] -> lif1.input: [[1]]`。

两个都是"算术没错、类型声明错"。而类型声明恰好是 verifier 和所有消费者读的东西。

---

## 4. 复用了什么（这一栏必须比"自研"长）

| 复用 | 用来做什么 | 我们不重写 |
|---|---|---|
| `nir` 1.0.8 | 交换格式、节点定义、HDF5 读写 | 格式本身 |
| `nirtorch` 2.6 | PyTorch → NIR 的模块映射机制 | 权重搬运 |
| `torch.export` + `hop.scan` | 图捕获与"保留循环"的载体 | 图捕获 |
| `xdsl` 0.74 | 解析与执行我们发出的 `.mlir`（SNAX 就构建在它上面） | MLIR 基础设施 |
| `mlir-opt` 通用 pass | `canonicalize` / `scf-to-cf` / `memref-to-ptr` 三步验证 | lowering 框架 |
| 上游论文的量化方法（PTQ4SNN、QAT threshold scaling） | 数值参考 | 量化算法 |
| 各厂 SDK（`targets.py` 里子进程探测） | 判断文件别人能不能读 | 后端实现 |

自己造的只有：`ir.py` 182 行 + `passes/` 275 行（占实现的 6%），以及本文档
之后新增的 `nir_fold.py`。

---

## 5. 那我们补哪一格

按"缺不缺、贵不贵、别人做没做"三列筛，只剩三件事，按成本从低到高：

### A. 时间维的归宿（已落地，`snndsl fold`）

`detect` 找出"这个图是 k 份逐位相同的副本"，`fold` 收回成一份，把 k 写进
`metadata["snndsl.timesteps"]`——**这是 `.nir` 里唯一能放 T 的地方**。
同时丢掉 T-1 个悬空副本被 `nir.write` 制造出的多余 Output。

实测：21 → 6 节点，T 步执行与 PyTorch **逐位置 differ = 0**，计算量 T² → T。
不折叠的条件也写死了：副本参数只要有一处不同就拒绝，因为那是一个真的展开网络，
不是一个循环。

### B. 交接处的语义归一（下一件该做的）

今天暴露的两个 nirtorch 缺陷、以及 `beta → (tau, r)` 里那个自由参数
（`r = tau` 还是 `r = 1`，选错则网络慢 9.5 倍且文件依然合法），本质是同一件事：
**从"某个库的某种写法"到"格式的语义"之间没有一步被任何工具负责。**

要做的是一个显式的、可检查的归一化层：每条映射规则写下来、每条都能被一个
数值差分测试钉住（转换前后跑同一段输入，脉冲轨迹必须逐位置相同），
不成立就报错而不是猜。这一层的产物是**映射表 + 差分测试**，不是代码量。

### C. 保真报告（**两档已落地**，`snndsl fidelity`）

NIR 论文把跨平台差异测了一遍，然后停在论文里。把它做成命令：
给定 `.nir` + 一段输入，输出"这台目标上跑出来的脉冲，与你在 PyTorch 里
训练的那条，逐位置差多少"。这是**唯一一个能把"能不能上硬件"从信仰变成数字**的东西，
也是这套代码里唯一有商业形状的部分。

已落地的两档都**不需要硬件**，也就是任何人都能复现的那部分：

| 扰动 | 为什么它合规却给出不同的脉冲 | 实测（两层 LIF，T=8，40 组输入） |
|---|---|---|
| int8 / int4 权重（逐通道） | 厂商做得最多的一档，也是最容易被当成替罪羊的一档 | int8 `0.63%`，int4 `4.32%`，top-1 `100% → 85%` |
| tau：欧拉 → 指数 | `v ← (1−dt/tau)v` 与 `v ← e^(−dt/tau)v` 是同一个连续常数的两种离散化，**文件里没有字段说明该用哪种** | tau=1.2 时 `1.95%`，tau=20 时 `0.03%`，差 65 倍 |

第二条是这张表里唯一无法用"少几位权重"解释的东西：同一份文件、同一组权重，
两套都声称读懂了的实现，在 tau 接近 dt 时把脉冲打散 65 倍。

基线取的是**文件本身**在 float32 下的语义，不是 PyTorch——文件才是要上板的东西，
产地到文件的差异归 `deploy --verify` 管。这条划分写进了命令的每行输出里。

还缺的一档是**真实目标后端**：Loihi 2 / Speck / SNAX 上那条轨迹。它需要硬件或
verilator，本仓库没有，所以那一档只能是"用执行配置近似"，不能叫"目标实测"。

### 明确不做

- **不注册 `snn` dialect 让别人 `require`**。方言的契约是每个 op 语义无歧义，
  而源码语义恰恰有歧义；做成方言等于先内部选定一种语义，于是又成了第 N 个实现。
- **不重写 lowering**。SNN-MLIR 已经覆盖 13 种 NIR 节点到 C11，比我们全。
- **不做 conv/pool 的 MLIR lowering**。留给 SNN-MLIR，我们只负责判断"能不能"。
- **不再给 `.snn` 加语法**。它不是语言，是 conformance 夹具。

---

## 6. "开箱即用"到底指什么

不是"装一个包就能训 SNN"——那有 snnTorch。是这个：

```bash
pip install 'snndsl[nir]'
python -m snndsl fold my_model.nir        # 把 T 份副本收回成 1 份，T 记进 metadata
python -m snndsl fidelity my_model.nir    # 换成 int8、换一种 tau 离散化，脉冲错多少位
python -m snndsl deploy my_model.snn      # 目标 × 障碍矩阵 + 与 PyTorch 的核对
```

（这里曾经写着一行 `snndsl audit`。它不存在，是写文档时先写了想要的东西。
现在这三条都能跑，`snndsl --help` 里都能查到。）

判据不是"功能多"，是**一个别人训练的模型，不改一行代码，能在这个工具里
得到一个可以据以做决定的结论**。

---

## 7. 什么时候该停

三条可证伪的线，任何一条不成立就停：

1. **`fold` 在 5 个外部模型上都能识别**（不同框架、不同写法）。现在是 1 个。
2. **`audit` 的结论被模型作者认一次**——"对，我这里确实会崩"或"不对，你搞错了"，
   两种都算数，沉默不算。
3. **保真差的数字有人用**。如果没人要这个数字，那这套东西只有学术价值。

到 2026 年底如果三条都没发生，正确做法是：保留 `deploy` 那套诊断和本文档的
实测数字，删掉编译器叙事。

---

## 8. 引用

- NIR：*Neuromorphic intermediate representation*, Nature Communications 15, 2024.
  <https://www.nature.com/articles/s41467-024-52259-9>（7 仿真器 / 4 硬件、SRNN 跨平台差异自述）
- SNN-MLIR：arXiv 2606.09213，<https://arxiv.org/html/2606.09213>；节点支持表
  <https://snn-mlir.readthedocs.io/en/latest/>
- SpiNNaker 2 + NIR 部署：arXiv 2504.06748（94.13%、0.765 mJ/帧）
- 训练显存：*Towards Lossless Memory-efficient Training of SNNs via Gradient
  Checkpointing and Spike Compression*, ICLR 2026
- 膜电位量化：*PTQ4SNN*, 2026（W4/M4，ImageNet −0.74）
- `hop.scan`：`torch._higher_order_ops.scan`，torch 2.10（私有 API，注意漂移）
