# Python 前端：用 PyTorch 写 SNN，而不是学一门新语法

写这份文档的原因：`.snn` 是一套自造语法。它的学习成本是零门槛的最大障碍，
而 SNN 用户本来就已经在 PyTorch 里写模型了。问题不是"要不要新语法"，而是
**从 PyTorch 模型到硬件时序，中间到底缺哪一步**。

先给结论，再用实测数据支撑：

> 缺的不是图捕获（`torch.export` 有了），也不是格式（NIR 有了），也不是后端
> （SNAX/SNN-MLIR 有了）。缺的是**把时间循环从一个 Python `for` 变成一个有
> 结构的循环节点，再从图里认出 SNN 语义**。这两件事没有任何现成工具做，
> 因为所有现成工具都在"时间维已经消失"之后才接手。

## 1. 时间循环是分水岭（实测）

同一个两层 LIF 网络，两种写法，同一台机器（torch 2.10），`torch.export` 之后的
FX 图规模：

| 步数 T | `for t in range(T)` | `hop.scan` |
|---:|---:|---:|
| 8 | 136 节点 | **13 节点** |
| 32 | 520 节点 | **13 节点** |
| 128 | 2,056 节点 | **13 节点** |

`for` 版本线性增长，`scan` 版本与 T 无关。SNN 常用的 T=100–1000，`for` 版本
会给出 1,600–16,000 个节点，图捕获还没结束就已经不可用了。

原因很直接：`torch.export` 把 Python 循环当成普通控制流展开（unroll），它没有
理由认为"这个循环是时间"。而在硬件上，时间是一段**重复执行的时序回路**，不是
T 份并联电路。展开后再也回不去。

`hop.scan`（`torch._higher_order_ops.scan`）是 PyTorch 官方的高阶算子，语义上
就是"carry 沿序列推进"，`torch.export` 会把它保留成一个循环节点而不是展开。
**这是整条路上唯一一个已经被上游解决的问题，而它恰好是分水岭。**

## 2. 已有工作：谁做了什么，能复用哪一段

| 工作 | 做什么 | 可以复用 | 不解决 |
|---|---|---|---|
| **snnTorch / SpikingJelly / Norse** | PyTorch 上训练 SNN，surrogate 梯度 | 模型的写法、神经元实现、数据集、训练脚本——**用户的代码一行不用改** | 没有静态图；时间循环是 Python `for` |
| **SpikingJelly `step_mode='m'`** | 时间维作为张量的一维 `[T,N,*]` | 多步前向，无状态层可整段向量化 | 有状态神经元内部仍是 `for t`；CUDA backend 是手写 kernel 融合，不走图编译 |
| **`torch.export`** | 捕获 FX 图，ATen 算子 | 图捕获、shape 传播、算子语义 | 把时间循环展开（见 §1）；不认 SNN |
| **`hop.scan`** | 保留循环结构 | **时间循环的载体** | 只是一个通用循环，不知道里面是神经元 |
| **NIRTorch** | `torch.fx` trace → NIR | `module_map` 机制、权重搬运、`to_nir`/`from_nir` | 按 `nn.Module` 逐个 trace，**看不见 `for t`**；NIR 本身没有时间维 |
| **NIR** | 硬件交换格式，四平台通用 | 下游的通用语言 | 没有循环/时序原语，时间交给执行器积分 |
| **SNN-MLIR (INTERA)** | NIR → MLIR → 裸机 C11 | 下游 lowering | 从 NIR 起步，时间维已经不存在了 |
| **SNAX-MLIR** | xdsl + RISC-V 集群 + verilator | 目标后端，能验证 | 面向 GEMM/卷积，不为时序回路设计 |
| **BrainTrace（广东省智能院, 2026-01）** | "像写 PyTorch 一样写 SNN"，Jaxpr 作 IR，自动生成在线学习代码 | 同类方向的独立验证：这条路有人走，且走通了 | 目标是**训练**（在线学习）与 JAX/XLA，不是硬件部署 |
| **Sango (Sandia)** | Python 构造 API | API 设计（层级命名、别名、拓扑排序） | 无 IR，无编译 |

一句话读这张表：**从 PyTorch 往下的每一段都有人，唯独"时间循环"这一格是空的，
而且它正好是所有人交接的地方——上游把它展开掉，下游拿到的已经没有它了。**

## 3. 缺的三件事，也就是我们该做的

### 3.1 把 `hop.scan` 降到硬件时序（纯 lowering，有现成结构）

`scan` 的 `carry` 就是膜电位 `v`，`xs` 是每个时间步的输入电流，`ys` 是脉冲输出。
映射到 MLIR 就是 `scf.for` + `iter_args`：

```
%v_final, %spikes = scf.for %t = 0 to T iter_args(%v = %v0) -> (f32) {
  %cur  = ...            // 这一层的输入
  %nv   = arith.addf %v, %cur        // 或 mulf 后再 addf
  %sp   = arith.cmpf ogt, %nv, %vth
  %v2   = arith.select %sp, ...      // reset
  scf.yield %v2
}
```

这个结构与 `snndsl/backends/mlir_export.py` 里 dense 归约的 `iter_args` 是同一个
形状，不需要新的 lowering 框架。工作量在"scan 的 region 怎么读"，不在"降到什么"。

### 3.2 从图里认出 SNN，而不要求用户打标

`NIRTorch` 的做法是让库作者写 `module_map`（"这是我的 LIF 类 → `nir.LIF`"）。
这要求每个框架各写一份映射，且**对没打标的代码无能为力**——而用户手上最多的
就是没打标的代码。

我们已有的 `recognise_lif` 是**数值探测**：给一段 `v <- c + a*v + b*i` 的算子
序列，反解出 `tau`、`r`、`v_leak`、`v_th`，并通过驱动一个"已发放"的神经元来
判定 reset 是 `subtract` 还是 `zero`。它不看类名、不看注释。

这正是 §3.2 需要的工具：**把模块级映射换成算子级识别**。代价是它只能认仿射
LIF（ALIF/自适应阈值就认不出），这个限制是明确的、可报错的。

### 3.3 语义一致性检查（已有，改接这条链）

现成的 `deploy` 已经能做"声明说 zero、实现做 subtract"这类判定。换成 Python
前端之后，这一项的价值反而更高：PyTorch 里**没有声明**，只有行为，所以矛盾会
以另一种形式出现——同一个 `LIFNode` 在训练脚本里被手工 reset、在推理脚本里被
自动 reset。这比 `.snn` 的 header/body 矛盾更难发现，也更常见。

## 4. 与 `.snn` 的关系：降级，不删除

`.snn` 现在的位置是"一个自造语法的前端"。按本文的结论，它不该再作为主入口：

- **保留**：作为可读的、diff 友好的**测试夹具**（conformance 用例写成 `.snn`
  比写成 Python 清楚），以及 `recognise_lif` 的输入源
- **降级**：不再是"你该学的东西"，README 首屏不再以它开场
- **不扩展**：不再加语法。语法每加一条，都是在一个错误的抽象上加码

## 5. 风险

**`hop.scan` 是私有 API**（`torch._higher_order_ops`，下划线前缀）。它会随
PyTorch 版本漂移，这是这条路线唯一的系统性风险。

缓解：用一层薄适配（`snndsl/frontend/scan_compat.py`）包住它，并在装 PyTorch
时校验版本；`torch.scan` 一旦进公开 API 就切过去。如果它哪天消失，退路是让
用户手写 `for t` 但**只展开一层**（把一个 block 提成函数、循环留在图上），
代价是要求用户改代码。

## 6. 判据

不要用"访谈了几个人"判断这条路对不对。用一个可跑的东西判断：

> **拿一个 SpikingJelly 写的、一行没改的 LIF 模型，`export` → 折叠 → `.mlir`
> → xdsl 解释器执行 → 与 PyTorch 逐位置比脉冲轨迹。**

`differ == 0` 就成立，不为 0 就说明 §3.1 或 §3.2 有错。这一步不需要硬件、
不需要 verilator、不需要任何人点头，一两天能做完。做完之后再谈用户。
