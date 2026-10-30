# Python 前端：用 PyTorch 写 SNN，而不是学一门新语法

写这份文档的原因：`.snn` 是一套自造语法。它的学习成本是零门槛的最大障碍，
而 SNN 用户本来就已经在 PyTorch 里写模型了。问题不是"要不要新语法"，而是
**从 PyTorch 模型到硬件时序，中间到底缺哪一步**。

先给结论，再用实测数据支撑：

> 缺的不是图捕获（`torch.export` 有了），也不是格式（NIR 有了），也不是后端
> （SNAX/SNN-MLIR 有了）。缺的是**把时间循环从一个 Python `for` 变成一个有
> 结构的循环节点，再从图里认出 SNN 语义**。这两件事没有任何现成工具做，
> 因为所有现成工具都在"时间维已经消失"之后才接手。

## 0. 网络怎么描述：三种写法，工具全认

| 你写的东西 | 谁写 | 工具怎么处理 | 产物 |
|---|---|---|---|
| `class Net(snn.Module)` + `forward` | 人 | 单步符号执行 → IR | 同一份 |
| `snn.Net` 调用序列 | 生成代码、要精确控制 | 每个方法直接加节点 | 同一份 |
| `.snn` 文本 | 机器 | 解析 → IR | 同一份 |

三种都收口在 `finalize()`，优化 pass 和 codegen 不知道你用了哪种。

### `forward` 怎么写

`forward` 描述**一个时间步**。T 是类属性，由编译器套上去——这和 `.snn` 里
`temporal: steps = 8` 是同一件事，所以不需要在 `forward` 里写时间循环。

```python
import snndsl as snn

class MnistNet(snn.Module):
    input_shape = (1, 28, 28)
    steps = 8
    reset = "zero"

    def __init__(self):
        super().__init__()
        self.fc1  = snn.Dense(256, scale=2.0)
        self.fc2  = snn.Dense(10)
        self.lif1 = snn.LIF(tau=2.0, reset="zero")
        self.lif2 = snn.LIF(tau=2.0, reset="zero")

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        return self.lif2(self.fc2(x))
```

节点名就是属性名，打印出来能和你写的代码逐行对上（实测）：

```console
$ python examples/mnist_forward.py --dry-run
id                    op        shape             inputs
--------------------------------------------------------------
x                     input     [1, 28, 28]
e                     encode    [1, 28, 28]       x
fc1                   dense     [256]             e
lif1                  neuron    [256]             fc1
fc2                   dense     [10]              lif1
lif2                  neuron    [10]              fc2
out                   output    [10]              lif2

program   : MnistNet
time      : T=8  dt=1.0  schedule=looped
graph     : 7 nodes, 203,530 params, 1,626,112 MAC (T x step)
```

**203,530 参数**，和 `examples/mnist_hw.snn` 逐位相同；训 1 epoch 的 `test_acc`
都是 **0.9198**。`snn.Module` 把图 traced 成同一份 IR，后面没有任何分叉
（`tests/test_forward_front.py::test_forward_matches_handle_style_exactly` 锁住这一点）。

### 为什么不直接用 `torch.jit.trace`

这是"用 Python 语法描述 SNN"最自然的想法，实测行不通。同样一个 LIF cell，
时间循环写在 `forward` 里，`torch.jit.trace` 跑一遍得到的是**展开后的图**：

| T | `torch.jit.trace` 图节点数 | 本工具 `forward` 图节点数 |
|---:|---:|---:|
| 1 | 13 | 5 |
| 2 | 17 | 5 |
| 4 | 25 | 5 |
| 8 | 41 | 5 |
| 16 | 73 | 5 |
| 32 | 137 | 5 |

trace 的图 **∝ T**，本工具的图**恒定**。三个后果，每一个都是硬伤：

1. **T 被固化进图里**。想在部署时把 T 从 8 改成 16，得重新 trace 一份。
2. **"这是一个循环"这个信息丢了**。下游看到的是一串重复的算子，认不出 cell，
   没法做 `T=8 > 4 → looped` 这类决策（上面 summary 里那条 `schedule` pass 就是）。
3. **硬件时序无从对应**。NIR/SNN-MLIR 要的是"一个 cell + T 参数"，给一份 137 节点的
   平铺图，等于让后端自己再猜一遍哪 4 个节点是一步。

所以本工具**不用 trace**，而是把 `forward` 在"T=1 的符号状态"下执行一遍：
每次 `self.fc1(x)` 只追加一个 IR 节点、返回一个句柄（`Tensor` 只有 `id` 和 `shape`，
没有数据），循环体走一次就停。得到的正是那个 cell。

代价是三条限制，都写成显式报错，不会静默出错：

| 你写了 | 结果 |
|---|---|
| `if x.sum() > 0:` 数据相关分支 | 图必须是静态的，这种写不了（报错） |
| `h + 1`、`torch.relu(h)` | 报 `cannot add int to a tensor handle`，只有 tensor 句柄存在 |
| `return None` / 返回 tuple | 报 `forward returned NoneType, not a tensor handle` |

### 两种 `for`，含义不同

这是 Python 前端相对 `.snn` 的**全部价值**：`.snn` 的 `pipeline:` 是平铺的，
写不出"N 个相同 block"。在 `forward` 里这是普通循环：

```python
def forward(self, x):
    for _ in range(self.depth):      # 展开：真的生成 depth 个 block
        x = self.lif(self.fc(x))
    return self.head(x)
```

```console
$ python examples/mnist_forward.py --dry-run     # DeepNet 那一段
fc      dense     [32]              e
lif     neuron    [32]              fc
fc_2    dense     [32]              lif
lif_2   neuron    [32]              fc_2
fc_3    dense     [32]              lif_2
lif_3   neuron    [32]              fc_3
head    dense     [10]              lif_3
out     output    [10]              head
```

`fc` / `fc_2` / `fc_3` 是**三个独立节点**。注意 weight 也被复制了三份——
PyTorch 里同一个 `self.fc` 用在三处是共享权重，这里不是（`docs/GAPS.md` #27）。

另一类 `for` 是时间循环。写了不算错（trace 时被钉成一次），但**没必要写**。
危险的是把时间循环写成字面量：`steps = 8` 的网络里写 `for t in range(8)`，
会被当成"展开 8 个 block"——图能编译、能训练、数字看着也像，只是完全错了。
所以这一条是硬报错：

```console
SNNError: Typo.forward loops over range(8) -- the same number as steps
    hint: write range(self.steps) for the time loop; ... A literal range() is
          treated as block duplication
```

### `snn.Net` 是图纸，不是网络

`forward` 前端在内部就是靠 `snn.Net` 记录节点的：`self.fc1(x)` 最终落到
`Net.dense(...)`。所以下面这个底层 API 依然是一等公民——`forward` 能表达的它都能
表达，反过来不成立（`forward` 只认你在 `forward` 里写出来的东西）。
它连 `nn.Module` 都不是：
实测 `isinstance(net, nn.Module)` 为 `False`、没有 `parameters()`：

```console
>>> type(net)
<class 'snndsl.pyfront.Net'>
>>> isinstance(net, torch.nn.Module)
False
>>> hasattr(net, "parameters")
False
>>> vars(net).keys()
dict_keys(['graph', '_counts', '_used_neurons', '_closed'])
```

它只持有 `graph`——也就是 IR。`net.dense(...)` 不分配任何张量，只是在 `graph` 里
追加一个节点并返回一个句柄：

```python
def _add(self, op, inputs, shape, attrs, ...):
    # ... 检查 _closed、分配不重名的 id ...
    node = IRNode(id=nid, op=op, inputs=[t.id for t in inputs], attrs=attrs, shape=shape)
    self.graph.add(node)
    return Tensor(self, node)          # Tensor 只是 (id, shape)，不是 torch.Tensor
```

`compile()` 之后它自我封闭，再加层直接报错，因为这个角色已经用完了：

```console
>>> net.compile()
>>> net.dense(h, 5)
SNNError: this Net is already compiled; start a new one
```

`Tensor` / `Neuron` 两个类同理：`snn.LIF(...)` 返回的 `Neuron` 里，方程是**字符串**，
直到 `spec()` 才交给 `.snn` 的解析器校验（`pyfront.py:88`）。所以 Python 前端并没有
绕过语义检查，它只是换了个方式写同一份被检查的文本。

### 那"网络结构"在哪

**先说一个容易漏的点：结构不是只由 `pipeline:` 描述的。** 一份 `.snn` 里三段都在
描述结构，缺一段就描述不全：

| 段 | 描述结构的哪一部分 | 产物 | 只删掉它会怎样 |
|---|---|---|---|
| `pipeline:` | **拓扑**：节点 + 有向边 | `graph.nodes`，边就是每个节点的 `inputs` | 没有图 |
| `neuron LIF:` | **算子语义**：状态变量、膜电位方程、阈值、复位、代理梯度 | `NeuronSpec`，挂在 `graph.neurons` | 剩下线性层，不是 SNN |
| `temporal:` | **时间语义**：`steps` / `dt` / `reset` | `TemporalSpec` | 不知道跑几步、怎么复位 |

中间那块最容易被当成"参数"，它不是。`examples/adaptive_lif.snn` 一个文件里同时
放了两个神经元模型，节点上直接标着用哪个：

```console
ALIF: states=['v', 'a']  body=5 条
LIF : states=['v']       body=3 条

  h1     dense   <- ['x$1']
  h1$1   neuron  <- ['h1']        # neuron h1 -> ALIF
  h2     dense   <- ['h1$1']
  h2$1   neuron  <- ['h2']        # neuron h2 -> ALIF
  y      dense   <- ['h2$1']
  y$1    neuron  <- ['y']         # neuron y  -> LIF
```

换成 ALIF 时**拓扑一个节点都没动**，但每个神经元的状态从 `{v}` 变成 `{v, a}`，
方程从 3 条变 5 条，生成的 `forward` 多一个 `_st["a"]` 和两行 `a` 的更新。
**在 ANN 里换激活函数是改配置，在 SNN 里换神经元模型是改状态空间**——这就是为什么
它必须进结构描述，而不能藏在超参数里。

至于边：IR 里没有单独的 `edges` 列表，每个节点的 `inputs` 就是入边（`ir.py` 只存
前驱，因为 codegen 按拓扑序输出 `v_h1 = f(v_x$1)`，前驱就是变量名）：

```console
  x     input   <- []
  x$1   encode  <- ['x']
  h1    dense   <- ['x$1']
  h1$1  neuron  <- ['h1']
  y     dense   <- ['h1$1']
  y$1   neuron  <- ['y']
  out   output  <- ['y$1']
```

同一张网络有三种身，只有第二种叫"结构"：

| 层 | 是什么 | 谁读 | 有权重吗 |
|---|---|---|---|
| 源 | `.snn` 文本，或 `net.dense(x, 256)` 这串调用 | 人 | 没有 |
| **IR** | 一张 DAG（节点 + 边）+ 一张神经元方程表 | pass、codegen、NIR 导出 | 没有，只有形状 |
| 模块 | `nn.Module`，`forward` 里 `for t in range(T)` 包着这张 DAG | PyTorch | **203,530 个** |

`examples/mnist_hw_py.py` 那张图的真身，就是七行：

```console
program mnist_hw  T=8 readout=spikes
  x              input    [1, 28, 28]        <- -
  e              encode   [1, 28, 28]        <- x
  h1             dense    [256]              <- e
  a1             neuron   [256]              <- h1
  y              dense    [10]               <- a1
  ay             neuron   [10]               <- y
  out            output   [10]               <- ay
```

节点 op 不靠 `nn.Sequential` 嵌套表达——Python 前端能生成的只有 11 种：
`input / encode / dense / conv / pool / flatten / dropout / neuron / add /
concat / output`。生成的 `SnnNet` 里 `named_children()` 是**空的**，权重是裸
`nn.Parameter`：

```console
>>> m.state_dict().keys()
odict_keys(['w_h1', 'b_h1', 'w_y', 'b_y'])
```

这不是省事，是把"结构"压到唯一一处可核对的地方。边就是 `forward` 里的变量
引用，一眼能对上——下面这个三块残差网络（Python 循环生成，`.snn` 写不出）：

```python
for i in range(3):
    t = n.neuron(n.dense(h, 256, name=f"r{i}"), LIF, name=f"ar{i}")
    h = n.add(h, t, name=f"skip{i}")        # 残差
```

生成出来就是（原文，注释带着节点名）：

```python
            v_ar0 = s
            # add skip0
            v_skip0 = v_a0 + v_ar0
            # dense r1 256->256
            v_r1 = F.linear(v_skip0.reshape(batch, -1), self.w_r1, self.b_r1)
```

所以"网络结构"这个问题，在本仓库里有三个不同答案，取决于问的是哪一层：
**写的是什么**（源）、**优化后剩什么**（IR，可 `net.describe()` 打印）、
**跑的是什么**（模块，可 diff 的 PyTorch）。三种前端（`.snn` 文本、`snn.Net` 调用、`snn.Module` 的
`forward`）由同一个 `finalize()` 收口（`pipeline.py:67`），不可能在优化或代码生成阶段分叉。

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

## 4. 与 `.snn` 的关系：降级成序列化格式，不删除

`.snn` 现在的位置是"一个自造语法的前端"。按本文的结论，它不该再作为主入口：

- **保留**：作为可读的、diff 友好的**测试夹具**（conformance 用例写成 `.snn`
  比写成 Python 清楚），以及 `recognise_lif` 的输入源
- **降级**：不再是"你该学的东西"，README 首屏不再以它开场
- **不扩展**：不再加语法。语法每加一条，都是在一个错误的抽象上加码

**这一步已经落地，不是计划**。`snndsl emit` 把 IR 打印回 `.snn`：

```console
$ snndsl emit examples/mnist_mlp.snn        # 规范化后打到 stdout
$ snndsl emit examples/mnist_py.py -o x.snn # Python 建的图 → 文本
```

`tests/test_serialize.py` 对 `examples/` 下全部 10 个示例跑
`compile → emit → compile`，比对 op 序列、形状、`T`、`reset`、`readout`
（覆盖 `add` 残差、ALIF、max-pool）。所以 `.snn` 现在的定义是
**"IR 的一种文本打印"**，与 `build` 生成的 `.py` 同级：一个给人读，一个给机器跑，
都不是必须经过的前置阶段。

### `.snn` 剩下什么独占能力

实测一条，也是唯一一条：

```console
$ snndsl deploy examples/mnist_py.py
>>> 模块顶层代码被执行了 <<<          # 用户的 import/副作用全都跑了
[NOT DEPLOYABLE] readout=membrane ...
```

`deploy x.py` 必须 import 模块才能拿到 `Net` 对象，于是模块顶层的每一行都会执行；
`deploy x.snn` 只解析文本。如果模型来自别人、你只想问"它能不能上芯片"而不想运行
它，`.snn` 是两条前端里唯一能做到的。这个场景本仓库还没有用户，所以它是**保留
理由，不是卖点**——真要做，还得配一个不 import 的静态提取路径（现在没有）。

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
