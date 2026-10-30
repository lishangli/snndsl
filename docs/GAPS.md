# 缺口登记表

这个项目的定位是补生态里没人管的那一层，所以**把自己还差什么写清楚**是它的
一部分，不是附录。下表每一条都要求三件事：状态、归属（我们的 bug / 上游的
bug / 标准的空白）、以及一条能当场验证的命令。

判断归属很重要。这个项目已经两次把自己的缺陷写成"标准不支持"和"上游没有"，
两次都看上去很专业。所以下面的表里，"我们的"占多数并不丢人，
**把"我们的"写成别的才是**。

## 一览

| # | 缺口 | 归属 | 状态 |
|---|---|---|---|
| 1 | 卷积 SNN 无法导出 | **我们的** | 已关闭 · `max\|Δ\|=0.0` |
| 2 | `avg` 池化被静默导成求和 | **我们的** | 已关闭 · 现为 blocker |
| 3 | `make_map` 把字典当函数调用 | **我们的** | 已关闭 · 卷积图可加载 |
| 4 | `AvgPool2d` 映射成求和池化 | **上游** nirtorch | 已绕开 · 待上游修 |
| 5 | 逐样本探测残留膜电位 | **我们的** | 已关闭 · `max\|Δ\|=0.0` |
| 6 | 编码增益没折进 NIR 驱动 | **我们的** | 已关闭 |
| 7 | `_const` 静默取首元素 | **我们的** | 已关闭 · 现在抛错 |
| 8 | `nir` 1.0.8 写 `v_reset`、1.0.4 不认 | **上游** nir | 已降级 · 语义记在 metadata |
| 9 | 最大池化 | **标准** 无 `MaxPool2d` | 开着 · 实测阻断 |
| 10 | 残差/多分支合并 | **标准** 无 `Add`/`Concat` | 开着 · 实测阻断 |
| 11 | ALIF 导出成积分器 | **标准** 无自适应阈值 | 开着 · top-1 一致率 0.500 |
| 12 | `tau` 约定（exp vs Euler） | **标准** 未写死 | 记录在案 · 非饱和区 0 处不同 |
| 13 | 第二个独立后端（Nengo/Lava） | **我们的** | 未开始 |
| 14 | 真机 / 厂商仿真器 | **外部** | 未做 · 只有 Python 神经元 |
| 15 | 12 场用户访谈 | **外部** | 未做 · 工具包已备 |
| 16 | `build` 要 torch 才能写文件 | **我们的** | 已关闭 · numpy-only 下通过 |
| 17 | `nir_sim` 拿 torch 当张量库 | **我们的** | 开着 · `--verify` 需要 PyTorch |
| 18 | `[full]` 缺 nirtorch / snntorch，第三方对照静默跳过 | **我们的** | 已关闭 · CI 里真跑 |
| 19 | `conv` / `pool` / 残差合并没有 MLIR lowering | **我们的** | 开着 · 报 `MLIRUnsupported`，不近似 |
| 20 | 时间维在交接口没有归宿 | **上游** nir + **我们的**兼容层 | 兼容层已关（`fold`）；导出侧也已写 T |
| 21 | `beta → (tau, r)` 有一个自由参数 | **我们的** | 已关（本仓库转换器）；对外部转换器开着 |
| 22 | 只能训练内置的四个数据集 | **我们的** | 已关闭 · 文件夹 / `.npz`，未知名字报错 |
| 23 | Python 前端表达不了 zero reset | **我们的** | 已关闭 · `LIF(reset=...)`，且 `deploy` 会抓头体不一致 |
| 24 | 语法接受 `decode`，后端没有 lowering | **我们的** | 报错已可读；**功能仍未实现** |
| 25 | `nn.Module` 被当成 `snndsl.Net` 收下 | **我们的** | 已关闭 · 按类型判断，报 `exposes SNN, not snndsl.Net` |
| 26 | 同一变量声明两次 `spike`，第一条静默失效 | **我们的** | 已关闭 · `SemaError`，带行号 |
| 27 | `forward` 里复用同一个层不共享权重 | **我们的** | 开着 · 每次调用新建节点，权重各一份 |
| 28 | `forward` 里写不了数据相关的 `if` | **我们的** | 开着 · 图必须静态，写了会报错 |
| 29 | SNN-MLIR 从 `r` 反推 `dt`，输入增益≠1 时静默改动力学 | **上游** snn-mlir | 已拦住 · 预检 `input_gain_baked_into_r` |
| 30 | 终端节点按字符串 `input`/`output` 找 | **上游** snn-mlir | 已绕开 · `export --target snn-mlir` |
| 31 | `snndsl export` 不传 `-o` 时写出名为 `build` 的无扩展名文件 | **我们的** | 已关闭 · 默认按输入名派生 |
| 32 | `snn-mlir run` 需要的 `snn-opt` 不在 PyPI | **外部** | 未通 · 需从 MLIR/LLVM 源码构建 |

---

## 已关闭的（这一版修掉）

### 1 · 卷积 SNN 无法导出 —— 我们的 bug

原结论写的是"NIR 神经元层是 1-D，吃不下 `(C,H,W)`，这是标准能力边界"。
实测：`nir.LIF.__post_init__` 里就是 `input_type = {"input": np.array(r.shape)}`，
参数数组什么形状，输入就是什么形状。`(C,H,W)` 合法，元素个数与展平形式相同。

真正的断点在三处，全是我们的：`nir_export` 用 `np.full(n_units, ...)` 压平参数、
`nir_sim` 把膜电位初始化成 1-D、`deploy` 以"多维激活"为由直接判 BLOCK。

```bash
snndsl run    examples/mnist_conv.snn --epochs 2 -o build_conv   # 约 3 分钟 CPU
snndsl deploy examples/mnist_conv.snn --weights build_conv/mnist_conv.pt --verify --data mnist
#   mean |error| 0.000e+00   max |error| 0.000e+00   top-1 agreement 1.000
#   nirtorch + snnTorch  ok   max|Δ|=0.00e+00  top1=1.000
```

### 2 · `avg` 池化被静默导成求和 —— 我们的 bug

`kind` 落进了 `else` 分支变成 `SumPool2d`。2×2 求和把驱动放大 4 倍：
首层发放率 0.02 → 0.37，输出偏差 7.6 倍，而 `deploy` 当时还报 `DEPLOYABLE`。

```bash
snndsl deploy examples/cifar_hw.snn | head -3
#   BLOCK ... max-pool —— NIR 1.0.x 只有 AvgPool2d 和 SumPool2d —— NIR 1.0.x 只有 AvgPool2d 和 SumPool2d
```

### 3 · `make_map` 把 `DEFAULT_MAP` 当函数调用 —— 我们的 bug

`DEFAULT_MAP` 是以节点类型为键的字典。当函数调用会抛 `TypeError`，
被 `except Exception` 吞掉后改写成 `NotImplementedError(kind)`，
于是任何没被单独处理的算子都长得像上游缺失。

```bash
python -m pytest tests/test_bridge.py -q      # 5 passed
```

### 5 · 逐样本探测残留膜电位 —— 我们的 bug

同一个 `net` 连续喂多个样本，snnTorch 的 `mem` 会被带到下一个样本。
症状是 `max|Δ| = 0.125`，恰好是 8 个时间步里差 1 个脉冲，最容易被当成阈值抖动。
`reset_states()` 在每个样本前清零，之后逐层 75,856 个脉冲位置 0 处不同。

### 6 · 编码增益没折进 NIR 驱动 —— 我们的 bug

`rate` 编码是恒定电流 `x * gain`，编码器没有 NIR 节点，所以必须折进驱动量。
不折的话 `gain=3` 的网被拿去和未缩放的参考比，报出 3 倍假偏差。
驱动量和参考输入必须是两个变量——改这一处时先写出了 9 倍的 bug，
因为把 `x * gain` 写回了 `x` 本身，`module(x)` 又乘了一次。

### 7 · `_const` 静默取 `arr[0]` —— 我们的 bug

多维参数数组只取第一个值。当前所有导出器写的都是同一个常数的副本，所以结果对；
一旦出现真正的逐神经元参数就会被静默平均掉。现在非全同时抛 `ValueError`。

### 8 · `v_reset` 的方言差 —— 上游

`nir` 1.0.8 写 `v_reset` 字段，1.0.4 直接
`unexpected keyword argument 'v_reset'`。而 SynSense 的两个包把 `nir` 钉在 1.0.4。
处理方式：复位为 0 时不写该字段（`nir` 自身零填充兜底，语义不变），
真实语义记在 `metadata["snndsl_reset"]`，另附 `_downgrade_dialect()`。

```bash
python -m snndsl.cli export examples/mnist_hw.snn -o /tmp/x.nir   # 写盘即回读自证
```

### 18 · `[full]` 缺 nirtorch / snntorch —— 我们的 bug

这是"自证"和"他证"之间最容易糊弄过去的一条。项目对外宣称第三方后端必须复现我们的
每一个脉冲（`tests/test_export_integrity.py` 也这么断言），而 `pip install -e ".[full]"`
装不到 `nirtorch` 和 `snntorch`——
探针于是返回 `not-installed`，测试跳过，一切看起来正常。**一个从没跑过的对照，
和一次通过的对照，在输出里长得一样。**

CI 第一次真正跑起来时暴露了它：四个 full job 全红，而失败信息只有一句
`not installed in this interpreter`，缺的是哪个包一个字没提。两处都改了：

- `[full]` 现在带 `nirtorch>=2.0` 和 `snntorch>=0.9`（都不进 `[nir]`，那个 extra
  是 numpy-only job 用来证明核心不依赖 torch 的）；
- `_explain()` 把 `No module named 'x'` 里的 `x` 带进结论。

```bash
python -m pytest tests/test_targets.py -q -k third_party   # 现在是真跑，不是 skip
```

### 16 · `build` 要 torch 才能写文件 —— 我们的 bug

产物导出函数住在 `snndsl/runner.py`，那个模块顶层 `import torch`。结果是"写三个
文本文件"这件事，要求机器先装 800 MB 的框架。第一次在空 venv 里手工复现 CI 定义的
job 就撞上了：`ModuleNotFoundError: No module named 'torch'`，退出码 1，一个产物都没有。
CI 一直没跑起来，所以这个 bug 一直没暴露。

修法是把导出挪进 `snndsl/artifacts.py`（不 import 任何后端），`runner` 保留再导出。
同一轮修掉配套的两个坑：缺后端时给 `error[backend]` 加安装建议而不是 traceback；
缺 `nir` 时不再留下一个没人解释的 `.nir.json`。

```bash
python -m venv /tmp/v && /tmp/v/bin/pip install numpy . \
  && /tmp/v/bin/snndsl build examples/mnist_mlp.snn -o /tmp/b   # 退出码 0
```

---

## 仍开着的

### 9 · 最大池化 —— 标准空白

NIR 1.0 的节点全集里没有 `MaxPool2d`：

```
Affine, AvgPool2d, Conv1d, Conv2d, CubaLI, CubaLIF, Delay, Flatten, I, IF,
Input, LI, LIF, Linear, Output, Scale, SumPool2d, Threshold
```

求和池化和平均池化都有节点，最大值没有。这不是"差不多"的问题：
2×2 最大值和 2×2 求和不是同一个函数，导成哪个都是换模型。所以 `deploy`
对它报 BLOCK，而不是 WARN。

```bash
snndsl deploy examples/cifar_hw.snn | grep BLOCK
```

**出路**：向 NIR 提 `MaxPool2d` 节点（Issue 3 里已有一节），或在前端把 max pool
换成 avg 并重新训练——`examples/cifar_hw_avg.snn` 就是后者，逐位一致。

### 10 · 残差 / 多分支合并 —— 标准空白

NIR 没有 `Add`、`Concat`、`Reshape`。两个分支在网络里汇合这件事在标准内无法表达，
所以 `examples/residual_snn.snn` 没有 merge 原语可用。

```bash
snndsl deploy examples/residual_snn.snn | grep -i "merge\|BLOCK"
```

### 11 · ALIF —— 标准只能表达成积分器

自适应阈值（`v_th` 随时间涨落）在 NIR 1.0 里没有表示。导出时只能写成固定阈值，
这不是近似而是换了个模型。

```bash
snndsl run    examples/adaptive_lif.snn --epochs 2 -o build_alif
snndsl deploy examples/adaptive_lif.snn --weights build_alif/adaptive_lif.pt --verify --data mnist
#   BLOCK h1$1 / h2$1   max |error| 1.000e+00   top-1 agreement 0.500
```

两个指标都要看：

- `deploy --verify` 报的是**聚合后**的读数（8 个样本、每样本 8 步），
  `max|Δ| = 1.0`、top-1 一致率 0.500——一半的样本连类别都变了；
- 逐时间步的脉冲一致率是 **36.38%**，方法见
  [`P0_DEPLOY_MATRIX.md`](P0_DEPLOY_MATRIX.md)（准确率 94.53% → 35.40%）。

两个数字都不是"误差"，是"换了个网络"。所以它是 BLOCK。

### 12 · `tau` 的两种约定 —— 规范未写死

`tau dv/dt = ...` 在离散化时是 `1 - dt/tau`（Euler）还是 `exp(-dt/tau)`，
规范没规定。这件事的实测结论比预想的窄：**非饱和区逐脉冲 0 处不同**
（524,288 个位置），差异集中在两条边界上——多脉冲（一步发多枚，NIR 未禁止、
我们的 LIF 不发）和 `tau` 很小的档位。细节与实测表在
[`SEMANTICS.md`](SEMANTICS.md)，对规范的提法在 Issue 3。

### 13 · 第二个独立后端 —— 我们的

现在"这是编译器不是运行时"这个说法，只有一个方向的证据（nirtorch）。
Nengo 或 Lava 会让它变成两个方向。未开始。

### 14 · 真机 / 厂商仿真器 —— 外部

所有对照用的都是厂商的 **Python 神经元**，不是硅片。硅片上还有位宽、定点化、
片上存储约束，这些一条都没测。这是本项目最硬的天花板，靠本地工作无法突破。

### 15 · 12 场用户访谈 —— 外部

`docs/INTERVIEW_KIT.md` 里是问题清单和证伪条件：

> 12 人里若不足 4 人把"部署一致性"列为前三大痛点 → 方向需要重估。

这一项不能由代码完成，也不能由写代码的人代答。

### 17 · `nir_sim` 拿 torch 当张量库 —— 我们的

参考解释器是"最该被审计的一段"，但它 `import torch` 只是为了 float32 张量。
后果：`snndsl deploy --verify` 需要整个 PyTorch——而它验证的命题恰恰是
"这个网络能不能离开 PyTorch"。说得出理由（torch 保证 float32 逐位复现，
换成 numpy 得重新证明一遍），但依赖面确实比必要的大。

改成 numpy 是独立的一项工作：会动 439 行数值代码，且必须保持 `max|Δ|=0.0`
与 75,856 个脉冲位置零差异不变。先记账，不混进这一版。

状态：**开着**。在无 torch 的机器上 `snndsl doctor` 会直接说：

```
what works right now
  verify against NIR : no
```

---

### 19 · MLIR 后端只有全连接 —— 我们的

`snndsl build` 现在产出 `.mlir`，但只覆盖 `input / encode / flatten / dense /
neuron / output`。`conv`、`pool`、残差 `add` 会 raise `MLIRUnsupported`。

这不是没来得及写，是有意不写：conv 和 pool 需要 linalg 的索引映射（`im2col`
或显式 `affine_map`），而残差合并需要一个 NIR 里也不存在的 add 原语。这三样
如果写错，产出的文件**能解析、能执行、数字是错的**——正是这个项目存在的理由
的反面。与其近似，不如报错，让下游（SNAX 的 `snaxc` 就是干这个的）接手。

代价说清楚：卷积 SNN 今天只能走 NIR，走不了 MLIR。CIFAR 那个例子就是这种情况，
`build` 会在 artifacts 下面打一行

```
note: no .mlir file was written - 'conv' has no MLIR lowering in this backend: ...
```

状态：**开着**，且短期不打算关。什么时候关：等有人真的需要 `.snn → MLIR → 芯片`
这条路，且能给出一条能验证的 conv 索引映射——那时候按测试驱动写，而不是先猜。

---

### 20 · 时间维在交接口没有归宿 —— 上游 + 我们的兼容层

`snnTorch` 的教程写法（`forward` 里 `for t in range(T)`）经 `nirtorch` 导出后，
`.nir` 里是 **T 份彼此没有边的副本**：膜电位在 Python 变量里，NIR 没有边能表达
它。副本逐位相同，于是只有最后一个算的是对的，另外 T-1 个算完丢弃——成本是
**T²**，而文件里**没有一个字节说明 T 是多少**。

复现：`python -m pytest tests/test_nir_fold.py -q`（6 条，全部对照 PyTorch 逐位置）。

我们的处理：`snndsl fold` 检测重复、收回成一份、把 T 写进
`metadata["snndsl.timesteps"]`，并丢掉 T-1 个悬空副本被 `nir.write` 制造出的
多余 `Output`。实测 21 → 6 节点，T 步执行与 PyTorch **differ = 0**。

状态：**兼容层已关，根因开着**。根因是 NIR 没有时间维，属于格式设计选择，
只能靠上游加一个"这是循环体"的声明；在那之前，`metadata` 是唯一能放 T 的地方。

同一件事的另一半曾经是我们自己的 bug：`snndsl export` 生成的图本来就是单步的，
却**没有把 T 写进去**，于是连 `snndsl fidelity` 都得让用户手工 `--steps`。现在导出
同时写 `snndsl.timesteps` / `snndsl.dt` / `snndsl.program`，`fidelity` 直接可读。
（`tests/test_custom_data.py::test_export_records_the_step_count` 守着它。）

同一段里暴露的两个上游缺陷（都已在我们这一侧兼容，未修上游）：

- `nirtorch` 2.6 的 `Output.input_type = {'input': None}` 让 `nir.write` 在 h5py 里
  抛 `TypeError: One of data, shape or dtype must be specified`——生产者存不下自己的输出。
- `nirtorch` 把神经元参数写成 size-1 数组，而 `nir.LIF.__post_init__` 用参数形状推
  `input_type`，于是一个接在 8 宽 `Affine` 后的 LIF 声明 `input_type=[1]`，
  `nir.read` 抛 `type mismatch: fc1.output: [[8]] -> lif1.input: [[1]]`。

两个都是"算术没错、类型声明错"，而类型声明正是 verifier 和所有消费者读的东西。

---

### 21 · `beta → (tau, r)` 有一个自由参数，选错则网络慢 9.5 倍 —— 我们的

snnTorch 的 `Leaky(beta=0.9)` 积分 `v ← beta·v + i`；NIR 的 LIF 积分
`v ← (1 − dt/tau)·v + (dt/tau)·r·i`。`beta` 只定下 `tau = −1/ln(beta)`，**`r` 是自由的**：
写 `r = 1` 得到的是慢 9.5 倍的同一个网络，文件合法、类型正确、数值全错。

这一条是写 `tests/test_nir_fold.py` 时先写错、被 `differ` 抓到才发现的
（第一版 `r = 1.0`，折叠后 `differ = 1.0`；改成 `r = tau` 后 `differ = 0`）。

状态：**在我们自己的转换里已关**；对所有外部转换器开着——没有任何工具检查这件事。

---

### 22 · 只能训练内置的四个数据集 —— 我们的 bug

`train:` 块的 `data =` 只认 `mnist` / `fashion_mnist` / `cifar10` / `synthetic`，
其他任何名字都**静默落到合成数据上**：4000 条可分的高斯团，训练能收敛，
`test_acc` 是个漂亮的数字，跟用户的数据没有任何关系。

这是本项目最不该犯的那类错——"文件合法、数字好看、语义全错"，而且发生在
第一次使用时，用户还没有能力分辨。

复现（旧行为已不可复现，改为用测试锁定新行为）：
`python -m pytest tests/test_custom_data.py -q`（6 条，最后两条分别验证
"陌生名字必须抛错"和"文件夹 / `.npz` 真的是那份数据"）。

状态：**已关闭**。`data =` 现在接受图片文件夹（`<类别>/*.png`，可再分 `train/`、
`test/`）与 `.npz`（`x_train`/`y_train`），图片按 `input` 声明的形状自动 resize，
uint8 按 1/255 归一；**不认识的名字直接抛 `BackendError`**，并给出三种合法写法。
实测 120 张 3 分类 png、12 epoch、1.0 秒到 95.83%（`docs/GETTING_STARTED.md` 第 3 步）。

---

### 23 · Python 前端写不出 zero reset —— 我们的 bug

`.snn` 里两种复位都能写：

```
v = v - s * v_th      # subtract：NIR 只能近似成 v_reset=0，强驱动下会漂
v = v * (1 - s)       # zero：NIR 有对应字段，逐位一致
```

而 Python 前端的 `LIF()` / `IF()` / `ALIF()` 把 body 写死成 subtract。想写 zero 只剩
`custom()`——自己拼字符串，`deploy` 之前不会提醒你。于是出现这种组合：

```python
net = snn.Net("m", reset="zero")          # 头说 zero
net.neuron(net.dense(x, 256), snn.LIF())  # 身体做 subtract
```

`deploy` 会正确地拦下来（`reset-mismatch`），但用户按提示改 header 改不动任何东西，
因为**真正跑的是 body**——这正是第 1 条那个坑的 Python 版。

状态：**已关闭**。三个工厂都加了 `reset="subtract" | "zero"`，它同时决定 body 的最后
一行和 `NeuronSpec.reset`，两者不可能再分叉：

```python
snn.LIF(reset="zero").body[-1]     # 'v = v * (1 - s)'
snn.LIF(reset="membrane")          # SNNError，不猜
```

`examples/mnist_hw_py.py` 用的就是它，`snndsl deploy` 给出 `[DEPLOYABLE]`，
和 `.snn` 版结论一致。

---

### 24 · `decode` 能解析、能建图、然后崩在 codegen —— 我们的 bug

`decode` 出现在四个地方：`lexer.py` 的关键字表、`sema.py` 的 `_op_decode`、
`ir.py` 的 `IO_OPS`、`serialize.py` 的 `_REBIND_OPS`。唯独 `torch_codegen.py`
里没有 `_op_decode`。于是这段程序解析通过、语义检查通过、pass 跑完，最后在
生成源码时抛：

```console
AttributeError: 'TorchCodeGen' object has no attribute '_op_decode'.
Did you mean: '_op_encode'?
```

注意报的是什么：**一个指向我们内部类名的 AttributeError**，外加一句"你是不是想写
`_op_encode`"。用户手里拿着一份语法合法的 `.snn`，得到的是这个。而 `snndsl deploy`
本该在训练之前就拦住它。

顺带发现这里**没有任何测试覆盖 `decode`**：`examples/` 和 `tests/` 里零处出现，
所以它从写下的那天起就没被跑过。

状态：**报错已关闭，功能仍然开着**。`_emit_node` 现在查 handler 是否存在，
不存在就抛 `SNNError`，给一句话和去处：

```console
$ snndsl deploy dec.snn
dec.snn: error[general]: 'decode' has no PyTorch lowering
    hint: the .snn grammar accepts it, the backend does not implement it
```

`tests/test_compiler.py::TestUnsupportedOp` 两条锁住它（含"不许是 AttributeError"）。
但**这只是把崩溃换成一句诚实的拒绝**——`decode` 想要的语义（把窗口内的脉冲读回
模拟量）仍然没实现。真要做，得让 codegen 带上跨时间步的窗口状态，并且 `nir_export`
不能再把 `decode` 当恒等（`nir_export.py:210` 现在就在 skip 它）。那是一次功能开发，
不是一次修补，所以先留着这个"明确拒绝"，而不是假装支持。

---

### 25 · 一个普通 `nn.Module` 被当成自家网络收下 —— 我们的 bug

`compile_python_file` 判断"这个 `.py` 交出来的是不是网络"，用的是：

```python
if net is None or not hasattr(net, "compile"):
```

而 `torch.nn.Module` **自 2.0 起也有 `compile`**（`torch.compile` 的入口）。所以一个
从 ANN 过来的用户，写完 `class SNN(nn.Module)`、实例命名成 `net`，交给工具，会通过
检查，然后：

```console
$ snndsl deploy std_snn.py
TypeError: compile() got an unexpected keyword argument 'passes'
  torch/nn/modules/module.py:3046 in compile
    self._compiled_call_impl = torch.compile(self._call_impl, *args, **kwargs)
```

**更糟的是不带 kwargs 的路径**（`compile_python_file(path)`）：

```console
>>> from snndsl.pipeline import compile_python_file
>>> r = compile_python_file("std_snn.py")
>>> repr(r)
'None'
```

`Module.compile()` 是原地生效、返回 `None` 的，于是"成功编译"了一个网络，产物是
`None`。没有异常，没有警告，直到下游某处取 `r.graph` 才炸——或者某个 `if r:` 直接
静默跳过。这正是本项目最不该犯的那类错：**没报错，所以看起来对了**。

状态：**已关闭**。改成按类型判断，并说清它看到了什么：

```console
$ snndsl deploy std_snn.py
std_snn.py: error[general]: std_snn.py exposes SNN, not snndsl.Net
    hint: this tool does not read nn.Module directly -- build the graph with
          snndsl.Net, or return one from build(); see docs/PYTHON_FRONTEND.md §0
```

`tests/test_compiler.py::TestNotAnSnndslNet` 三条：类名出现在报错里、不许静默返回
`None`、不许漏出 torch 的 `TypeError`。

这个 bug 的根源比"漏一个判断"更深：**本工具确实读不了普通的 `nn.Module`**。
它的算子作用在 `torch.Tensor` 上，图只存在于 trace/export 的产物里；要接这条路得先
`torch.export` + `hop.scan`（`docs/PYTHON_FRONTEND.md` §1、§3.2）。

注意这条**不**意味着「用 Python 写网络不支持」：`snn.Module` 的 `forward`
是直接支持的（`docs/PYTHON_FRONTEND.md` §0），区别只在于它的层调用返回 IR 句柄
而不是 `torch.Tensor`，所以单步符号执行就能读出静态图。这个 bug 影响的是
"凭空接管一个别人写的 `nn.Module`"，那是另一件事。

---

### 26 · 两条 `spike` 指向同一个变量，第一条悄悄失效 —— 我们的 bug

写 neuron 块时忘了删旧行（或者在 LIF 上改 ALIF），会得到两条 `spike s = ...`。
语义检查一句话不说，IR 里两条都留着，生成的 `forward` 是：

```python
s = surrogate_fast_sigmoid((v) - (v_th), 4.0)          # 这行的结果没人用
s = surrogate_fast_sigmoid((v) - ((v_th * 0.5)), 4.0)  # 这行覆盖它
v = (v * (1 - s))
```

第一条阈值判断**完全无效**，而文件合法、能编译、能训练、能导出。要发现它只能去读
生成的代码。这是"写了却没跑"里最难查的一种。

状态：**已关闭**。`build_neuron_spec` 记录已声明的 spike 变量，重复直接抛：

```console
$ snndsl deploy dup.snn
dup.snn:14: error[sema]: neuron 'BAD' declares spike 's' twice
        spike s = v > v_th * 0.5
    hint: the second line silently overwrites the first; delete one, or
          emit them under different names
```

只限制 `spike`，不限制普通赋值——`v` 在一个 body 里被写多次（更新、复位）是正常的。
`tests/test_compiler.py::TestDuplicateSpike` 两条（报错文本 + 行号必须指向第二条）。

---

### 27 · `forward` 里复用同一个层，权重不共享 —— 我们的（开着）

PyTorch 里 `self.fc` 用在三处就是**共享权重**。本工具里每次调用都追加一个新节点，
codegen 按节点生成参数，于是三份独立的 weight：

```python
def forward(self, x):
    for _ in range(self.depth):
        x = self.lif(self.fc(x))       # fc / fc_2 / fc_3，三段参数
    return self.head(x)
```

```console
$ python examples/mnist_forward.py --dry-run     # DeepNet
fc      dense  [32]  <- e
fc_2    dense  [32]  <- lif
fc_3    dense  [32]  <- lif_2
```

参数量是 `depth ×` 而不是 `1 ×`。语义上是"三个不同层"，不是"一层用三次"。

状态：**开着**。没有静默出错（图是对的，只是层不同），但和 PyTorch 直觉相反，
所以记在这里。修法要给 `IRNode` 加一个 `weight_group` 字段，codegen 里让多个节点
指向同一个 `nn.Parameter`；`state_dict` 的 key 合并会连带影响 checkpoint 兼容性，
不是一行改动。退路：自己改生成的源码（把 `self.w_fc_2 = self.w_fc` 接上），
或者别复用层、直接接受 `depth ×` 参数。`snn.Net` 建图器同样没有共享权重的表达，
两边都要等 `weight_group`。

### 28 · `forward` 里写不了数据相关的分支 —— 我们的（开着）

```python
def forward(self, x):
    if x.shape[0] > 1:                 # 编译期常量判断：可以，Python 直接折叠
        ...
    if self.training:                  # 可以，self 是普通对象
        ...
```

上面两种都能写。下面这种不行：

```python
def forward(self, x):
    h = self.fc(x)
    if h.sum() > 0:                    # h 是句柄，没有数值
        h = self.fc2(h)
    return h
```

`h` 是 `Tensor` 句柄，`h.sum()` 不存在，会直接 `AttributeError`。这类依赖**数据**
的分支本来就没有静态图表示，任何图编译器都一样（`torch.export` 也要求分支可静态化）。

状态：**开着**，不打算修。报错来自 Python 本身，信息是 `'Tensor' object has no
attribute 'sum'`，能看懂但没指出"是因为图必须静态"。如果实际用起来这条困扰人，
再加一个 `Tensor.__getattr__` 给出定向提示。

### 29 · SNN-MLIR 从 `r` 反推时间步，增益不是 1 时静默改网络 —— 上游（已拦住）

`nodes/lif.py:89-90` 这样恢复一步的长度：

```python
dt = float(node.tau.flat[0] / node.r.flat[0])
decay = float(1 - (dt / node.tau.flat[0]))
```

只有当 `r == tau/dt`（神经元以增益 1 积分输入）时才成立。一个写成
`v = v*(1 - dt/tau) + 2*i` 的神经元导出 `r = 2*tau/dt`，于是后端算出
`decay = 1 - 1/r = 0.75`，而真实衰减是 `1 - dt/tau = 0.5`；输入增益 2 也一并丢掉。

我们这边的导出是**对的**（NIR 的方程确实描述那张网），错在消费者的反推。但结果一样：
`check` 说 `supported`，C 编得过，数字是错的。所以进预检：

```console
$ python examples/hw_chain.py --sweep      # adaptive_lif 那一行同理
  r = 4 but tau/dt = 2: the backend reads decay = 1 - 1/r = 0.75 instead of 0.5,
  and feeds the input at gain 1
```

状态：**上游的**，我们这边以预检拦住（`test_input_gain_rule_names_the_decay_the_backend_will_use`
断言 0.5 与 0.75 两个数）。给上游的 issue 草稿要点：`r` 在 NIR 里同时承担
"输入增益 × tau/dt"两个角色，反推时必须知道增益；建议要求 `|r - tau/dt| < eps` 否则报错，
而不是默默算一个 decay。

### 30 · 终端节点按字符串找 —— 上游（已绕开）

`snn_mlir/_graph.py:26` 写死 `TERMINALS = ("input", "output")`，而 NIR 标准用
`NIRGraph.inputs`/`.outputs`（从节点类型推出）。结果是合法的 `.nir` 被报
`Graph has no 'input' node`。

状态：**上游的**，我们这边用 `--target snn-mlir` 让名字，原名字留在
`report["terminals"]` 和文件 metadata 里。这是给一个消费者的让步，默认导出不改名。

### 31 · `export` 继承了 `build` 的 `-o` 默认值 —— 我们的（已关闭）

`snndsl export x.snn`（不传 `-o`）会写出一个**叫 `build`、没有扩展名**的文件，
旁边再掉一个 `build.report.json`。原因是 `export` 复用了 `build`/`run` 的参数组，
而那组的 `--out` 默认值是目录名 `"build"`。

状态：**已关闭**。`export` 现在单独把默认置空，按输入文件名派生 `<stem>.nir`。
这条属于"能跑、但产物在错地方"的一类，测试很难自然覆盖到——记在这里是因为
它是被 `--target` 的实测顺带撞出来的，不是被测试撞出来的。

### 32 · 最后一步要的工具链不在 PyPI —— 外部

`snn-mlir run` 需要 `snn-opt` + `mlir-opt`/`mlir-translate`/`llc`。前者是
MLIR/LLVM 源码树里的 CMake 产物：

```console
$ pip install snn-opt && python -c "import snn_opt; print(dir(snn_opt)[:4])"
['AffineSubspaceProjector', 'ConvergenceConfig', 'DykstraProjector', ...]
```

装是装上了，但不是同一个东西——那是半定规划/SOCP 求解器，只是撞名。

状态：**未通**。所以链路终点是 `build/` 目录（`network.mlir` + `main.c` + 数据头），
不是 `results.csv`。`snndsl/backends/snn_mlir.run()` 把后端的原始报错带回来，不假装跑过。
要闭合这一环只能编 LLVM，不属于本仓库范围。

---

## 怎么用这张表

- 想判断能不能用：先看第 9–11、17、19 条属于你的网络吗，属于就别绕，绕不过去。
- 想贡献：第 13 条是价值最高且不需要外部条件的；第 9–11 条需要向上游推。
- 想质疑结论：每一条都给了命令。跑不出同样的数字，就是这张表错了，
  提 issue 时把 `snndsl doctor` 一起贴上。
