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

## 怎么用这张表

- 想判断能不能用：先看第 9–11、17 条属于你的网络吗，属于就别绕，绕不过去。
- 想贡献：第 13 条是价值最高且不需要外部条件的；第 9–11 条需要向上游推。
- 想质疑结论：每一条都给了命令。跑不出同样的数字，就是这张表错了，
  提 issue 时把 `snndsl doctor` 一起贴上。
