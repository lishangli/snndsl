# P0 · 第三方交叉验证报告

> **结论先说**：同一份 `.nir` 文件，在**不经过本项目任何代码**的运行时里跑出完全相同的输出。
> 1024 张真实 MNIST 测试样本，预测一致率 **100%**，输出最大差 **0.0**。
>
> 同时，这次验证**推翻了一个我们自己的说法**：卷积 SNN 无法用当前 NIR 标准表达。
> 之前"能导出 NIR"对 CIFAR 网络是不成立的。现在它是一条明确的 `BLOCK`。

---

## 1. 为什么必须做这件事

在此之前，最有力的一句话是"zero reset 下逐位一致"。它有一个漏洞：
**那个"参考解释器"是我们自己按规范写的**。用自己写的解释器证明自己的导出正确，
对外部工程师来说仍然是"你们自己说自己对"。

这件事没法靠更仔细的自我检查解决。必须要有一个我们的代码完全不参与的环节。

## 2. 谁提供了什么

| 环节 | 由谁实现 | 版本 |
|---|---|---|
| 读 `.nir` 文件 | `nir`（neuromorphs 官方） | 1.0.8 |
| 把图变成可执行模块 | `nirtorch.load`（neuromorphs/NIRTorch） | 2.6 |
| 线性层 / 卷积 / 池化 / Flatten | NIRTorch 的 `DEFAULT_MAP` | 2.6 |
| **神经元（积分、阈值、复位）** | `snnTorch` 的 `snn.Leaky` | 1.0.0 |
| 参数换算 | **本项目** | — |

本项目只贡献一件事：把 NIR 的神经元参数翻译成 snnTorch 的参数。

```
NIR  :  tau dv/dt = (v_leak - v) + r*i
离散 :  v[k+1] = (1 - dt/tau) v[k] + (dt/tau)(v_leak + r i[k])
映射 :  beta = 1 - dt/tau          → snnTorch 的衰减
        驱动 = (dt/tau)(v_leak + r*i) → snnTorch 的输入
```

这份映射会随结果一起打印，可以逐项核对：

```
h1_1   beta=0.5  th=1.0  gain=1.0  offset=0.0
y_1    beta=0.5  th=1.0  gain=1.0  offset=0.0
```

## 3. 结果

```
$ python3 examples/cross_validate.py
--- third-party cross-validation ---
  artifact      : build_hw/mnist_hw.nir
  graph engine  : nirtorch.load
  layers        : nirtorch DEFAULT_MAP
  neuron        : snnTorch snn.Leaky
  steps         : 8
  samples       : 1024

  accuracy ours        : 94.34%
  accuracy third-party : 94.34%
  prediction agreement : 100.00%
  max |output diff|    : 0.000e+00
  verdict              : CONSISTENT
```

`max |output diff| = 0.0` 是逐元素相等，不是"误差很小"。
两个实现在 1024 个样本、每个 8 个时间步上产生的脉冲计数完全相同。

产物：`build_hw/cross_validate.json`（机器可读，含映射表与后端版本）。

## 4. 这次验证抓到了 6 个问题

其中最值得说的不是"验证通过"，而是**验证过程中被揪出来的东西**。

### 我们自己的（4 个）

| # | 问题 | 后果 | 现在 |
|---|---|---|---|
| 1 | 训练结束自动导出的 `.nir` **不带权重**（全零） | 文件是空壳，且任何检查都会"通过"——零对零 | `runner` 训练后带权重重导，`export_nir(strict=True)` 拒绝空壳 |
| 2 | `deploy --verify` 验证的是**内存图**，不是磁盘文件 | 上面那个 bug 因此长期不可见 | 改为先写盘、再**从盘读回**验证 |
| 3 | 验证输入是 `torch.rand` 噪声 | 脉冲网络在噪声下不发射，比较的是"零对零" | 默认用真实数据集；输出恒定则判 `INCONCLUSIVE` |
| 4 | 卷积导出时 `input_shape` 传了 `(C,H,W)`，NIR 要 `(H,W)` | CIFAR 网络**从来没能导出过**，一导出就索引越界 | 修好；并且发现更深的阻断（见第 5 节） |

第 1、2 条是同一类错误的两个面：**验证的对象不是交付的对象**。
这类错误特别危险，因为所有检查都会绿。

### 第三方的（2 个）

**① NIRTorch 2.6 的 `DEFAULT_MAP` 丢失所有 bias**

```python
# nirtorch/nir_interpreter.py:14
module.weight.data = torch.from_numpy(linear.weight)
module.weight.bias = torch.from_numpy(linear.bias)   # ← 不是 module.bias
```

第二行写到了一个临时属性上，`module.bias` 保持随机初始化。**不报错，静默失效。**

实测影响：预测一致率从 100% 掉到 98.73%。我们因此接管了 `Affine` 的映射，
并在结果里写明原因。

**② `nir_to_torch`（官方推荐的替代 API）接不了 snnTorch 神经元**

它把神经元的 `(spike, membrane)` 元组直接接到下游 Linear，报
`linear(): argument 'input' must be Tensor, not tuple`。
`load` 里有专门的 snnTorch 分支，`nir_to_torch` 里没有——
即"新 API 暂时不支持它自己文档推荐的神经元后端"。我们用的是 `load`，并在代码里注明。

## 5. 一个必须说清楚的边界：卷积 SNN 跨不过去

修完卷积导出的形状 bug 之后，碰到了更硬的东西：

```
BLOCK c1$1  this neuron receives a (32, 32, 32) activation (from 'conv'),
            but a NIR neuron layer carries a 1-D per-neuron parameter array
            and NIR 1.0.8 has no Reshape to restore the spatial dims
```

验证过程（最小复现）：

```
conv.output      = [32, 32, 32]
LIF(32768).input = [32768]          # 由参数数组长度决定
→ NIRGraph 构造失败：type mismatch
```

NIR 1.0.8 的节点集合是固定的：
`Affine, AvgPool2d, Conv1d, Conv2d, CubaLI, CubaLIF, Delay, Flatten, IF, LI, LIF,
Linear, Scale, SumPool2d, Threshold`。

神经元层（LIF/LI/IF/CubaLIF）的类型由**一维逐神经元参数数组**决定；
而 `Flatten` 只有"展平"一个方向，**没有 Unflatten / Reshape** 能把 `(C,H,W)` 还原回来。
所以 `conv → neuron → conv` 这条链在当前标准里无法表达。

这不是我们的实现缺陷，是标准的能力边界。已经做的三件事：

1. `analyze()` 对每个多维神经元报 `BLOCK`，`cifar_hw.snn` 现在是 `NOT DEPLOYABLE`；
2. 导出直接抛 `NIRShapeError`，而不是写出一个 `nir.read()` 读不回来的文件；
3. 这条边界写进 README 的局限章节。

**对产品的影响是明确的**：今天能无损部署的是**全连接 SNN**。
卷积 SNN 需要 NIR 侧的多维神经元扩展，或者目标编译器自己的方言。
这是一个可以和上游一起提的提案，不是一句"以后会支持"。

## 6. 还剩下的

| 项 | 状态 |
|---|---|
| 真机验证 | **未做**。第三方验证证明的是"规范语义下一致"，不等于"在 Xylo / Loihi 上一致" |
| 卷积 SNN | 标准层阻断，已有明确 BLOCK 与原因 |
| `subtract` reset | 不可无损表达，仍是 `WARN` |
| 第三方数量 | 1 个（NIRTorch + snnTorch 组合）。再加一个（Nengo 或 Lava）会更强 |

## 7. 复现

```bash
# 1. 训练（或直接用仓库里的 build_hw/mnist_hw.pt）
python3 -m snndsl run examples/mnist_hw.snn --out build_hw --epochs 4

# 2. 我方解释器 vs 训练实现
python3 -m snndsl deploy examples/mnist_hw.snn --verify --weights build_hw/mnist_hw.pt

# 3. 第三方运行时 vs 训练实现
python3 examples/cross_validate.py --out build_hw/cross_validate.json

# 4. 卷积网络为什么过不去
python3 -m snndsl deploy examples/cifar_hw.snn
```
