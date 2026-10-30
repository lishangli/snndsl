# 实测结果

以下数字来自本仓库在 32 核 CPU（无 GPU）上的真实运行，命令与产物都可复现。
没有 GPU 集群、没有超参搜索，全部是 baseline。

## 1. MNIST MLP-SNN（`examples/mnist_mlp.snn`，spike readout）

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

### 为什么能耗只有 1.01×

因为第一层（784→256）占了 98.7% 的 MAC，而它的输入是 rate 编码的**模拟电流**，
不是脉冲，必须按 MAC 计费。这是 MLP 结构决定的，不是实现问题。
报告里的敏感性分析给出结论：

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

能耗常数取 45 nm CMOS（Horowitz ISSCC 2014：MAC=4.6 pJ，AC=0.9 pJ）：
ANN 每个突触一次 MAC；SNN 仅在输入有脉冲时累加，`AC = 发放率 × fan-in × fan-out`。

## 2. readout 的选择决定分类头质量（`examples/mnist_membrane.snn`）

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

代价是 `membrane` readout 无法无损部署到 NIR：脉冲神经元输出脉冲，不输出 reset 前的
膜电位。需要上硬件的网络请用 spike readout，或按 `examples/mnist_hw.snn` 的写法单独加
一层不发放的读出层。这条约束的实测记录在 [`SEMANTICS.md`](SEMANTICS.md)。

## 3. CIFAR-10 CNN-SNN（`examples/cifar_cnn.snn`，仅编译侧验证）

| 指标 | 数值 |
|---|---|
| 参数量 | 2,193,226 |
| 每步 MAC | 74,529,792 |
| 节点数 | 15 → 14（`fuse` 折叠 flatten） |
| 激活内存 | 534,608 → 262,144 B（**−51.0%**） |
| 编译耗时 | ~4 ms |

未在本仓库训练到收敛（CPU 上代价过高），仅用于验证卷积/池化/展平算子链与融合 pass。

卷积网络**能训练到收敛**的例子是 `examples/mnist_conv.snn`：两层卷积，3 分钟 CPU，
**98.06%**（int8 PTQ 97.94%），自己仿真与第三方运行时逐位一致。

## 4. 两种部署证据

```bash
$ python3 examples/verify_deployment.py      # 自己的 NIR 解释器
$ python3 examples/cross_validate.py         # 别人的：nirtorch + snnTorch
```

两者都是 1024 张真实 MNIST、`max |output diff| = 0.000e+00`、预测一致率 100%。
第一个证明"写出去的 NIR 和训练的网络是一回事"，第二个证明"这不是我们自己说的"。
细节见 [`P0_DEPLOY_MATRIX.md`](P0_DEPLOY_MATRIX.md) 与
[`P0_CROSS_VALIDATION.md`](P0_CROSS_VALIDATION.md)。
