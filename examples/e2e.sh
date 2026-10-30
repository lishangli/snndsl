#!/usr/bin/env bash
# 端到端跑一遍：写好的网络 -> 部署预检 -> 训练 -> 导出 -> 等价验证 -> 保真 -> 推理
#
#   bash examples/e2e.sh                        # 用 examples/mnist_hw.snn
#   bash examples/e2e.sh path/to/mynet.snn      # 用你自己的网络
#   EPOCHS=3 SAMPLES=2000 bash examples/e2e.sh  # 少跑几轮、少测几张
#
# 产物全部落在 $OUT（默认 build_e2e/），最后会逐个说明它们是什么。
set -euo pipefail

SRC="${1:-examples/mnist_hw.snn}"
OUT="${OUT:-build_e2e}"
EPOCHS="${EPOCHS:-3}"
SAMPLES="${SAMPLES:-2000}"
NAME="$(basename "${SRC%.*}")"

# -- with `pip install -e .` the console script exists; otherwise fall back to -m
if command -v snndsl >/dev/null 2>&1; then
    SNN=(snndsl)
else
    SNN=(python3 -m snndsl)
fi

step() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

[ -f "$SRC" ] || { echo "找不到 $SRC" >&2; exit 1; }
mkdir -p "$OUT"

step "0. 这台机器能做什么"
"${SNN[@]}" doctor

step "1. 训练之前：它能不能导出到硬件（不需要数据，不需要权重）"
"${SNN[@]}" deploy "$SRC" || true

step "2. 训练 ${EPOCHS} 个 epoch"
"${SNN[@]}" run "$SRC" --epochs "$EPOCHS" --out "$OUT" --json "$OUT/metrics.json"

step "3. 导出 NIR"
"${SNN[@]}" export "$SRC" --weights "$OUT/$NAME.pt" -o "$OUT/$NAME.nir"

step "4. 导出前后还是同一个网络吗（读回磁盘、独立解释器、逐样本比）"
"${SNN[@]}" deploy "$SRC" --verify --weights "$OUT/$NAME.pt" --data mnist \
    --report "$OUT/deploy_report.html"

step "5. 量化与换 tau 离散化之后，脉冲错多少"
"${SNN[@]}" fidelity "$OUT/$NAME.nir"

step "6. 推理：一个样本 + 全量精度"
"${SNN[@]}" infer "$SRC" --weights "$OUT/$NAME.pt" --samples 1 --ascii
"${SNN[@]}" infer "$SRC" --weights "$OUT/$NAME.pt" --samples "$SAMPLES" --repeats 3

step "产物"
ls -1sh "$OUT" | sed 's/^/  /'
cat <<EOF

  每个文件是干什么的：
    $NAME.py             生成的可读 PyTorch 源码（可 diff、可手改）
    $NAME.ir.json        静态 IR，给别的后端消费
    $NAME.mlir           MLIR 文本（标准方言，mlir-opt / xdsl-opt 可读）
    $NAME.nir            神经形态硬件中间表示，Lava / SpiNNaker2 / Speck / Xylo 读它
    $NAME.pt             训练好的 checkpoint
    metrics.json         训练曲线、稀疏度、能耗
    deploy_report.html   单文件部署报告，可直接转发（内联 SVG，无脚本无外链）

  全部在 $OUT/ 下。下一步：
    snndsl infer $SRC --weights $OUT/$NAME.pt --image your.png
EOF
