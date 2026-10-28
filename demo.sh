#!/usr/bin/env bash
# End-to-end demonstration of the snndsl toolchain.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. inspect: IR, shapes, optimisation report =="
python3 -m snndsl inspect examples/mnist_membrane.snn

echo
echo "== 2. build: generated module + IR json + NIR =="
python3 -m snndsl build examples/cifar_cnn.snn -o build

echo
echo "== 3. run: train on MNIST, measure sparsity / energy / int8 =="
python3 -m snndsl run examples/mnist_membrane.snn --out build_demo

echo
echo "== 4. infer: one sample, with the input rendered as ASCII =="
python3 -m snndsl infer examples/mnist_membrane.snn --out build_demo \
    --weights build_demo/mnist_membrane.pt --samples 1 --ascii

echo
echo "== 5. infer: full test set + latency =="
python3 -m snndsl infer examples/mnist_membrane.snn --out build_demo \
    --weights build_demo/mnist_membrane.pt --samples 10000 --repeats 10

echo
echo "== 6. export: NIR for neuromorphic hardware, with weights =="
python3 -m snndsl export examples/mnist_membrane.snn -o build_demo/mnist.nir \
    --weights build_demo/mnist_membrane.pt

echo
echo "== 7. python frontend: the same IR, built from Python =="
PYTHONPATH=. python3 examples/mnist_py.py --dry-run

echo
echo "== 8. python frontend: a loop-generated 3-block network =="
PYTHONPATH=. python3 examples/mnist_py.py --layers 3 --dry-run | head -20

echo
echo "== 9. report: single-file HTML =="
python3 -m snndsl report examples/mnist_membrane.snn -o build_demo/report.html \
    --weights build_demo/mnist_membrane.pt

echo
echo "== 10. tests =="
python3 -m unittest discover -s tests
