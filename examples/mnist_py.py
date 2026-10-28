"""Python-embedded version of examples/mnist_membrane.snn.

Same network, same IR, same backends -- but built with ordinary Python, which
means ordinary control flow works:

    python examples/mnist_py.py            # build + train + measure
    python examples/mnist_py.py --layers 4 # 4 hidden layers, generated in a loop
    python examples/mnist_py.py --dry-run  # just show the graph
"""
from __future__ import annotations

import argparse
import sys

import snndsl as snn


def build(hidden: int = 256, layers: int = 1, steps: int = 8) -> snn.Net:
    LIF = snn.LIF(tau=2.0, v_th=1.0)

    net = snn.Net("mnist_py", steps=steps, readout="membrane")
    x = net.encode(net.input(1, 28, 28), "rate", gain=1.0)

    h = x
    for i in range(layers):
        # a loop is all it takes to unroll N blocks -- the text DSL cannot do this
        h = net.neuron(net.dense(h, hidden, scale=2.0, name=f"h{i + 1}"),
                       LIF, name=f"a{i + 1}")

    y = net.neuron(net.dense(h, 10, scale=1.0, name="y"), LIF, name="ay")
    net.output(y)

    net.train(data="mnist", epochs=4, batch=128, lr=1e-3,
              opt="adam", loss="ce", sched="cosine", seed=0)
    net.budget(spikes_per_sample=40000)
    return net


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the graph only")
    ap.add_argument("--out", default="build_py")
    args = ap.parse_args(argv)

    net = build(args.hidden, args.layers, args.steps)
    print(net.describe())
    print()

    if args.dry_run:
        res = net.compile(quiet=False)
        print(res.summary())
        return 0

    metrics = net.run(out_dir=args.out, epochs=args.epochs)
    print(f"\nparams={metrics['params']:,}  "
          f"spikes/sample={metrics['spikes']['spikes_per_sample']:.0f}")
    if "train" in metrics:
        print(f"final test acc: {metrics['train']['final_test_acc'] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
