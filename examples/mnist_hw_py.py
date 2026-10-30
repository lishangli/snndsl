"""`examples/mnist_hw.snn` written as Python -- same network, same IR, same deploy.

This exists to make one point checkable: the two front ends are alternatives,
not stages. Run either one and you get the same `.nir`:

    python examples/mnist_hw_py.py            # build + train + measure
    snndsl deploy examples/mnist_hw_py.py     # same verdict as the .snn version
    snndsl emit   examples/mnist_hw_py.py     # print it as .snn source

Why write it in Python at all: a loop, a parameter or a helper function is
ordinary Python here, while the text format has no functions and no loops.
"""
from __future__ import annotations

import argparse
import sys

import snndsl as snn


def build(hidden: int = 256, steps: int = 8) -> snn.Net:
    # zero reset (v <- v * (1 - s)) is the one NIR can hold exactly; see
    # docs/GAPS.md #1 -- subtract reset exports with v_reset=0 and diverges.
    LIF = snn.LIF(tau=2.0, v_th=1.0, reset="zero")
    net = snn.Net("mnist_hw", steps=steps, reset="zero", readout="spikes")

    x = net.encode(net.input(1, 28, 28), "rate", gain=1.0)
    h = net.neuron(net.dense(x, hidden, scale=2.0, name="h1"), LIF, name="a1")
    net.output(net.neuron(net.dense(h, 10, scale=1.0, name="y"), LIF, name="ay"))

    net.train(data="mnist", epochs=10, batch=128, lr=1e-3)
    net.budget(spikes_per_sample=40000)
    return net


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the graph only")
    ap.add_argument("--emit", action="store_true", help="print the .snn form")
    ap.add_argument("--out", default="build_hw_py")
    args = ap.parse_args(argv)

    net = build(args.hidden, args.steps)
    if args.emit:
        print(net.to_snn(), end="")
        return 0

    print(net.describe())
    if args.dry_run:
        print()
        print(net.compile(quiet=False).summary())
        return 0

    metrics = net.run(out_dir=args.out, epochs=args.epochs)
    print(f"\nparams={metrics['params']:,}  "
          f"spikes/sample={metrics['spikes']['spikes_per_sample']:.0f}")
    if "train" in metrics:
        print(f"final test acc: {metrics['train']['final_test_acc'] * 100:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
