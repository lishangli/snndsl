"""An SNN described by `forward` -- the whole network is visible in one function.

Same network as `examples/mnist_hw.snn`, written the way you write any other
model. `forward` describes **one time step**; T lives in `steps = 8` and is
applied by the compiler, which is exactly what `.snn` does with
`temporal: steps = 8`. So there is no loop over time to write here.

    python examples/mnist_forward.py --dry-run     # graph + cost, no training
    python examples/mnist_forward.py               # train, deploy, measure
    snndsl emit examples/mnist_forward.py          # print it as .snn text
    snndsl deploy examples/mnist_forward.py        # same verdict as the .snn one
"""
from __future__ import annotations

import argparse

import snndsl as snn


class MnistNet(snn.Module):
    """28x28 -> 256 -> 10, one LIF per layer, T = 8 time steps."""

    input_shape = (1, 28, 28)
    steps = 8
    readout = "spikes"
    # zero reset (v <- v * (1 - s)) is the only form NIR holds exactly;
    # docs/GAPS.md #1 -- subtract reset exports with v_reset=0 and diverges.
    reset = "zero"

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = snn.Dense(256, scale=2.0)
        self.fc2 = snn.Dense(10, scale=1.0)
        self.lif1 = snn.LIF(tau=2.0, v_th=1.0, reset="zero")
        self.lif2 = snn.LIF(tau=2.0, v_th=1.0, reset="zero")

    def forward(self, x):
        x = self.lif1(self.fc1(x))
        return self.lif2(self.fc2(x))


class DeepNet(snn.Module):
    """Why this front end exists: a *generating* loop.

    One line of Python emits three blocks. `.snn` cannot express this without
    writing the three layers out; here it is a loop, and `depth` is a parameter.
    """

    input_shape = (1, 8, 8)
    steps = 4
    readout = "membrane"

    def __init__(self, depth: int = 3, width: int = 32) -> None:
        super().__init__()
        self.depth = depth
        self.fc = snn.Dense(width, scale=2.0)
        self.lif = snn.LIF(tau=2.0)
        self.head = snn.Dense(10)

    def forward(self, x):
        for _ in range(self.depth):          # three blocks, not time steps
            x = self.lif(self.fc(x))
        return self.head(x)


class ResNet(snn.Module):
    """A residual block: `+` is the `add` op, and shapes are checked on the spot."""

    input_shape = (64,)
    steps = 4
    readout = "membrane"

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = snn.Dense(64, scale=1.5)
        self.fc2 = snn.Dense(64, scale=1.5)
        self.lif1 = snn.LIF(tau=2.0)
        self.lif2 = snn.LIF(tau=2.0)
        self.head = snn.Dense(10)

    def forward(self, x):
        h = self.lif1(self.fc1(x))
        h = self.lif2(self.fc2(h)) + x       # skip connection
        return self.head(h)


# what `snndsl deploy` / `snndsl emit` picks up: instantiating a description is
# free (no graph is built until compile()), so this is not eager work
net = MnistNet()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--out", default="build_forward")
    args = ap.parse_args(argv)

    model = MnistNet()

    if args.emit:
        print(model.to_snn("# traced from MnistNet.forward"), end="")
        return 0

    print("--- MnistNet.forward, as the compiler sees it ---")
    print(model.describe())

    print("\n--- DeepNet: `for i in range(self.depth)` really is three blocks ---")
    print(DeepNet().describe())

    print("\n--- ResNet: `h + x` is the add op ---")
    print(ResNet().describe())

    if args.dry_run:
        print()
        print(model.compile(load=False).summary())
        return 0

    result = model.compile()
    print()
    print(result.summary())

    from snndsl.runner import run_program_from_result
    metrics = run_program_from_result(result, out_dir=args.out, epochs=args.epochs)
    print(f"\nparams={metrics['params']:,}  "
          f"spikes/sample={metrics['spikes']['spikes_per_sample']:.0f}")
    if "train" in metrics:
        print(f"history={metrics['train']['history']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
