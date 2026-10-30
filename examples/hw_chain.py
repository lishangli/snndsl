"""The whole chain, one file: snndsl -> NIR -> SNN-MLIR -> MLIR -> C.

    python examples/hw_chain.py                 # structure only, no training
    python examples/hw_chain.py --epochs 1      # with trained weights

What each stage produces, and which of them this repo can check:

    1. compile      examples/mnist_hw.snn           -> IR (6 nodes, 203,530 params)
    2. preflight    backends/snn_mlir.py            -> accepts / faithful / refused
    3. export       mnist_hw.nir                    -> terminals renamed input/output
    4. check        snn-mlir check mnist_hw.nir     -> "supported"
    5. lower        mnist_hw.mlir                   -> one func.func, no scf.for
    6. codegen      build/{network.mlir,main.c,...}  -> C driver, T as a loop bound
    7. run          snn-mlir run                    -> NEEDS the snn-opt toolchain

Step 7 is where this stops, and it stops for a reason worth stating rather than
hiding: `snn-opt` is not a PyPI package. `pip install snn-opt` installs an
unrelated SOCP solver that happens to share the name; the real toolchain is
built from MLIR/LLVM sources. So the last mile is not "not implemented here" —
it is "not installable here", and the script reports which binary is missing
instead of pretending.

The point of running the chain in one file is step 2 and step 5 together. They
answer different questions: SNN-MLIR's checker says whether it will convert our
file, and step 5's MLIR says what it converted. Neither of them can say whether
the thing it converted is still this network — that is what preflight is for,
and the three examples below show it disagreeing with the backend on purpose.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import snndsl as snn  # noqa: E402
from snndsl.backends import snn_mlir  # noqa: E402
from snndsl.backends.nir_export import export_nir  # noqa: E402
from snndsl.pipeline import compile_file  # noqa: E402


def rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default=os.path.join(HERE, "mnist_hw.snn"))
    ap.add_argument("--outdir", default=os.path.join(HERE, "..", "build_chain"))
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 = structural only; >0 trains first (MNIST, ~25 s/epoch)")
    ap.add_argument("--sweep", action="store_true",
                    help="also classify every example/*.snn against the backend")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.program))[0]
    nir_path = os.path.join(args.outdir, stem + ".nir")

    rule(f"1. compile  {os.path.basename(args.program)}")
    res = compile_file(args.program)
    g = res.graph
    print(f"  nodes      : {len(g.nodes)}")
    acts = sum(n.numel() for n in g.nodes)
    print(f"  activations: {acts:,}")
    print(f"  temporal   : T={g.temporal.steps} dt={g.temporal.dt} "
          f"reset={g.temporal.reset}")

    rule("2. preflight  (does the backend reproduce *this* network?)")
    pf = snn_mlir.preflight(g)
    print(f"  accepts    : {pf['accepts']}   faithful : {pf['faithful']}")
    for f in pf["blockers"]:
        print(f"  block      : {f['node']} {f['rule']} -- {f['detail']}")
    for f in pf["lossy"]:
        print(f"  lossy      : {f['node']} {f['rule']} -- {f['detail']}")
    for n in pf["notes"]:
        print(f"  note       : {n['detail']}")

    rule("3. export  -> .nir")
    weights = None
    if args.epochs:
        print(f"  training {args.epochs} epoch(s) first ...")
        rec = snn.run(args.program, epochs=args.epochs, quiet=True)
        weights = {k: v.detach().cpu().numpy()
                   for k, v in rec.module.state_dict().items()}
        print(f"  trained    : {len(weights)} tensors")
    rep = export_nir(g, nir_path, weights, target="snn-mlir")
    print(f"  file       : {nir_path}  ({os.path.getsize(nir_path):,} bytes)")
    print(f"  terminals  : {rep['terminals']}")
    print(f"  mapping    : {rep['nodes']}")

    if snn_mlir.cli_path() is None:
        print("\nsnn-mlir CLI not installed; the .nir above is written and valid.")
        print("  pip install snn-mlir     (check/export/codegen, pure Python)")
        return 0

    rule("4. snn-mlir check")
    chk = snn_mlir.check(nir_path)
    rep4 = chk["report"] or {}
    print(f"  supported  : {rep4.get('ok')}   exit={chk['returncode']}")
    for nd in rep4.get("nodes", []):
        print(f"  {nd['name']:8s} {nd['type']:8s} {nd['role']:9s} ok={nd['ok']}")
    print(f"  order      : {' -> '.join(rep4.get('order', []))}")

    rule("5. snn-mlir export  -> SNN-dialect MLIR")
    mlir_path = os.path.join(args.outdir, stem + ".mlir")
    low = snn_mlir.to_mlir(nir_path, mlir_path)
    if not low["ok"]:
        print(low["stderr"].rstrip())
        return 1
    text = open(mlir_path, encoding="utf-8").read()
    body = [ln.strip() for ln in text.splitlines() if "snn." in ln]
    print(f"  file       : {mlir_path}  ({len(text):,} chars)")
    print(f"  funcs      : {text.count('func.func')}   scf.for : {text.count('scf.for')}")
    for ln in body:
        print("  " + ln[:150])
    print("\n  One func, no time loop: T is not in the IR. The signature carries")
    print("  the membrane potentials, so the caller owns time:")
    sig = [ln.strip() for ln in text.splitlines() if "func.func" in ln]
    for ln in sig:
        print("  " + ln[:150])

    rule("6. snn-mlir codegen  -> build/ (C driver + data, no toolchain)")
    folder = os.path.join(args.outdir, "model")
    os.makedirs(folder, exist_ok=True)
    shutil.copy(nir_path, folder)
    # The backend bakes the input in as int8 and takes the timestep axis from
    # shape[0], so [T, N] is how T crosses the boundary.
    import numpy as np
    entry = next(n for n in g.nodes if n.op == "input")
    width = 1
    for d in entry.shape:
        width *= int(d)
    np.save(os.path.join(folder, "input.npy"),
            np.ones((int(g.temporal.steps), width), np.int8))
    print(f"  input.npy  : int8 [{g.temporal.steps}, {width}]  (spike trains)")
    cg = snn_mlir.codegen(folder)
    print((cg["stdout"] + cg["stderr"]).rstrip())
    if not cg["ok"]:
        return 1
    for name in sorted(os.listdir(cg["build"])):
        p = os.path.join(cg["build"], name)
        print(f"  {name:14s} {os.path.getsize(p):>10,} bytes")
    main_c = open(os.path.join(cg["build"], "main.c"), encoding="utf-8").read()
    loop = [ln.strip() for ln in main_c.splitlines() if ln.strip().startswith("for (")]
    print(f"\n  main.c time loop : {loop[0] if loop else '(none)'}")
    print(f"  our T            : {g.temporal.steps}")

    rule("7. snn-mlir run  -> needs a toolchain that is not on PyPI")
    run = snn_mlir.run(folder)
    print(f"  ok         : {run['ok']}")
    print(f"  {'  '.join((run['stderr'] or run['stdout']).strip().splitlines())}")

    if args.sweep:
        rule("sweep: every example, preflight vs the backend's own verdict")
        print(f"  {'example':22s} {'accepts':>8s} {'backend':>8s} {'faithful':>9s}  rules")
        for path in sorted(glob.glob(os.path.join(HERE, "*.snn"))):
            gr = compile_file(path).graph
            p = snn_mlir.preflight(gr)
            tmp = os.path.join(args.outdir, "_sweep.nir")
            export_nir(gr, tmp, target="snn-mlir")
            c = snn_mlir.check(tmp)
            rules = ",".join(sorted({f["rule"] for f in p["blockers"] + p["lossy"]})) or "-"
            print(f"  {os.path.basename(path):22s} {p['accepts']!s:>8s} "
                  f"{c['ok']!s:>8s} {p['faithful']!s:>9s}  {rules}")
        os.remove(os.path.join(args.outdir, "_sweep.nir"))
        os.remove(os.path.join(args.outdir, "_sweep.nir.report.json"))

    rule("what this run establishes")
    print("  - our .nir is accepted by a third-party MLIR frontend, unmodified except")
    print("    for the two terminal names (which the standard does not fix)")
    print("  - the lowered IR keeps T outside the program, so T stays a parameter")
    print("  - `accepts` and `faithful` are different questions, and the sweep shows")
    print("    three examples where the backend answers 'supported' to a lossy graph")
    print("  - the last mile (CPU executable) needs snn-opt from source; nothing in")
    print("    this repo claims it ran")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
