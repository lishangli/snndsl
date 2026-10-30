"""Training on your own data, and the step count a `.nir` has to carry.

Two claims are checked here, both about information that used to be lost in
silence:

* ``data =`` pointing at a folder of images or a ``.npz`` really loads *that*
  data. Any other unrecognised name is an error -- falling back to the built-in
  synthetic task would attach a plausible accuracy number to the wrong dataset,
  which is worse than a failure.
* ``snndsl export`` writes the step count into the file. Without it every
  downstream reader has to guess how often to run a single-step graph.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from snndsl.backends import runtime as rt  # noqa: E402
from snndsl.errors import BackendError  # noqa: E402
from snndsl.pipeline import compile_source, load_module  # noqa: E402

SRC = """
program own
const DATA = "__DATA__"
const EPOCHS = 1

temporal:
    steps = 2
    reset = zero

neuron LIF:
    param tau = 2.0
    param v_th = 1.0
    state v = 0.0
    v = v * (1 - dt / tau) + i
    spike s = v > v_th
    v = v * (1 - s)
    surrogate = fast_sigmoid(alpha=4.0)

pipeline:
    input  x : 1, 8, 8
    encode x -> rate with gain=1.0
    dense  h : 8 <- x with scale=2.0
    neuron h -> LIF
    dense  y : 3 <- h with scale=1.0
    neuron y -> LIF
    output out <- y

train:
    data = DATA
    epochs = EPOCHS
    batch = 4
    lr = 0.01

infer:
    mode = event
    readout = spikes
"""


def _write_folder(root, per_class=6, size=8):
    """Three classes of deterministic blocks -- enough to check plumbing."""
    from PIL import Image

    for c in range(3):
        d = root / str(c)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            arr = np.full((size, size), (c * 60 + i * 7) % 256, dtype=np.uint8)
            arr[i % size, :] = 255
            Image.fromarray(arr, mode="L").save(d / f"{i:03d}.png")
    return root


def test_folder_of_images_is_the_dataset(tmp_path):
    root = _write_folder(tmp_path / "ds")
    loaders = rt.build_dataloaders({"data": str(root), "seed": 0}, (1, 8, 8), batch_size=4)

    assert loaders["kind"] == "folder"
    x, y = next(iter(loaders["train"]))
    assert tuple(x.shape[1:]) == (1, 8, 8)
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert set(y.tolist()) <= {0, 1, 2}
    # 18 images, no explicit split -> 80/20
    assert len(loaders["train"].dataset) == 14
    assert len(loaders["test"].dataset) == 4


def test_npz_is_the_dataset(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "d.npz"
    np.savez(path,
             x_train=rng.integers(0, 256, (12, 8, 8), dtype=np.uint8),
             y_train=np.arange(12) % 3,
             x_test=rng.integers(0, 256, (5, 8, 8), dtype=np.uint8),
             y_test=np.arange(5) % 3)
    loaders = rt.build_dataloaders({"data": str(path)}, (1, 8, 8), batch_size=4)

    assert loaders["kind"] == "npz"
    assert len(loaders["train"].dataset) == 12
    assert len(loaders["test"].dataset) == 5        # the explicit split is used
    x, _ = next(iter(loaders["train"]))
    assert tuple(x.shape[1:]) == (1, 8, 8)
    assert x.max() <= 1.0                            # uint8 is scaled by 1/255


def test_unknown_dataset_name_is_an_error():
    with pytest.raises(BackendError) as exc:
        rt.build_dataloaders({"data": "my_dataset"}, (1, 8, 8))
    assert "my_dataset" in str(exc.value)
    assert "synthetic" in (exc.value.diagnostic.hint or "")


def test_flat_input_is_flattened(tmp_path):
    rng = np.random.default_rng(1)
    path = tmp_path / "flat.npz"
    np.savez(path, x_train=rng.random((6, 8, 8)).astype(np.float32),
             y_train=np.arange(6) % 3)
    loaders = rt.build_dataloaders({"data": str(path)}, (64,), batch_size=3)
    x, _ = next(iter(loaders["train"]))
    assert x.shape[1:] == (64,)


def test_it_trains_on_the_folder(tmp_path):
    """The whole path: .snn -> compile -> loaders -> BPTT -> better than chance?"""
    root = _write_folder(tmp_path / "ds", per_class=8)
    res = compile_source(SRC.replace("__DATA__", str(root)), quiet=True)
    Net = load_module(res, str(tmp_path / "own.py"))
    model = Net()

    graph = res.graph
    shape = tuple(graph.node(graph.inputs[0]).shape)
    loaders = rt.build_dataloaders(graph.train, shape, batch_size=4)
    before = [p.detach().clone() for p in model.parameters()]

    hist = rt.train_module(model, graph, loaders, graph.train, log=lambda *a: None)
    assert len(hist["history"]) == 1
    assert 0.0 <= hist["final_test_acc"] <= 1.0
    # weights moved: a loop that silently skipped the optimiser step would pass
    # every shape assertion above and still return a number here
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=False))


def test_export_records_the_step_count(tmp_path):
    nir = pytest.importorskip("nir")
    from snndsl.backends.nir_export import export_nir

    res = compile_source(SRC.replace("__DATA__", "synthetic"), quiet=True)
    out = tmp_path / "own.nir"
    export_nir(res.graph, str(out))

    meta = dict(nir.read(str(out)).metadata)
    assert meta["snndsl.timesteps"] == str(res.graph.temporal.steps)
    assert meta["snndsl.program"] == "own"
