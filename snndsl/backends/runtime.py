"""Runtime support for compiled SNN modules.

Everything here is imported by *generated* code, so the API is intentionally
small and stable:

* ``surrogate_<kind>(x, alpha)`` - hard forward spike, soft backward gradient
* ``poisson_encode`` / ``rate_current`` / ``latency_encode`` - input coding
* ``estimate_energy`` - 45 nm MAC/AC energy proxy used in compile reports
* ``build_dataloaders`` / ``train_module`` / ``evaluate`` - training plumbing

The surrogate-gradient formulation follows the standard literature (Neftci et
al., 2019, "Surrogate Gradient Learning in Spiking Neural Networks").
"""
from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..errors import BackendError

# --------------------------------------------------------------- surrogates


class _SpikeFn(torch.autograd.Function):
    """Heaviside forward, surrogate derivative on the backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float, kind: str) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(x)
        ctx.alpha = float(alpha)
        ctx.kind = kind
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        k = ctx.kind
        if k in ("fast_sigmoid", "sigmoid"):
            sg = 1.0 / (1.0 + a * x.abs()) ** 2
        elif k == "atan":
            sg = (a / 2.0) / (1.0 + (math.pi / 2.0 * a * x) ** 2)
        elif k == "triangle":
            sg = torch.clamp(1.0 - a * x.abs(), min=0.0)
        elif k == "super_spike":
            sg = 1.0 / (1.0 + a * x.abs()) ** 2
        elif k == "rectangular":
            sg = (x.abs() < 0.5 / max(a, 1e-6)).to(x.dtype) * a
        else:
            sg = torch.ones_like(x)
        return grad_output * sg, None, None


def surrogate_fast_sigmoid(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    return _SpikeFn.apply(x, alpha, "fast_sigmoid")


def surrogate_sigmoid(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    return _SpikeFn.apply(x, alpha, "sigmoid")


def surrogate_atan(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return _SpikeFn.apply(x, alpha, "atan")


def surrogate_triangle(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _SpikeFn.apply(x, alpha, "triangle")


def surrogate_super_spike(x: torch.Tensor, alpha: float = 4.0) -> torch.Tensor:
    return _SpikeFn.apply(x, alpha, "super_spike")


def surrogate_rectangular(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _SpikeFn.apply(x, alpha, "rectangular")


def surrogate_straight_through(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """For spike predicates that are not a simple threshold comparison."""
    hard = (x > 0).to(x.dtype) if x.dtype.is_floating_point else x.to(torch.float32)
    return hard + x.to(hard.dtype) - x.to(hard.dtype).detach()


SURROGATES = {
    "fast_sigmoid": surrogate_fast_sigmoid, "sigmoid": surrogate_sigmoid,
    "atan": surrogate_atan, "triangle": surrogate_triangle,
    "super_spike": surrogate_super_spike, "rectangular": surrogate_rectangular,
    "straight_through": surrogate_straight_through,
}


# ------------------------------------------------------------------ encoding
def rate_current(x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Deterministic rate coding: a constant current proportional to intensity."""
    return x * gain


def poisson_encode(x: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """Bernoulli/rate coding: per-step spike with probability ``x * gain``."""
    p = (x * gain).clamp(0.0, 1.0)
    return (torch.rand_like(p) < p).to(x.dtype)


def latency_encode(x: torch.Tensor, t: int = 0, gain: float = 1.0,
                   steps: int = 8) -> torch.Tensor:
    """Time-to-first-spike: strong inputs fire early, weak ones late."""
    delay = (1.0 - (x * gain).clamp(0.0, 1.0)) * (steps - 1)
    return (delay <= t).to(x.dtype)


def delta_encode(x: torch.Tensor, prev: Optional[torch.Tensor] = None,
                 gain: float = 1.0, thresh: float = 0.05):
    """Send a spike only where the input changed enough (event-camera style)."""
    if prev is None:
        return (x.abs() * gain > thresh).to(x.dtype), x
    d = x - prev
    return (d.abs() * gain > thresh).to(x.dtype), x


def rate_readout(accum: torch.Tensor, steps: int) -> torch.Tensor:
    return accum / max(steps, 1)


# ------------------------------------------------------------- energy model
# 45 nm CMOS, Horowitz 2014: 32-bit float MAC vs 32-bit int AC.
ENERGY_MAC_PJ = 4.6
ENERGY_AC_PJ = 0.9


def estimate_energy(layer_stats: List[Dict[str, Any]], steps: int) -> Dict[str, Any]:
    """Compare dense ANN cost with event-driven SNN cost.

    ``layer_stats`` entries need ``op``, ``synapses`` (fan-in per output) and
    ``input_spikes`` (number of non-zero input events for one sample, one step).
    """
    ann_ops = 0
    snn_ops = 0
    for st in layer_stats:
        if st["op"] not in ("dense", "conv"):
            continue
        ann_ops += st.get("synapses", 0) * steps
        snn_ops += st.get("input_spikes", 0) * steps
    e_ann = ann_ops * ENERGY_MAC_PJ
    e_snn = snn_ops * ENERGY_AC_PJ
    return {
        "ann_macs": ann_ops,
        "snn_accumulates": snn_ops,
        "ann_energy_pj": e_ann,
        "snn_energy_pj": e_snn,
        "energy_ratio": (e_ann / e_snn) if e_snn > 0 else float("inf"),
        "assumptions": {"mac_pj": ENERGY_MAC_PJ, "ac_pj": ENERGY_AC_PJ,
                        "process": "45nm CMOS (Horowitz 2014)"},
    }


# ---------------------------------------------------------------- dataloaders
def _synthetic(shape: Tuple[int, ...], classes: int, n: int, seed: int = 0,
               centre_seed: int = 7):
    """Deterministic separable toy task, used when torchvision data is absent.

    Class centres come from ``centre_seed`` so train/test splits share the same
    decision problem; only the noise draw differs.
    """
    gc = torch.Generator().manual_seed(centre_seed)
    centres = torch.rand(classes, *shape, generator=gc)
    g = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, classes, (n,), generator=g)
    x = centres[labels] + 0.55 * torch.randn(n, *shape, generator=g)
    x = x.clamp(0, 1)
    return torch.utils.data.TensorDataset(x, labels)


def _default_root() -> str:
    """Prefer a local ``data/`` dir, fall back to ``$SNNDSL_DATA``."""
    env = os.environ.get("SNNDSL_DATA")
    if env:
        return env
    for cand in ("./data", "../data"):
        if os.path.isdir(cand):
            return cand
    return "./data"


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff", ".webp")
_ARRAY_KEYS = (("x_train", "y_train"), ("train_x", "train_y"), ("x", "y"))
_TEST_KEYS = (("x_test", "y_test"), ("test_x", "test_y"))


def _as_input_tensor(arr: np.ndarray, input_shape: Tuple[int, ...]) -> torch.Tensor:
    """Turn a raw array into the (N, ...) float tensor the graph expects.

    Integer arrays are scaled by 1/255 - that is the only convention every
    image dataset shares, and guessing anything else would be worse than
    documenting it.
    """
    a = np.asarray(arr)
    if a.dtype == np.uint8 or (np.issubdtype(a.dtype, np.integer) and a.max(initial=0) > 1):
        a = a.astype(np.float32) / 255.0
    t = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32)
    if len(input_shape) == 1:
        return t.reshape(t.shape[0], -1)
    if t.dim() == 3:                       # (N, H, W) of a single-channel dataset
        t = t.unsqueeze(1)
    return t


def _array_loader(path: str, input_shape: Tuple[int, ...], seed: int):
    """`.npz` / `.npy` of pre-loaded tensors: x_train/y_train[, x_test/y_test]."""
    if path.endswith(".npz"):
        blob = np.load(path)
        keys = next((k for k in _ARRAY_KEYS if k[0] in blob and k[1] in blob), None)
        if keys is None:
            raise BackendError(
                f"{path} has no arrays named x_train/y_train",
                hint=f"found {sorted(blob.files)}; rename them to x_train/y_train "
                     "(and optionally x_test/y_test)")
        x_tr = _as_input_tensor(blob[keys[0]], input_shape)
        y_tr = torch.as_tensor(np.asarray(blob[keys[1]]), dtype=torch.long)
        found = next((k for k in _TEST_KEYS if k[0] in blob and k[1] in blob), None)
        if found is not None:
            x_te = _as_input_tensor(blob[found[0]], input_shape)
            y_te = torch.as_tensor(np.asarray(blob[found[1]]), dtype=torch.long)
            return x_tr, y_tr, x_te, y_te, "npz"
    else:
        # a bare .npy has inputs only; inventing labels would be worse than refusing
        raise BackendError(
            f"{path} holds inputs but no labels",
            hint="use a .npz with x_train/y_train, or a folder of <class>/*.png")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x_tr.shape[0], generator=g)
    cut = int(0.8 * x_tr.shape[0])
    return x_tr[idx[:cut]], y_tr[idx[:cut]], x_tr[idx[cut:]], y_tr[idx[cut:]], "npz"


def _folder_loader(path: str, input_shape: Tuple[int, ...], seed: int):
    """Directory of ``<class>/*.png``; a ``train/`` + ``test/`` split if present."""
    try:
        import torchvision
        import torchvision.transforms as T
    except Exception as exc:                       # pragma: no cover - env dependent
        raise BackendError(
            f"reading image folders needs torchvision ({exc})",
            hint='pip install "snndsl[full]"') from exc

    tf: List[Any] = []
    if len(input_shape) == 3:
        c, h, w = (int(v) for v in input_shape)
        if c == 1:
            tf.append(T.Grayscale(num_output_channels=1))
        elif c == 3:
            tf.append(T.Lambda(lambda im: im.convert("RGB")))
        tf.append(T.Resize((h, w)))
    tf.append(T.ToTensor())
    xform = T.Compose(tf)

    def _ds(root):
        return torchvision.datasets.ImageFolder(root, transform=xform)

    def _subdirs(p):
        return sorted(d for d in os.listdir(p)
                      if os.path.isdir(os.path.join(p, d))) if os.path.isdir(p) else []

    train_dir = os.path.join(path, "train")
    if not _subdirs(train_dir):                    # no explicit split -> random 80/20
        full = _ds(path)
        n = len(full)
        if n < 2:
            raise BackendError(f"{path} holds no images",
                               hint="expected <path>/<class>/*.png")
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(n, generator=g).tolist()
        cut = int(0.8 * n)
        tr = torch.utils.data.Subset(full, idx[:cut])
        te = torch.utils.data.Subset(full, idx[cut:])
        x_tr, y_tr = _stack(tr)
        x_te, y_te = _stack(te)
        return x_tr, y_tr, x_te, y_te, "folder"
    tr = _ds(train_dir)
    test_dir = next((os.path.join(path, s) for s in ("test", "val", "valid")
                     if _subdirs(os.path.join(path, s))), None)
    if test_dir is None:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(tr), generator=g).tolist()
        cut = int(0.8 * len(tr))
        full_tr = tr
        tr = torch.utils.data.Subset(full_tr, idx[:cut])
        te = torch.utils.data.Subset(full_tr, idx[cut:])
    else:
        te = _ds(test_dir)
    x_tr, y_tr = _stack(tr)
    x_te, y_te = _stack(te)
    return x_tr, y_tr, x_te, y_te, "folder"


def _stack(ds):
    """Materialise a small image dataset into tensors (folders are usually small)."""
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def _custom_source(name: str) -> Optional[str]:
    """Return ``'folder'`` / ``'array'`` when ``name`` points at something on disk."""
    raw = name
    for prefix in ("folder:", "imagefolder:", "images:"):
        if raw.lower().startswith(prefix):
            return "folder"
    for prefix in ("npz:", "npy:", "array:"):
        if raw.lower().startswith(prefix):
            return "array"
    if os.path.isdir(raw) or raw.lower().endswith(_IMAGE_EXTS):
        return "folder"
    if raw.lower().endswith((".npz", ".npy")):
        return "array"
    return None


def _strip_prefix(name: str) -> str:
    for prefix in ("folder:", "imagefolder:", "images:", "npz:", "npy:", "array:"):
        if name.lower().startswith(prefix):
            return name[len(prefix):]
    return name


def build_dataloaders(cfg: Dict[str, Any], input_shape: Tuple[int, ...],
                      batch_size: int = 128, root: Optional[str] = None,
                      num_workers: int = 0) -> Dict[str, Any]:
    """Build loaders from the ``train:`` block.

    ``data =`` accepts four things:

    * ``mnist`` / ``fashion_mnist`` / ``cifar10`` - downloaded via torchvision
    * ``synthetic`` - a separable toy task, for smoke tests only
    * a directory of ``<class>/*.png`` images (optionally with ``train/`` and
      ``test/`` subdirectories); images are resized to the declared input shape
    * a ``.npz`` holding ``x_train``/``y_train`` (and optionally
      ``x_test``/``y_test``) as uint8 or float arrays

    Anything else is an error. Falling back to the synthetic task would produce
    a model that has nothing to do with the user's data, and a plausible-looking
    accuracy number attached to the wrong dataset is worse than a failure.
    """
    name = str(cfg.get("data", cfg.get("dataset", "mnist"))).strip()
    low = name.lower()
    root = root or str(cfg.get("root") or _default_root())
    torch.manual_seed(int(cfg.get("seed", 0)))
    flat = len(input_shape) == 1
    tv = None
    try:
        import torchvision
        import torchvision.transforms as T
        tv = torchvision
    except Exception:
        tv = None

    def _wrap(ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            drop_last=False, pin_memory=False)

    def _wrap_test(ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=max(batch_size, 256), shuffle=False,
            num_workers=num_workers)

    kind = _custom_source(name)
    if kind is not None:
        path = os.path.expanduser(_strip_prefix(name))
        if not os.path.exists(path):
            raise BackendError(
                f"dataset path {path!r} does not exist",
                hint="pass a directory of <class>/*.png images or a .npz with "
                     "x_train/y_train")
        loader = _folder_loader if kind == "folder" else _array_loader
        x_tr, y_tr, x_te, y_te, label = loader(path, input_shape, int(cfg.get("seed", 0)))
        if x_tr.shape[1:] != (() if flat else tuple(input_shape)) and not flat:
            raise BackendError(
                f"{path} yields samples of shape {tuple(x_tr.shape[1:])}, "
                f"but the graph declares {tuple(input_shape)}",
                hint="change the `input` line in the pipeline block, or resize "
                     "the data to match")
        sub = int(cfg.get("subset", 0) or 0)
        if sub and sub < x_tr.shape[0]:
            x_tr, y_tr = x_tr[:sub], y_tr[:sub]
        tr = torch.utils.data.TensorDataset(x_tr, y_tr)
        te = torch.utils.data.TensorDataset(x_te, y_te)
        return {"train": _wrap(tr), "test": _wrap_test(te),
                "kind": label, "flat": flat, "adapt": False}

    if tv is not None and low in ("mnist", "fashion_mnist", "fashion", "cifar10", "cifar"):
        tf = T.Compose([T.ToTensor()])
        if low == "mnist":
            tr = tv.datasets.MNIST(root, train=True, download=True, transform=tf)
            te = tv.datasets.MNIST(root, train=False, download=True, transform=tf)
        elif low in ("fashion_mnist", "fashion"):
            tr = tv.datasets.FashionMNIST(root, train=True, download=True, transform=tf)
            te = tv.datasets.FashionMNIST(root, train=False, download=True, transform=tf)
        else:
            tr = tv.datasets.CIFAR10(root, train=True, download=True, transform=tf)
            te = tv.datasets.CIFAR10(root, train=False, download=True, transform=tf)
        subset = int(cfg.get("subset", 0) or 0)
        if subset and subset < len(tr):
            idx = torch.randperm(len(tr))[:subset].tolist()
            tr = torch.utils.data.Subset(tr, idx)
        # adapt tensor layout to the declared input shape
        return {"train": _wrap(tr), "test": _wrap_test(te),
                "kind": low, "flat": flat, "adapt": True}

    if low not in ("synthetic", "toy"):
        raise BackendError(
            f"unknown dataset {name!r}",
            hint="use mnist | fashion_mnist | cifar10 | synthetic, or a path to a "
                 "folder of <class>/*.png images, or a .npz with x_train/y_train")

    n = int(cfg.get("subset", 0) or 4000)
    classes = int(cfg.get("classes", 10))
    tr = _synthetic(input_shape, classes, n, seed=11)
    te = _synthetic(input_shape, classes, max(n // 4, 200), seed=99)
    return {"train": _wrap(tr), "test": _wrap_test(te),
            "kind": "synthetic", "flat": flat, "adapt": False}


def adapt_batch(x: torch.Tensor, input_shape: Tuple[int, ...]) -> torch.Tensor:
    """Reshape a (B, 1, 28, 28) batch to the graph's declared input shape."""
    if len(input_shape) == 1:
        need = input_shape[0]
        x = x.reshape(x.shape[0], -1)
        if x.shape[1] < need:
            x = F.pad(x, (0, need - x.shape[1]))
        elif x.shape[1] > need:
            x = x[:, :need]
        return x
    # channel-first (C, H, W)
    c = input_shape[0]
    if x.shape[1] != c:
        x = x.repeat(1, c, 1, 1)[:, :c]
    return x


# ------------------------------------------------------------------ training
def evaluate(model: nn.Module, loader, input_shape, device="cpu",
             max_batches: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = adapt_batch(x, input_shape).to(device)
            y = y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    model.train()
    return {"acc": correct / max(total, 1), "n": total}


def train_module(model: nn.Module, graph, loaders, cfg: Dict[str, Any],
                 device: str = "cpu", log=print) -> Dict[str, Any]:
    """Generic BPTT training loop driven by the ``train:`` DSL block."""
    input_shape = tuple(graph.node(graph.inputs[0]).shape)
    epochs = int(cfg.get("epochs", 3))
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    opt_name = str(cfg.get("opt", "adam")).lower()
    loss_name = str(cfg.get("loss", "ce")).lower()
    reg = cfg.get("reg")
    label_smooth = float(cfg.get("label_smooth", 0.0))
    clip = float(cfg.get("clip", 0.0))
    max_batches = cfg.get("max_batches")

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adam":
        opt = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name in ("adamw",):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name in ("sgd",):
        opt = torch.optim.SGD(params, lr=lr, momentum=float(cfg.get("momentum", 0.9)),
                              weight_decay=wd)
    else:
        raise ValueError(f"unknown optimizer {opt_name!r}")

    sched = None
    if str(cfg.get("sched", "")).lower() in ("cosine", "cos"):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history: List[Dict[str, float]] = []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        run_loss = 0.0
        run_correct = 0
        run_n = 0
        for bi, (x, y) in enumerate(loaders["train"]):
            if max_batches is not None and bi >= int(max_batches):
                break
            x = adapt_batch(x, input_shape).to(device)
            y = y.to(device)
            out = model(x)
            if loss_name in ("ce", "cross_entropy"):
                loss = F.cross_entropy(out, y, label_smoothing=label_smooth)
            elif loss_name in ("mse",):
                tgt = F.one_hot(y, out.shape[1]).to(out.dtype)
                loss = F.mse_loss(out, tgt)
            else:
                raise ValueError(f"unknown loss {loss_name!r}")
            if isinstance(reg, dict) and reg.get("fn") == "spike_l2":
                coef = float(reg.get("args", [0.0])[0]) if reg.get("args") else 0.0
                loss = loss + coef * model.spike_penalty()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            run_loss += float(loss.detach()) * y.numel()
            run_correct += int((out.argmax(1) == y).sum())
            run_n += y.numel()
        if sched is not None:
            sched.step()
        train_loss = run_loss / max(run_n, 1)
        train_acc = run_correct / max(run_n, 1)
        ev = evaluate(model, loaders["test"], input_shape, device, max_batches=40)
        rec = {"epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
               "test_acc": ev["acc"], "elapsed_s": time.time() - t0,
               "lr": opt.param_groups[0]["lr"]}
        history.append(rec)
        log(f"  epoch {ep}/{epochs}  loss={train_loss:.4f}  "
            f"train_acc={train_acc * 100:.2f}%  test_acc={ev['acc'] * 100:.2f}%  "
            f"({rec['elapsed_s']:.1f}s)")
    return {"history": history, "elapsed_s": time.time() - t0,
            "final_test_acc": history[-1]["test_acc"] if history else 0.0}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------- post-training quant


@torch.no_grad()
def quantize_weights_int8(model: nn.Module, per_channel: bool = True) -> Dict[str, Any]:
    """Weight-only symmetric int8 PTQ (fake-quantised in place).

    Activations stay fp32 because spiking tensors are already 0/1; only the
    synapse matrices shrink, which is exactly where SNN memory goes.
    Returns the size accounting so a report can quote real numbers.
    """
    before = 0
    after = 0
    per_layer: List[Dict[str, Any]] = []
    for name, p in model.named_parameters():
        if p.dim() < 2:
            before += p.numel() * 4
            after += p.numel() * 4
            continue
        w = p.data
        if per_channel:
            reduce_dims = tuple(range(1, w.dim()))
            amax = w.abs().amax(dim=reduce_dims, keepdim=True)
        else:
            amax = w.abs().amax()
        scale = (amax / 127.0).clamp(min=1e-8)
        q = torch.round(w / scale).clamp(-127, 127)
        p.data = q * scale
        before += w.numel() * 4
        after += w.numel() * 1 + (w.numel() / max(int(np.prod(w.shape[:-1])), 1)) * 4
        per_layer.append({"name": name, "numel": w.numel(),
                          "bytes_fp32": w.numel() * 4,
                          "bytes_int8": w.numel()})
    return {
        "scheme": "int8 symmetric per-channel" if per_channel else "int8 symmetric per-tensor",
        "bytes_fp32": before,
        "bytes_quantized": after,
        "compression": (before / after) if after else None,
        "layers": per_layer,
    }


def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())
