"""Forward-style front end: describe an SNN the way you already describe an ANN.

The handle-passing API in :mod:`snndsl.pyfront` (``h = net.dense(h, 256)``) is a
graph builder -- fine for generated code, awkward for a human. This module adds
the shape everyone already knows::

    import snndsl as snn

    class MnistNet(snn.Module):
        input_shape = (1, 28, 28)
        steps = 8

        def __init__(self):
            super().__init__()
            self.fc1 = snn.Dense(256, scale=2.0)
            self.fc2 = snn.Dense(10)
            self.lif1 = snn.LIF(tau=2.0, reset="zero")
            self.lif2 = snn.LIF(tau=2.0, reset="zero")

        def forward(self, x):
            for t in range(self.steps):
                x = self.lif1(self.fc1(x))
            return self.lif2(self.fc2(x))

    model = MnistNet().compile()          # same IR, same passes, same codegen

How this works, and why it is not tracing
-----------------------------------------
``forward`` is executed **exactly once, with the time loop pinned to one
iteration**. Every op call records an IR node and returns a symbolic handle, so
one pass over the loop body yields one copy of the cell -- which is precisely
the graph we want. ``torch.jit.trace`` cannot be used here: it would run the
loop T times and unroll the graph into T copies, and it would also lose the
state updates, because ``v = v * (1 - dt/tau) + i`` is a plain Python rebinding.

Two kinds of ``for`` therefore coexist, and they are told apart by their bound:

* ``for t in range(self.steps)`` -- the *time* loop. Trace sees one iteration.
* ``for i in range(3)`` -- a *generating* loop. Trace sees three iterations,
  so it really does emit three blocks. This is the whole point of describing a
  network in Python instead of text: ``.snn`` has no loops.

Anything the IR cannot hold is rejected loudly rather than silently flattened:
a data-dependent ``if``, a literal ``range(8)`` used as a time loop, a ``print``
inside ``forward``. See :func:`_check_forward` and :func:`_Tracer.op`.
"""
from __future__ import annotations

import ast
import inspect
import os
import tempfile
import textwrap
from typing import Any, Dict, Optional, Sequence, Tuple

from . import pyfront
from .errors import SNNError
from .pyfront import ALIF, IF, LIF, Neuron, Tensor, custom  # re-exported

__all__ = [
    "ALIF",
    "IF",
    "LIF",
    "Add",
    "AvgPool",
    "Concat",
    "Conv2d",
    "Dense",
    "Dropout",
    "Flatten",
    "Layer",
    "MaxPool",
    "Module",
    "Neuron",
    "custom",
]


# --------------------------------------------------------------------- tracer
class _Tracer:
    """Holds the ``Net`` being built while ``forward`` runs."""

    _active: Optional["_Tracer"] = None

    def __init__(self, net: pyfront.Net):
        self.net = net
        self.scope_counts: Dict[str, int] = {}

    # ------------------------------------------------------------- lifecycle
    def __enter__(self) -> "_Tracer":
        if _Tracer._active is not None:
            raise SNNError("a forward() is already being traced; nested "
                           "Module.forward calls are not supported",
                           hint="return the handle from a helper method instead "
                                "of calling another Module's forward")
        _Tracer._active = self
        return self

    def __exit__(self, *exc: Any) -> None:
        _Tracer._active = None

    # ---------------------------------------------------------------- naming
    def node_name(self, scope: Optional[str], op: str) -> Optional[str]:
        """Turn an attribute name (``self.fc1``) into a unique node id."""
        if not scope:
            return None
        stem = scope
        n = self.scope_counts.get(stem, 0)
        self.scope_counts[stem] = n + 1
        return stem if n == 0 else f"{stem}_{n + 1}"

    # ------------------------------------------------------------------- ops
    def op(self, op_name: str, scope: Optional[str], fn, *args, **kwargs):
        name = self.node_name(scope, op_name)
        if name:
            kwargs["name"] = name
        return fn(*args, **kwargs)


def _tracer() -> _Tracer:
    t = _Tracer._active
    if t is None:
        raise SNNError(
            "network layers can only be called while forward() is being traced",
            hint="build the graph with `model = MyNet().compile()`, then run the "
                 "compiled nn.Module -- not the description object")
    return t


# ---------------------------------------------------------------------- layers
class Layer:
    """A callable that appends one IR node. ``self.fc1`` names the node."""

    __slots__ = ("scope",)

    def __init__(self) -> None:
        self.scope: Optional[str] = None

    def __call__(self, *args: Any, **kwargs: Any):  # pragma: no cover - abstract
        raise NotImplementedError

    def build(self, net: pyfront.Net, *args: Any,
              name: Optional[str] = None, **kwargs: Any) -> Tensor:
        raise NotImplementedError

    def _call(self, *args: Any, **kwargs: Any) -> Tensor:
        t = _tracer()
        return t.op(type(self).__name__.lower(), self.scope, self.build,
                    t.net, *args, **kwargs)

    def extra_repr(self) -> str:
        return ""


class Dense(Layer):
    """Fully connected. Flattens its input, exactly like the ``dense`` op."""

    __slots__ = ("bias", "init", "scale", "units")

    def __init__(self, units: int, scale: float = 1.0, bias: bool = True,
                 init: str = "kaiming_normal") -> None:
        super().__init__()
        self.units = int(units)
        self.scale = float(scale)
        self.bias = bool(bias)
        self.init = init

    def __call__(self, x: Tensor) -> Tensor:
        return self._call(x)

    def build(self, net: pyfront.Net, x: Tensor,
              name: Optional[str] = None) -> Tensor:
        return net.dense(x, self.units, scale=self.scale, bias=self.bias,
                         init=self.init, name=name)

    def extra_repr(self) -> str:
        return f"units={self.units}, scale={self.scale}"


class Conv2d(Layer):
    __slots__ = ("bias", "kernel", "out_channels", "padding", "scale", "stride")

    def __init__(self, out_channels: int, kernel: int = 3, stride: int = 1,
                 padding: int = 0, scale: float = 1.0, bias: bool = True) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.kernel = int(kernel)
        self.stride = int(stride)
        self.padding = int(padding)
        self.scale = float(scale)
        self.bias = bool(bias)

    def __call__(self, x: Tensor) -> Tensor:
        return self._call(x)

    def build(self, net: pyfront.Net, x: Tensor,
              name: Optional[str] = None) -> Tensor:
        return net.conv(x, self.out_channels, kernel=self.kernel,
                        stride=self.stride, padding=self.padding,
                        scale=self.scale, bias=self.bias, name=name)

    def extra_repr(self) -> str:
        return (f"out_channels={self.out_channels}, kernel={self.kernel}, "
                f"stride={self.stride}, padding={self.padding}")


class MaxPool(Layer):
    __slots__ = ("kernel", "stride")

    def __init__(self, kernel: int = 2, stride: Optional[int] = None) -> None:
        super().__init__()
        self.kernel = int(kernel)
        self.stride = stride

    def __call__(self, x: Tensor) -> Tensor:
        return self._call(x)

    def build(self, net: pyfront.Net, x: Tensor,
              name: Optional[str] = None) -> Tensor:
        return net.pool(x, "max", self.kernel, self.stride, name=name)

    def extra_repr(self) -> str:
        return f"kernel={self.kernel}"


class AvgPool(MaxPool):
    def build(self, net: pyfront.Net, x: Tensor,
              name: Optional[str] = None) -> Tensor:
        return net.pool(x, "avg", self.kernel, self.stride, name=name)


class Flatten(Layer):
    __slots__ = ()

    def __call__(self, x: Tensor) -> Tensor:
        return self._call(x)

    def build(self, net: pyfront.Net, x: Tensor,
              name: Optional[str] = None) -> Tensor:
        return net.flatten(x, name=name)


class Dropout(Layer):
    __slots__ = ("p",)

    def __init__(self, p: float = 0.1) -> None:
        super().__init__()
        self.p = float(p)

    def __call__(self, x: Tensor) -> Tensor:
        return self._call(x)

    def build(self, net: pyfront.Net, x: Tensor,
              name: Optional[str] = None) -> Tensor:
        return net.dropout(x, self.p, name=name)


class Add(Layer):
    """Elementwise sum of two or more branches (residual connections)."""

    __slots__ = ()

    def __call__(self, *xs: Tensor) -> Tensor:
        return self._call(*xs)

    def build(self, net: pyfront.Net, *xs: Tensor,
              name: Optional[str] = None) -> Tensor:
        return net.add(*xs, name=name)


class Concat(Layer):
    __slots__ = ("axis",)

    def __init__(self, axis: int = 1) -> None:
        super().__init__()
        self.axis = int(axis)

    def __call__(self, *xs: Tensor) -> Tensor:
        return self._call(*xs)

    def build(self, net: pyfront.Net, *xs: Tensor,
              name: Optional[str] = None) -> Tensor:
        return net.concat(*xs, axis=self.axis, name=name)


# ---------------------------------------------------------------------- Module
def _self_attr(node: ast.AST, names: Sequence[str]) -> bool:
    return (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self" and node.attr in names)


def _check_forward(fn, steps: int, cls_name: str) -> None:
    """Reject the mistakes that would otherwise produce a plausible wrong graph.

    Without this, ``for t in range(8)`` inside a ``steps = 8`` network would
    unroll into eight *separate* blocks -- it compiles, it trains, and the
    numbers look almost right. That failure mode is worse than any crash.
    """
    try:
        src = textwrap.dedent(inspect.getsource(fn))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError):
        return  # exec'd or interactive: nothing to inspect, trust the user

    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        it = node.iter
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name)
                and it.func.id == "range"):
            continue
        if any(_self_attr(a, ("steps", "T")) for a in it.args):
            continue
        for a in it.args:
            if isinstance(a, ast.Constant) and a.value == steps and steps > 1:
                raise SNNError(
                    f"{cls_name}.forward loops over range({steps}) -- the same "
                    f"number as steps",
                    hint="write range(self.steps) for the time loop; the tracer "
                         "pins it to one iteration and the graph stays a loop. A "
                         "literal range() is treated as block duplication")




class Module:
    """Base class for a forward-described SNN.

    Class attributes
    ----------------
    ``input_shape``  required, e.g. ``(1, 28, 28)`` or ``(784,)``
    ``steps``        time steps T (default 8)
    ``dt``           time constant of the discretisation (default 1.0)
    ``reset``        ``"subtract"`` or ``"zero"`` (default ``"subtract"``)
    ``readout``      ``"membrane"`` or ``"spikes"`` (default ``"membrane"``)
    ``auto_encode``  insert the input encoder automatically (default True)
    ``name``         program name; defaults to the class name
    """

    input_shape: Optional[Tuple[int, ...]] = None
    steps: int = 8
    dt: float = 1.0
    reset: str = "subtract"
    readout: str = "membrane"
    schedule: str = "auto"
    auto_encode: bool = True
    name: Optional[str] = None

    def __init__(self) -> None:
        self._compiled: Optional[Any] = None

    # ------------------------------------------------------- attribute names
    def __setattr__(self, key: str, value: Any) -> None:
        if isinstance(value, (Layer, Neuron)):
            if value.scope is not None and value.scope != key:
                raise SNNError(
                    f"the same {type(value).__name__} instance is bound to both "
                    f"self.{value.scope} and self.{key}",
                    hint="forward() would have no way to tell them apart; call "
                         "snn.LIF(...) / snn.Dense(...) twice instead")
            value.scope = key
        object.__setattr__(self, key, value)

    # ------------------------------------------------------------- interface
    def forward(self, x: Tensor) -> Tensor:  # pragma: no cover - abstract
        raise NotImplementedError(f"{type(self).__name__}.forward is not defined")

    def __call__(self, *args: Any, **kwargs: Any):
        """Calling a description object compiles it and runs the real thing."""
        if self._compiled is None:
            self.compile()
        return self._compiled(*args, **kwargs)

    # ------------------------------------------------------------ convenience
    def encode(self, x: Tensor, scheme: str = "rate", gain: float = 1.0) -> Tensor:
        """Turn the analog input into current. Applied automatically by default."""
        if self.auto_encode:
            raise SNNError(
                "the input encoder is already applied: auto_encode is True",
                hint="drop self.encode(x) from forward(), or set "
                     "auto_encode = False to place it yourself")
        return _tracer().net.encode(x, scheme, gain)

    def add(self, *xs: Tensor) -> Tensor:
        return _tracer().net.add(*xs)

    def concat(self, *xs: Tensor, axis: int = 1) -> Tensor:
        return _tracer().net.concat(*xs, axis=axis)

    # --------------------------------------------------------------- compile
    def compile(self, passes: Optional[Sequence[str]] = None,
                class_name: str = "SnnNet", quiet: bool = False, load: bool = True):
        """Trace ``forward`` once, then run the shared compiler on the result."""
        from .pipeline import load_module

        cls = type(self)
        steps = int(getattr(self, "steps", 8))
        if steps < 1:
            raise SNNError(f"steps must be >= 1, got {steps}")
        shape = self.input_shape
        if not shape:
            raise SNNError(
                f"{cls.__name__} has no input_shape",
                hint="set a class attribute, e.g. `input_shape = (1, 28, 28)`; "
                     "the graph is static, so shapes must be known before "
                     "forward() runs")
        program = self.name or cls.__name__

        _check_forward(cls.forward, steps, cls.__name__)

        net = pyfront.Net(program, steps=steps, dt=float(self.dt),
                          reset=self.reset, readout=self.readout,
                          schedule=self.schedule)
        net.graph.filename = f"<forward {cls.__name__}>"
        net.graph.meta["frontend"] = "forward"

        # Pin the time loop to one iteration. This is what keeps the graph a
        # loop instead of T unrolled copies; every other for-loop still runs.
        object.__setattr__(self, "steps", 1)
        try:
            with _Tracer(net):
                x = net.input(*tuple(int(d) for d in shape))
                if self.auto_encode:
                    x = net.encode(x, "rate", 1.0)
                out = self.forward(x)
        finally:
            object.__setattr__(self, "steps", steps)

        if not isinstance(out, Tensor):
            raise SNNError(
                f"{cls.__name__}.forward returned {type(out).__name__}, "
                f"not a tensor handle",
                hint="return the last layer's output, e.g. `return self.fc2(x)`")
        net.output(out)

        result = net.compile(passes=passes, class_name=class_name, quiet=quiet)
        self._result = result
        if load:
            # a private directory, not cwd: compiling a description should not
            # leave _gen_MnistNet.py next to the user's sources
            out = tempfile.mkdtemp(prefix="snndsl_gen_")
            path = os.path.join(out, f"{program}.py")
            self._compiled = load_module(result, path)()
        return result

    # --------------------------------------------------------------- display
    def describe(self) -> str:
        return self.compile(load=False).graph.render_text()

    def to_snn(self, comment: str = "") -> str:
        """Print the traced graph as ``.snn`` text (the deploy-time format)."""
        from .serialize import to_snn
        return to_snn(self.compile(load=False).graph, comment=comment)

    def parameters_report(self) -> Dict[str, Any]:
        st = self.compile(load=False).graph.stats()
        return st
