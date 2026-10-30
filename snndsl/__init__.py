"""snndsl - a domain specific language and compiler for spiking neural networks.

Quick start
-----------
>>> from snndsl import compile_source, load_module
>>> res = compile_source(open("examples/mnist_mlp.snn").read())
>>> Net = load_module(res, "build/net.py")
>>> model = Net()
"""
from .errors import Diagnostic, DiagnosticBag, SNNError
from .ir import IRGraph, IRNode, NeuronSpec, TemporalSpec
from .pipeline import CompileResult, compile_file, compile_source, load_module
from .pyfront import ALIF, IF, LIF, Net, Neuron, Tensor, custom, net

__version__ = "0.3.0"
__all__ = [
    "SNNError", "Diagnostic", "DiagnosticBag", "IRGraph", "IRNode",
    "NeuronSpec", "TemporalSpec", "CompileResult", "compile_file",
    "compile_source", "load_module", "__version__",
    # Python-embedded front end
    "Net", "net", "Tensor", "Neuron", "LIF", "IF", "ALIF", "custom",
]
