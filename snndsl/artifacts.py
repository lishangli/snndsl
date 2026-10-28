"""Artifact emission: the files a successful compile leaves on disk.

This module is deliberately dependency-free. Emitting a python module, an IR
dump and a ``.nir`` file is a *compiler* job, so it must work on a machine that
has nothing installed but numpy. It used to live in :mod:`snndsl.runner`, whose
module-level ``import torch`` dragged the whole training stack into
``snndsl build`` and made the command die with a bare ``ModuleNotFoundError`` on
a backend-less box -- a compile that needs a GPU framework to write a text file
is a packaging bug, not a feature.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .pipeline import CompileResult


def export_artifacts(result: CompileResult, out_dir: str,
                     class_name: str = "SnnNet") -> Dict[str, str]:
    """Write ``<name>.py``, ``<name>.ir.json`` and (if nir is importable)
    ``<name>.nir`` into *out_dir*.

    Returns a mapping from artifact kind to path. A missing NIR backend is not
    fatal: the keys ``nir_error`` (why) and ``nir_json`` (the fallback the
    exporter left behind, if any) carry the news so callers can report it
    without failing the build.
    """
    os.makedirs(out_dir, exist_ok=True)
    name = result.graph.name
    paths = {
        "module": os.path.join(out_dir, f"{name}.py"),
        "ir": os.path.join(out_dir, f"{name}.ir.json"),
    }
    with open(paths["module"], "w", encoding="utf-8") as fh:
        fh.write(result.module_source)
    with open(paths["ir"], "w", encoding="utf-8") as fh:
        fh.write(result.graph.to_json())
    result.module_path = paths["module"]
    nir_path = os.path.join(out_dir, f"{name}.nir")
    try:
        from .backends.nir_export import export_nir
        export_nir(result.graph, nir_path)
        paths["nir"] = nir_path
    except Exception as exc:      # NIR is optional
        paths["nir_error"] = str(exc)
        # the exporter writes a JSON description of the graph when the nir
        # package is missing; report it instead of leaving an unexplained file
        # next to the artifacts
        if os.path.exists(nir_path + ".json"):
            paths["nir_json"] = nir_path + ".json"
    return paths
