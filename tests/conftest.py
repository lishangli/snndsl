"""Keep the test suite honest about what is actually installed.

The compiler core must work with numpy alone - torch and nir are backends, not
prerequisites - and CI runs exactly that configuration to keep it true. Without
this file, four test modules import torch at module scope and collection fails
before a single assertion runs, which would make the numpy-only job red for a
reason that has nothing to do with the code.

Skipping is explicit: the reason is printed, so a short run is never mistaken
for a green one.
"""
from __future__ import annotations

import importlib.util

HAVE_TORCH = importlib.util.find_spec("torch") is not None
HAVE_NIR = importlib.util.find_spec("nir") is not None

# module -> (condition that must hold, what is missing)
_REQUIRES = {
    "test_deploy.py": (HAVE_TORCH, "torch"),
    "test_export_integrity.py": (HAVE_TORCH and HAVE_NIR, "torch + nir"),
    "test_rockpool_probe.py": (HAVE_TORCH and HAVE_NIR, "torch + nir"),
    "test_targets.py": (HAVE_TORCH and HAVE_NIR, "torch + nir"),
}

collect_ignore = []
for _name, (_ok, _why) in _REQUIRES.items():
    if not _ok:
        collect_ignore.append(_name)


def pytest_report_header(config) -> str:
    bits = [
        f"torch: {'yes' if HAVE_TORCH else 'no'}",
        f"nir: {'yes' if HAVE_NIR else 'no'}",
    ]
    if collect_ignore:
        bits.append(f"ignored {len(collect_ignore)} backend test modules")
    return "snndsl environment: " + ", ".join(bits)
