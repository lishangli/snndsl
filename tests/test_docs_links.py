"""Every relative link in the documentation must resolve.

Cheap test, high value: a broken link in a README is the first thing a reader
hits and the last thing anyone notices, because nothing fails when you rename a
file. This walks every Markdown file in the repository and checks that each
relative target exists.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _markdown_files():
    for path in sorted(ROOT.rglob("*.md")):
        parts = set(path.parts)
        if ".git" in parts or any(p.startswith("build") for p in parts):
            continue
        yield path


def test_every_relative_link_resolves():
    broken = []
    checked = 0
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for raw in LINK.findall(text):
            if raw.startswith(("http://", "https://", "mailto:", "#")):
                continue
            anchor = raw.split("#", 1)[0]
            if not anchor:
                continue
            checked += 1
            target = (path.parent / anchor).resolve()
            if not target.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {raw}")
    assert checked > 20, f"only {checked} relative links found; the scan is broken"
    assert not broken, "broken links:\n" + "\n".join(broken)
