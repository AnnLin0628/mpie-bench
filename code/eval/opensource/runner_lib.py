#!/usr/bin/env python3
"""Shared helpers for opensource runners."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from common import weights_root


def code_root() -> Path:
    env = os.environ.get("MPIE_CODE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "mpie_code"


def prepend_sys_path(*paths: Path | str) -> None:
    for p in paths:
        s = str(Path(p).expanduser().resolve())
        if s not in sys.path:
            sys.path.insert(0, s)


def find_first(*cands: Path) -> Path | None:
    for p in cands:
        if p.is_file() or p.is_dir():
            return p
    return None


def limit_refs(refs: list[Path], max_n: int) -> list[Path]:
    if max_n <= 0 or len(refs) <= max_n:
        return refs
    return refs[:max_n]
