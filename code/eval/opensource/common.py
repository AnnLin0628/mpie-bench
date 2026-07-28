#!/usr/bin/env python3
"""Open source runner Compatible entrance: the implementation has converged to the upper level pack_io.py。"""
from __future__ import annotations

import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1]
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from pack_io import (  # noqa: E402
    add_common_args,
    already_done,
    count_outputs,
    iter_todo,
    load_manifest,
    make_meta,
    pack_root,
    resolve_refs,
    sample_out_path,
    seed_model_id,
    weights_root,
    write_meta,
)

__all__ = [
    "add_common_args",
    "already_done",
    "count_outputs",
    "iter_todo",
    "load_manifest",
    "make_meta",
    "pack_root",
    "resolve_refs",
    "sample_out_path",
    "seed_model_id",
    "weights_root",
    "write_meta",
]
