#!/usr/bin/env python3
"""Review pack Read and write (open source / closed source).

See the same directory for unified layout RESULT_LAYOUT.md：
  $PACK/outputs/<model_id>/<sample_id>.png
  $PACK/outputs/<model_id>/_meta/<sample_id>.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator


def pack_root(p: str | Path | None = None) -> Path:
    if p:
        return Path(p).expanduser().resolve()
    env = os.environ.get("MPIE_TEST_PACK")
    if env:
        return Path(env).expanduser().resolve()
    # Prefer the in-repo official pack shipped with this repository.
    repo_pack = Path(__file__).resolve().parents[2] / "data" / "testset"
    if (repo_pack / "manifest.jsonl").exists():
        return repo_pack
    return Path.home() / "mpie_testset_pack"


def weights_root(p: str | Path | None = None) -> Path:
    """Open source weight root directory. Priority MPIE_WEIGHTS,otherwise ~/mpie_weights。"""
    if p:
        return Path(p).expanduser().resolve()
    env = os.environ.get("MPIE_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / "mpie_weights"


def load_manifest(root: Path) -> list[dict]:
    man = root / "manifest.jsonl"
    if not man.exists():
        raise FileNotFoundError(
            f"missing {man}; First export_pack.py / make_subset.py Run the review again"
        )
    rows = []
    for line in man.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sample_out_path(root: Path, model_id: str, sample_id: str) -> Path:
    d = root / "outputs" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sample_id}.png"


def already_done(root: Path, model_id: str, sample_id: str) -> bool:
    p = sample_out_path(root, model_id, sample_id)
    return p.exists() and p.stat().st_size > 1000


def iter_todo(
    root: Path,
    model_id: str,
    limit: int = 0,
    shard_id: int = 0,
    num_shards: int = 1,
) -> Iterator[dict]:
    """Yield pending samples. With num_shards>1, only keep index % num_shards == shard_id."""
    n = 0
    for i, row in enumerate(load_manifest(root)):
        if num_shards > 1 and (i % num_shards) != shard_id:
            continue
        sid = row["sample_id"]
        if already_done(root, model_id, sid):
            continue
        yield row
        n += 1
        if limit and n >= limit:
            break


def resolve_refs(root: Path, row: dict) -> list[Path]:
    paths = []
    for rel in row.get("ref_relpaths") or []:
        p = root / rel
        if not p.exists():
            raise FileNotFoundError(p)
        paths.append(p)
    return paths


def seed_model_id(base: str, seed: int, seed_tag: bool) -> str:
    """When seed_tag, write to outputs/<base>_s<seed>/ so multi-seed runs don't clobber seed0."""
    if seed_tag:
        return f"{base}_s{int(seed)}"
    return base


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument(
        "--pack",
        default="",
        help="Test-set pack root (default: $MPIE_TEST_PACK or data/testset)",
    )
    ap.add_argument("--limit", type=int, default=0, help="Max new samples to run (0 = all pending)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-id", type=int, default=0, help="Multi-GPU shard index (0-based)")
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total multi-GPU shards; keep rows where index %% num_shards == shard_id",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="generate RNG seed(default 0, consistent with the full history)",
    )
    ap.add_argument(
        "--seed-tag",
        action="store_true",
        help="Write the output to outputs/<model>_s<seed>/(many seed The experiment must be turned on; it is turned off by default to avoid changing the full path)",
    )
    return ap


def write_meta(root: Path, model_id: str, sample_id: str, meta: dict) -> Path:
    d = root / "outputs" / model_id / "_meta"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{sample_id}.json"
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return path


def make_meta(
    *,
    sample_id: str,
    model_id: str,
    backend: str,
    seconds: float | None = None,
    n_refs: int = 0,
    ok: bool = True,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """unified meta schema(Open source / Closed sources all write these fields.extra Expandable). """
    m: dict[str, Any] = {
        "sample_id": sample_id,
        "model_id": model_id,
        "backend": backend,  # "opensource" | "closedsource"
        "ok": ok,
        "n_refs": n_refs,
        "seconds": round(seconds, 2) if seconds is not None else None,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if error:
        m["error"] = error
        m["ok"] = False
    if extra:
        m.update(extra)
    return m


def count_outputs(root: Path, model_id: str) -> int:
    d = root / "outputs" / model_id
    if not d.is_dir():
        return 0
    return sum(1 for p in d.glob("*.png") if p.is_file())
