#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mesh_v3 → Checklist_M Second value placement (mesh_bin/）。

protocol: docs/eval_human_consistency_anat_inter.md §4

Default read $PACK/judgments/human_consistency/_thresholds.json(If none, the default is started).
If so _split.json, by default only export split internal entry;--all-mesh Export all mesh_v3。

usage:
  python export_mesh_checklist.py --pack "$MPIE_TEST_PACK"
  python export_mesh_checklist.py --pack ... --split pilot,holdout
  python export_mesh_checklist.py --pack ... --all-mesh --force
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import (  # noqa: E402
    atomic_write_json,
    load_thresholds,
    map_mesh_to_checklist,
    mesh_rec_path,
    pair_key,
)
from pack_io import pack_root  # noqa: E402


def _load_split_keys(split_path: Path, names: Iterable[str]) -> List[dict]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    want = set(names)
    items: List[dict] = []
    for name, rows in (data.get("splits") or {}).items():
        if name not in want:
            continue
        for r in rows:
            items.append(r)
    return items


def _iter_all_mesh(pack: Path) -> List[Tuple[str, str, Path]]:
    root = pack / "judgments" / "mesh_v3"
    out: List[Tuple[str, str, Path]] = []
    if not root.is_dir():
        return out
    for mid_dir in sorted(root.iterdir()):
        if not mid_dir.is_dir() or mid_dir.name.startswith("_"):
            continue
        mid = mid_dir.name
        for p in sorted(mid_dir.glob("*.json")):
            if p.name.startswith("_"):
                continue
            out.append((p.stem, mid, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", default="")
    ap.add_argument(
        "--split",
        default="guide,pilot,holdout,main",
        help="Comma separated split name; with --all-mesh Mutually exclusive priority --all-mesh",
    )
    ap.add_argument("--all-mesh", action="store_true", help="Export all mesh_v3,neglect split")
    ap.add_argument(
        "--thresholds",
        default="",
        help="default $PACK/judgments/human_consistency/_thresholds.json",
    )
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pack = pack_root(args.pack) if args.pack else pack_root()
    hc = pack / "judgments" / "human_consistency"
    thr_path = Path(args.thresholds).expanduser() if args.thresholds else hc / "_thresholds.json"
    thr = load_thresholds(thr_path if thr_path.is_file() else None)
    cal_path = pack / "judgments" / "mesh_v3" / "_calibration.json"
    calib = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.is_file() else {}
    item_map_path = hc / "_item_map_calib.json"
    item_map = (
        json.loads(item_map_path.read_text(encoding="utf-8"))
        if item_map_path.is_file()
        else None
    )
    if item_map:
        print(
            f"[mesh_bin] using item_map {item_map.get('version')} "
            f"({len(item_map.get('items') or {})} items)",
            flush=True,
        )
    else:
        print("[mesh_bin] no _item_map_calib.json → heuristic map_v5", flush=True)

    jobs: List[Tuple[str, str, Optional[Path]]] = []
    if args.all_mesh:
        for sid, mid, p in _iter_all_mesh(pack):
            jobs.append((sid, mid, p))
    else:
        split_path = hc / "_split.json"
        if not split_path.is_file():
            raise SystemExit(
                f"missing {split_path}; run select_consistency_split.py first "
                f"or pass --all-mesh"
            )
        names = [x.strip() for x in args.split.split(",") if x.strip()]
        for r in _load_split_keys(split_path, names):
            sid, mid = r["sample_id"], r["model_id"]
            jobs.append((sid, mid, mesh_rec_path(pack, mid, sid)))

    if args.limit > 0:
        jobs = jobs[: args.limit]

    out_dir = hc / "mesh_bin"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ok = n_skip = n_miss = n_fail = 0
    for sid, mid, mp in jobs:
        out_p = out_dir / f"{pair_key(sid, mid)}.json"
        if out_p.is_file() and not args.force:
            n_skip += 1
            continue
        if mp is None or not mp.is_file():
            n_miss += 1
            continue
        try:
            rec = json.loads(mp.read_text(encoding="utf-8"))
            if "model_id" not in rec:
                rec["model_id"] = mid
            if "sample_id" not in rec:
                rec["sample_id"] = sid
            mapped = map_mesh_to_checklist(
                rec, thresholds=thr, calib=calib, item_map=item_map
            )
            mapped["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            mapped["source_mesh"] = str(mp.relative_to(pack)) if mp.is_relative_to(pack) else str(mp)
            atomic_write_json(out_p, mapped)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"[mesh_bin] FAIL {sid} {mid}: {e!r}", flush=True)

    summary = {
        "pack": str(pack),
        "out_dir": str(out_dir),
        "thresholds": str(thr_path) if thr_path.is_file() else "DEFAULT",
        "thresholds_version": thr.get("version"),
        "item_map_version": (item_map or {}).get("version"),
        "n_jobs": len(jobs),
        "n_ok": n_ok,
        "n_skip": n_skip,
        "n_miss_mesh": n_miss,
        "n_fail": n_fail,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    atomic_write_json(out_dir / "_export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
