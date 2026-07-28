#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""From complete pack sampling N strips (layered by cat),generate smoke subset pack。

usage(SG, finish running first export_pack.py）:
  python make_subset.py --src ~/mpie_testset_pack --out "$MPIE_TEST_PACK" --n 100
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _load_contact_map(csv_path: Path) -> dict[str, str]:
    """sample_id -> contact_c from targets_tagged.csv."""
    import csv

    m: dict[str, str] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = f"{row['board_cat']}__{row['anchor']}__{row['target_id']}"
            m[sid] = (row.get("contact_c") or "").strip()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--out", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--contact-levels",
        default="",
        help="optional filter, e.g. C2,C3 (needs --contact-csv)",
    )
    ap.add_argument(
        "--contact-csv",
        default=str(
            Path.home()
            / "mpie_bench/data/manifests/prompt_distribution/targets_tagged.csv"
        ),
        help="targets_tagged.csv for contact_c filter",
    )
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    man = src / "manifest.jsonl"
    if not man.exists():
        raise SystemExit(f"missing {man}; run export_pack.py first")

    rows = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
    levels = {x.strip() for x in args.contact_levels.split(",") if x.strip()}
    if levels:
        cmap = _load_contact_map(Path(args.contact_csv).expanduser())
        before = len(rows)
        rows = [r for r in rows if cmap.get(r["sample_id"]) in levels]
        print(f"contact filter {sorted(levels)}: {before} → {len(rows)}", flush=True)
        if len(rows) < args.n:
            raise SystemExit(
                f"only {len(rows)} rows after contact filter; lower --n or widen levels"
            )
    by_cat: dict[str, list] = defaultdict(list)
    for r in rows:
        by_cat[r["cat"]].append(r)

    rng = random.Random(args.seed)
    cats = sorted(by_cat.keys())
    # Each category at least 1 bar (if the class is not empty), and then fill it in proportion to n
    picked = []
    for c in cats:
        pool = by_cat[c][:]
        rng.shuffle(pool)
        if pool:
            picked.append(pool.pop())
            by_cat[c] = pool

    remain = args.n - len(picked)
    # Weighted polling by remaining pool size
    while remain > 0:
        progressed = False
        for c in cats:
            if remain <= 0:
                break
            pool = by_cat[c]
            if not pool:
                continue
            picked.append(pool.pop())
            remain -= 1
            progressed = True
        if not progressed:
            break
    if len(picked) > args.n:
        picked = picked[: args.n]
    rng.shuffle(picked)

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Copy the image used
    for r in picked:
        for rel in list(r.get("ref_relpaths") or []) + ([r["gt_relpath"]] if r.get("gt_relpath") else []):
            s = src / rel
            d = out / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            if s.exists() and not d.exists():
                shutil.copy2(s, d)

    with (out / "manifest.jsonl").open("w") as fp:
        for r in picked:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "n_samples": len(picked),
        "seed": args.seed,
        "src": str(src),
        "contact_levels": sorted(levels) if levels else None,
        "by_cat": {c: sum(1 for r in picked if r["cat"] == c) for c in cats},
        "note": "stratified subset (optional contact_c filter)",
    }
    (out / "pack_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
