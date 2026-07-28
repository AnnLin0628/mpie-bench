#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exist Development machine Top: Put the complete test set prompt + Printed with reference picture GPU host Available pack。

usage(SG,wait prompt After completion):
  python export_pack.py --out ~/mpie_testset_pack
  # Sync to GPU host example:
  # rsync -avP ~/mpie_testset_pack/ USER@gpu-host:~/mpie_testset_pack/

Each sample = One corresponding to a target picture prompt（N≈2500）。
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

HOME = Path.home()
ROOT = HOME / "mpie_bench/data/cc0_review_full"
PROMPT_DIR = HOME / "mpie_bench/data/manifests/prompts_full"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HOME / "mpie_testset_pack"))
    ap.add_argument("--require-complete", action="store_true",
                    help="The scene target map must be complete before exporting; by default, all existing prompt of target")
    args = ap.parse_args()
    out = Path(args.out).expanduser().resolve()
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped_err = 0
    for f in sorted(PROMPT_DIR.glob("prompts_*.json")):
        for item in json.loads(f.read_text()):
            cat, anchor = item["cat"], item["anchor"]
            vlm = item.get("vlm") or {}
            if "error" in vlm:
                skipped_err += 1
                continue
            refs = item.get("refs") or []
            tgts = item.get("tgts") or []
            tmap = {f"T{i+1}": fn for i, fn in enumerate(tgts)}
            # copy refs once per scene
            ref_rel = []
            for i, fn in enumerate(refs):
                src = ROOT / cat / "flat" / fn
                if not src.exists():
                    print(f"WARN missing ref {src}")
                    continue
                rel = f"images/{cat}/{anchor}/R{i+1}_{fn}"
                dst = out / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    shutil.copy2(src, dst)
                ref_rel.append(rel)
            for t in vlm.get("targets") or []:
                tid = t.get("target")
                fn = tmap.get(tid, "")
                prompt = (t.get("prompt") or "").strip()
                if not prompt or not fn:
                    continue
                # optional GT for later eval
                src_t = ROOT / cat / "flat" / fn
                gt_rel = f"images/{cat}/{anchor}/GT_{fn}"
                if src_t.exists():
                    dst_t = out / gt_rel
                    if not dst_t.exists():
                        shutil.copy2(src_t, dst_t)
                sample_id = f"{cat}__{anchor}__{tid}"
                rows.append({
                    "sample_id": sample_id,
                    "cat": cat,
                    "anchor": anchor,
                    "target": tid,
                    "target_file": fn,
                    "prompt": prompt,
                    "ref_relpaths": ref_rel,
                    "gt_relpath": gt_rel if src_t.exists() else "",
                    "confidence": t.get("confidence"),
                    "flag_underage": t.get("flag_underage"),
                })

    man = out / "manifest.jsonl"
    with man.open("w") as fp:
        for r in rows:
            fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        "n_samples": len(rows),
        "n_cats": len({r["cat"] for r in rows}),
        "skipped_error_scenes": skipped_err,
        "prompt_dir": str(PROMPT_DIR),
    }
    (out / "pack_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    by_cat = defaultdict(int)
    for r in rows:
        by_cat[r["cat"]] += 1
    print(json.dumps({"wrote": str(man), **meta, "by_cat": dict(by_cat)}, ensure_ascii=False, indent=2))
    if args.require_complete and len(rows) < 2500:
        raise SystemExit(f"only {len(rows)} samples (<2500); wait for prompt job or drop --require-complete")


if __name__ == "__main__":
    main()
