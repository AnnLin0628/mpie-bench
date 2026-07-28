#!/usr/bin/env python3
"""Mean ArcFace S_id conditioned on face_visible_rate == 1 (closed-source focus).

Usage (CPU only):
  python fvr_conditional_id.py --pack ~/mpie_testset_pack
  python fvr_conditional_id.py --pack ~/mpie_testset_pack --models gpt-image-2,gemini-3-pro-image,seedream-5-pro
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CLOSED_DEFAULT = (
    "gpt-image-2,gemini-3-pro-image,seedream-5-pro"
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--models", default=CLOSED_DEFAULT)
    ap.add_argument("--fvr-min", type=float, default=0.9999)
    args = ap.parse_args()
    pack = Path(args.pack)
    arc = pack / "judgments" / "arcface_v1"
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    print(f"# FVR-conditional ID  pack={pack}")
    print(f"# filter: face_visible_rate >= {args.fvr_min}")
    print("| model | n_all | n_FVR1 | mean S_id (all) | mean S_id (FVR=1) | mean matched_sim (FVR=1) |")
    print("|---|---:|---:|---:|---:|---:|")

    for mid in models:
        d = arc / mid
        if not d.is_dir():
            print(f"| {mid} | missing | | | | |")
            continue
        all_ids, fvr1_ids, fvr1_msim = [], [], []
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("_"):
                continue
            j = json.loads(p.read_text())
            sid = j.get("S_id")
            fvr = j.get("face_visible_rate")
            if sid is None or fvr is None:
                continue
            sid_f = float(sid)
            fvr_f = float(fvr)
            all_ids.append(sid_f)
            if fvr_f >= args.fvr_min:
                fvr1_ids.append(sid_f)
                ms = j.get("matched_similarity")
                if ms is not None:
                    fvr1_msim.append(float(ms))
        def mean(xs):
            return round(sum(xs) / len(xs), 4) if xs else None
        print(
            f"| {mid} | {len(all_ids)} | {len(fvr1_ids)} | "
            f"{mean(all_ids)} | {mean(fvr1_ids)} | {mean(fvr1_msim)} |"
        )


if __name__ == "__main__":
    main()
