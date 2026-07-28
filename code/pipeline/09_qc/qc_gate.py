#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 9: pre-assembly QC gate — block bad keyframes before Stage 10 samples.

Rules (all thresholds configurable):
  R1 missing caption / confidence=low / flag_underage=1 → block
  R2 caption vs detect person count mismatch → block (needs_review)
  R3 unreadable target or short side < min-side → block
  R4 fewer than 2 identities with usable refs (tier≤max-tier) → block
Blocked frames: captions.needs_review=1 (underage in R1 also sets selected=0).
Print per-rule block counts for QA reconciliation.

Usage: python qc_gate.py --db ~/mpie_data/manifests/mpie.db [--min-side 512]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-side", type=int, default=512)
    ap.add_argument("--count-tol", type=int, default=0)
    ap.add_argument("--max-tier", type=int, default=4)
    args = ap.parse_args()
    conn = connect(args.db)

    stats = {"pass": 0, "r1_caption": 0, "r1_underage": 0, "r2_count": 0,
             "r3_image": 0, "r4_refs": 0}
    kfs = rows(conn, "SELECT * FROM keyframes WHERE selected=1")
    for kf in kfs:
        cap = rows(conn, "SELECT * FROM captions WHERE kf_id=?", (kf["kf_id"],))
        cap = cap[0] if cap else None

        def block(rule):
            stats[rule] += 1
            conn.execute("UPDATE captions SET needs_review=1 WHERE kf_id=?",
                         (kf["kf_id"],))

        # R1: caption hard checks
        if cap and cap["flag_underage"]:
            stats["r1_underage"] += 1
            conn.execute("UPDATE keyframes SET selected=0 WHERE kf_id=?",
                         (kf["kf_id"],))  # Minors photographed: Directly out of the warehouse, No review left
            continue
        if not cap or (cap["confidence"] or "").lower() == "low":
            if cap:
                block("r1_caption")
            else:
                stats["r1_caption"] += 1
            continue
        # R2: Headcount consistency(caption number of people judging vs YOLO Number of people detected)
        if abs((cap["n_person"] or 0) - (kf["n_person"] or 0)) > args.count_tol:
            block("r2_count")
            continue
        # R3: The target frame is readable and the resolution is up to standard
        img = cv2.imread(kf["frame_path"])
        if img is None or min(img.shape[:2]) < args.min_side:
            block("r3_image")
            continue
        # R4: ≥2 Personal identity has a qualified reference picture(The reference picture cannot be from this frame)
        n_ok = rows(conn, """
            SELECT COUNT(DISTINCT p.identity_id) c FROM persons p
            JOIN refs r ON r.identity_id = p.identity_id
            WHERE p.track_id LIKE ? AND r.kf_id != ? AND r.tier <= ?""",
            (kf["kf_id"] + "_p%", kf["kf_id"], args.max_tier))[0]["c"]
        if n_ok < 2:
            block("r4_refs")
            continue
        stats["pass"] += 1
        conn.execute("UPDATE captions SET needs_review=0 WHERE kf_id=?",
                     (kf["kf_id"],))
    conn.commit()
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
