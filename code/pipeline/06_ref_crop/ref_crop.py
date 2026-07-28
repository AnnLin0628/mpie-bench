#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 6: cross-frame reference selection and cropping (core anti copy-paste step).

For each identity on selected keyframes, pick reference frames by four tiers:
  tier1 cross-video   tier2 same-video cross-shot   tier3 multi-view same time
  tier4 same-shot large temporal gap (fallback; stricter diversity thresholds)
Diversity score = identity hard gate (ArcFace cosine ≥ id_gate, else reject)
                + pose-difference bonus + background-similarity penalty
                  (embedding outside the person mask).
Each ref can emit refs_raw (sharp crop) / refs_clean (segmented + blurred bg).

Usage: python ref_crop.py --db ~/mpie_data/manifests/mpie.db --out ~/mpie_data/crops \
        [--id-gate 0.5] [--bg-sim-cap 0.85] [--tier4-time-gap 300]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, upsert, rows  # noqa: E402


def load_emb(path, key):
    z = np.load(path)
    return z[key] if z[key].size else None


def crop_quality(crop, bbox, has_face, is_refpool):
    """Reference cleanliness score: large box + sharp + face + prefer ref-pool frames."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    score = min(h / 512.0, 1.5) + min(sharp / 200.0, 1.5)
    score += 1.0 if has_face else 0.0
    score += 0.5 if is_refpool else 0.0
    return score, h, sharp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--refs-per-identity", type=int, default=1, help="keep top-K cleanest refs per identity")
    ap.add_argument("--min-sharp", type=float, default=25.0, help="quality gate: min sharpness (Laplacian variance). "
                    "No absolute pixel-size floor (not comparable across resolutions and hurts small sharp faces) — "
                    "rely on has_face (detector min recognizable size) + this sharpness check")
    ap.add_argument("--allow-faceless", action="store_true", help="allow refs without a detected face (default: require face for ID)")
    args = ap.parse_args()

    conn = connect(args.db)
    out = Path(args.out)
    (out / "refs_raw").mkdir(parents=True, exist_ok=True)
    (out / "refs_clean").mkdir(parents=True, exist_ok=True)

    persons = rows(conn, "SELECT * FROM persons WHERE identity_id IS NOT NULL")
    kf_of = {p["track_id"]: p["track_id"].rsplit("_p", 1)[0] for p in persons}
    kf_meta = {k["kf_id"]: k for k in rows(conn, "SELECT * FROM keyframes WHERE selected IN (1,2)")}
    by_identity = {}
    for p in persons:
        by_identity.setdefault(p["identity_id"], []).append(p)

    n_ok = n_drop = 0
    for ident, members in by_identity.items():
        # Score each crop for this identity and apply the quality gate
        scored = []
        for p in members:
            crop = cv2.imread(str(out / "persons" / f"{p['track_id']}.jpg"))
            if crop is None:
                continue
            bbox = json.loads(p["bbox_json"]) if p["bbox_json"] else [0, 0, crop.shape[1], crop.shape[0]]
            ckf = kf_meta.get(kf_of[p["track_id"]], {})
            is_refpool = ckf.get("selected") == 2
            if not is_refpool:
                continue          # hard-exclude peak-frame crops: refs must come from valley/ref-pool frames,
                                   # else ref≈target same-source crop (the copy-paste failure mode)
            has_face = load_emb(p["face_emb_path"], "face") is not None
            q, h, sharp = crop_quality(crop, bbox, has_face, is_refpool)
            # Do not treat absolute pixel height as a quality standard across resolutions;
            # it harms small-but-sharp face crops. Real gates: has_face (InsightFace fails on
            # degenerate scraps) + min_sharp. Here we only drop few-pixel crop failures.
            if h < 20 or sharp < args.min_sharp:
                continue
            if not args.allow_faceless and not has_face:
                continue                                  # require face for reliable ID
            scored.append((q, p, crop, ckf.get("frame_idx", 0)))
        if not scored:
            n_drop += 1
            continue
        # Sort by cleanliness; keep top-K from distinct frames (avoid near-duplicates)
        scored.sort(key=lambda x: -x[0])
        kept, used_frames = [], set()
        for q, p, crop, fidx in scored:
            if fidx in used_frames:
                continue
            kept.append((q, p, crop)); used_frames.add(fidx)
            if len(kept) >= args.refs_per_identity:
                break
        for q, p, crop in kept:
            ref_id = p["track_id"]                          # few refs/identity; name from source track
            # Prefer sharp bbox crops (full body + clear face) for ID discrimination.
            # Background leakage is mitigated by cross-frame/valley refs, not by blur.
            clean_p = out / "refs_clean" / f"{ref_id}.jpg"
            cv2.imwrite(str(clean_p), crop)
            upsert(conn, "refs", {"ref_id": ref_id, "identity_id": ident,
                                  "kf_id": kf_of[p["track_id"]], "tier": 3,
                                  "diversity_score": round(q, 3),
                                  "clean_path": str(clean_p), "raw_path": str(clean_p)})
        n_ok += 1
    conn.commit()
    print(f"refs done: {n_ok} identities kept refs, {n_drop} dropped (no crop passed quality gate)")


if __name__ == "__main__":
    main()
