#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panoptic GT stats: people-per-sequence and nearest inter-person joint distance (contact density).

Run after panoptic_gt_sweep.sh (mpie env). World units are centimeters.
Print a ranked table to choose which sequences / camera views to download in HD.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path.home() / "mpie_data/datasets/panoptic"
STEP = 30  # 30fps, samples per second1frame
GHOST_MEAN = 20.0  # Average distance of joints with the same name<20cm Condemned as double image of the same person(Ghosting is only a few seconds awaycm, There are also real people who can cling to each other30cm+)


def dedup(bodies):
    """legacy15 oldGTThere are tons of ghosting dummies(Repeated tagging by the same person): Merge according to the average distance of joints with the same name. """
    n = len(bodies)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for k in range(i + 1, n):
            a, b = bodies[i], bodies[k]
            ok = (a[:, 3] > 0.1) & (b[:, 3] > 0.1)
            if ok.sum() < 5:
                continue
            if float(np.linalg.norm(a[ok, :3] - b[ok, :3], axis=1).mean()) < GHOST_MEAN:
                parent[find(i)] = find(k)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    kept = []
    for idx in groups.values():
        best = max(idx, key=lambda i: (bodies[i][:, 3] > 0.1).sum())
        j = bodies[best]
        kept.append(j[j[:, 3] > 0.1][:, :3])
    return kept, n - len(kept)


rows = []
for seqdir in sorted(p for p in ROOT.iterdir() if p.is_dir()):
    pd = next((seqdir / n for n in ("hdPose3d_stage1_coco19", "hdPose3d_stage1")
               if (seqdir / n).is_dir()), None)
    if pd is None:
        continue
    files = sorted(pd.glob("body3DScene_*.json"))
    counts, mins = [], []
    n_ghost = n_body = 0
    for f in files[::STEP]:
        try:
            js = json.loads(f.read_text())
        except Exception:
            continue
        raw = []
        for b in js.get("bodies", []):
            j = np.array(b.get("joints19") or b.get("joints15"), dtype=float).reshape(-1, 4)
            if (j[:, 3] > 0.1).sum() >= 5:
                raw.append(j)
        ppl, g = dedup(raw)
        n_ghost += g
        n_body += len(raw)
        counts.append(len(ppl))
        if len(ppl) >= 2:
            mins.append(min(
                float(np.linalg.norm(a[:, None] - b_[None], axis=-1).min())
                for i, a in enumerate(ppl) for b_ in ppl[i + 1:]))
    if not counts:
        continue
    c = np.array(counts)
    m = np.array(mins) if mins else np.array([])
    rows.append({
        "seq": seqdir.name, "frames": len(files),
        "n_med": int(np.median(c)), "n_max": int(c.max()),
        "pct_ge2": round(100 * float((c >= 2).mean())),
        "pct_ge3": round(100 * float((c >= 3).mean())),
        "contact20": round(100 * float((m < 20).mean())) if len(m) else 0,
        "close45": round(100 * float((m < 45).mean())) if len(m) else 0,
        "ghost_pct": round(100 * n_ghost / max(n_body, 1)),
    })

rows.sort(key=lambda r: (-r["contact20"], -r["close45"], -r["pct_ge3"]))
print(f"{'seq':26s} {'Frames':>6s} Average among people/peoplemax  ≥2people%  ≥3people%  touch<20cm%  close<45cm%  ghosting%")
for r in rows:
    print(f"{r['seq']:26s} {r['frames']:6d} {r['n_med']:4d}/{r['n_max']:<4d} "
          f"{r['pct_ge2']:5d} {r['pct_ge3']:6d} {r['contact20']:9d} {r['close45']:10d} {r['ghost_pct']:5d}")

out = Path.home() / "panoptic_sweep_stats.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
print(f"\nAlready saved {out} — Paste the above table back, Certainly HD Video download list")
