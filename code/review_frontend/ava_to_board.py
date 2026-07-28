#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AVA upload -> 8080 scene board format (same as panoptic).

Input: $MPIE_ROOT/data/raw/ava_review_stage/ava_review/<movie>/
        targets_pick/*.jpg  refs_clean/*.jpg  ref_clusters.json({gi:[{fn,face_h,det}]})
Output: $MPIE_ROOT/data/cc0_review_full/ava/
        flat/               hard-linked jpgs (same disk, zero copy)
        ref_clusters.json   {"groups":[{"avg_sim",members:[{video_id,thumb}]}],"no_face":[]}
Scene = movie (actors do not cross movies; one scene per movie).
avg_sim nominal 0.56: clustered at 0.55 on a GPU host, top-12 per group,
does not trigger board MAX_GROUP=12 / MIN_SIM=0.32 split guard.
"""
import json
import os
from pathlib import Path

SRC = Path.home() / "mpie_bench/data/raw/ava_review_stage/ava_review"
DST = Path.home() / "mpie_bench/data/cc0_review_full/ava"
FLAT = DST / "flat"
FLAT.mkdir(parents=True, exist_ok=True)

groups, n_t, n_r, dup = [], 0, 0, 0
movies = sorted(p.name for p in SRC.iterdir() if p.is_dir())
for movie in movies:
    md = SRC / movie
    for p in sorted((md / "targets_pick").glob("*.jpg")):
        d = FLAT / p.name
        if d.exists():
            dup += 1
        else:
            os.link(p, d)
        n_t += 1
    clus = json.loads((md / "ref_clusters.json").read_text())
    for gi in sorted(clus, key=int):
        members = []
        for m in clus[gi]:
            p = md / "refs_clean" / m["fn"]
            if not p.exists():
                continue
            d = FLAT / m["fn"]
            if not d.exists():
                os.link(p, d)
            members.append({"video_id": movie, "thumb": m["fn"]})
            n_r += 1
        if members:
            groups.append({"avg_sim": 0.56, "members": members})

(DST / "ref_clusters.json").write_text(json.dumps({"groups": groups, "no_face": []}, ensure_ascii=False))
print(f"movies {len(movies)} | targets {n_t} (dup names {dup}) | refs {n_r} | actor groups {len(groups)}")
print(f"flat files: {len(list(FLAT.glob('*.jpg')))}")
