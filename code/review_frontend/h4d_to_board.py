#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harmony4D final -> 8080 scene board format (same path as ava/panoptic).

Input: $MPIE_ROOT/data/final/harmony4d/ (manifest.json + refs/ + targets/)
Output: $MPIE_ROOT/data/cc0_review_full/harmony4d/
        flat/                hard-linked jpgs renamed to board _f/_r convention
        ref_clusters.json    {"groups":[{"avg_sim",members:[{video_id,thumb}]}],"no_face":[]}
        board_name_map.json  board filename -> final original (delete sync)
Scenes = manifest action scenes (8); video id = scene name; underscores → "-" to avoid
gen_scene_board _r/_f split bugs. Board deletes are bookkeeping; map back to final on apply.
"""
import json
import os
import shutil
from pathlib import Path

SRC = Path.home() / "mpie_bench/data/final/harmony4d"
DST = Path.home() / "mpie_bench/data/cc0_review_full/harmony4d"
FLAT = DST / "flat"
if FLAT.exists():
    shutil.rmtree(FLAT)
FLAT.mkdir(parents=True)

manifest = json.loads((SRC / "manifest.json").read_text())


def token(orig):
    """h4d_001_ballroom__t0.jpg -> 001-ballroom--t0 (no _r/_f substring, reversible)."""
    return orig[:-4].removeprefix("h4d_").replace("_", "-")


groups, name_map, n_t, n_r = [], {}, 0, 0
for sc in manifest["scenes"]:
    scene = sc["scene"]
    for aid, refs in sorted(sc["actors"].items()):
        members = []
        for orig in refs:
            fn = f"{scene}_r{token(orig)}.jpg"
            os.link(SRC / "refs" / orig, FLAT / fn)
            name_map[fn] = orig
            members.append({"video_id": scene, "thumb": fn})
            n_r += 1
        if members:
            groups.append({"avg_sim": 0.56, "members": members})
    for orig in sc["targets"]:
        fn = f"{scene}_f{token(orig)}.jpg"
        os.link(SRC / "targets" / orig, FLAT / fn)
        name_map[fn] = orig
        n_t += 1

(DST / "ref_clusters.json").write_text(json.dumps({"groups": groups, "no_face": []}, ensure_ascii=False))
(DST / "board_name_map.json").write_text(json.dumps(name_map, ensure_ascii=False, indent=1))
print(f"scenes {len(manifest['scenes'])} | targets {n_t} | refs {n_r} | actor groups {len(groups)}")
print(f"flat files: {len(list(FLAT.glob('*.jpg')))}")
