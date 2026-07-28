#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""merge panoptic_extract.py --tag Multi-card sharding ref_clusters/meta, Review tar。

flat/ The file names of each fragment naturally do not conflict., Just spell json. usage, After all the parts have been run,):
  python panoptic_merge_shards.py
"""
import json
import subprocess
from pathlib import Path

OUT = Path.home() / "mpie_data/panoptic_review/panoptic"

groups, no_face, meta = [], [], {}
shards = sorted(OUT.glob("ref_clusters_*.json"))
if not shards:
    raise SystemExit("No ref_clusters_<tag>.json Sharding")
for rc in shards:
    d = json.loads(rc.read_text())
    groups += d["groups"]
    no_face += d.get("no_face", [])
    mt = OUT / rc.name.replace("ref_clusters_", "meta_")
    meta.update(json.loads(mt.read_text()))
    print(f"{rc.name}: cast{len(d['groups'])}")

(OUT / "ref_clusters.json").write_text(json.dumps({"groups": groups, "no_face": no_face}, ensure_ascii=False))
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
for rc in shards:
    (OUT / rc.name.replace("ref_clusters_", "meta_")).unlink(missing_ok=True)
    rc.unlink()

n_ref = sum(1 for v in meta.values() if v["kind"] == "ref")
n_tgt = sum(1 for v in meta.values() if v["kind"] == "target")
sess = sorted({v["sess"] for v in meta.values()})
print(f"merge: {len(sess)}sequence cast{len(groups)} reference{n_ref} Target{n_tgt}")

tar = Path.home() / "panoptic_review.tar"
subprocess.run(["tar", "cf", str(tar), "-C", str(OUT.parent), OUT.name], check=True)
subprocess.run(f"md5sum {tar} > {tar}.md5", shell=True, check=True)
print(f"Pack: {tar} (+.md5)  passSGback: bash ingest_review_pkg.sh panoptic {tar.name}")
