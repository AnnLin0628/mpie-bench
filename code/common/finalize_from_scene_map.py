#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export the scene board scene map Assemble into final data set final/<dataset>/。

Suitable for cluster-free(GTidentity)And the review diagrams are all in a single flat/ Catalog of data sets: CHI3D / Panoptic / EgoHumans。
(CC0 Walk apply_scene_map.py fall sqlite Library, on GPU shards upper assembly, This script is not used.)

product structure and final/harmony4d Alignment:
  final/<dataset>/refs/     Clean reference images of each actor's stars
  final/<dataset>/targets/  Review all retained target maps
  final/<dataset>/manifest.json  scene -> actors(Star reference) -> targets(+bindings)

self contained(Standard library only). usage:
  python finalize_from_scene_map.py --dataset chi3d \
      --map ~/cc0_scene_map_chi3d.json \
      --flat $MPIE_ROOT/data/cc0_review_full/chi3d/flat \
      --out  $MPIE_ROOT/data/final/chi3d
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--flat", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    flat = Path(args.flat).expanduser()
    out = Path(args.out).expanduser()
    smap = json.loads(Path(args.map).expanduser().read_text())

    if out.exists():
        sys.exit(f"{out} Already exists, Remove manually first/Delete and run again(Prevent false coverage)")
    (out / "refs").mkdir(parents=True)
    (out / "targets").mkdir()

    scenes_out, missing = [], []

    def take(fn, sub):
        src = flat / fn
        if not src.exists():
            missing.append(fn)
            return
        shutil.copy2(src, out / sub / fn)

    for sc in smap["scenes"]:
        if sc.get("merged_into"):        # Scenarios that cross categories and go together are classified into the target category, Don't fall here
            continue
        actors = {}
        for a in sc["actors"]:
            star = a.get("star") or (a["refs"][0] if a["refs"] else None)
            if not star:
                continue
            actors[a["label"]] = [star]
            take(star, "refs")
        for fn in sc["targets"]:
            take(fn, "targets")
        scenes_out.append({
            "scene": sc["videos"][0],
            "videos": sc["videos"],
            "actors": actors,
            "targets": sorted(sc["targets"]),
            "bindings": sc.get("bindings") or {},
        })

    n_refs = sum(len(v) for s in scenes_out for v in s["actors"].values())
    n_targets = sum(len(s["targets"]) for s in scenes_out)
    manifest = {
        "dataset": args.dataset,
        "format": "scene(cast) -> actors(Starred clean reference image) -> targets(Review retention target map, bindings=Actors appearing in the target image)",
        "source_map": Path(args.map).name,
        "n_scenes": len(scenes_out),
        "n_refs": n_refs,
        "n_targets": n_targets,
        "scenes": scenes_out,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    if missing:
        print(f"⚠ {len(missing)} files in flat Can't find it: {missing[:5]} ...")
    print(f"{args.dataset}: {len(scenes_out)}scene {n_refs}reference {n_targets}Target -> {out}")


if __name__ == "__main__":
    main()
