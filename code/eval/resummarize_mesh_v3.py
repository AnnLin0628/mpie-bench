#!/usr/bin/env python3
"""Rebuild _summary.json from existing per-sample judgments (no HMR / no rewrite)."""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_mesh_inter import summarize

def main(root: Path) -> None:
    root = Path(root)
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        recs = []
        for p in sorted(sub.glob("*.json")):
            if p.name.startswith("_"):
                continue
            recs.append(json.loads(p.read_text()))
        summary = summarize(recs)
        (sub / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(sub.name, {
            k: summary.get(k)
            for k in (
                "n_ok", "S_anat_mesh", "S_anat_person_mean", "S_anat_scale_mean",
                "S_anat_hand_mean", "S_anat_part_mesh_mean", "S_anat_residual_mean",
                "S_inter_mesh", "recon_fail_rate",
            )
        })

if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else
              Path.home() / "mpie_testset_pack/judgments/mesh_v3"))
