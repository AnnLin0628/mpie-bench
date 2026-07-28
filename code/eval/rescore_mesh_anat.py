#!/usr/bin/env python3
"""Rescore S_anat_mesh with residual-heavy penalty formula (no Multi-HMR).

Uses cached S_anat_residual / scale / ownership / part_mesh / under_detect.
Does NOT recompute residual from pixels — for that, FORCE re-run score_mesh_v3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anat_extended import compose_anat_score  # noqa: E402
from rescore_mesh_inter import summarize  # noqa: E402


def rescore_anat(rec: dict) -> dict:
    if not rec.get("ok"):
        return rec
    if rec.get("recon_fail"):
        out = dict(rec)
        out["S_anat_mesh"] = 0.0
        out["P_anat_resid"] = 1.0
        out["P_anat_detect"] = 1.0
        out["anat_formula"] = "recon_fail→0"
        return out

    s_resid = rec.get("S_anat_residual")
    # need residual (or person) to apply new formula; else leave Anat untouched
    if s_resid is None and rec.get("S_anat_person") is None:
        return rec

    composed = compose_anat_score(
        s_residual=float(s_resid) if s_resid is not None else None,
        s_overcount=rec.get("S_anat_overcount"),
        s_scale=rec.get("S_anat_scale"),
        s_ownership=rec.get("S_anat_ownership"),
        s_part_mesh=rec.get("S_anat_part_mesh"),
        s_person=rec.get("S_anat_person"),
        s_abhuman=rec.get("S_anat_abhuman"),
        under_detect=bool(rec.get("under_detect")),
        recon_fail=False,
    )
    out = dict(rec)
    out.update(composed)
    return out


def main(root: Path) -> None:
    root = Path(root)
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        recs: List[dict] = []
        n_upd = 0
        for p in sorted(sub.glob("*.json")):
            if p.name.startswith("_"):
                continue
            old = json.loads(p.read_text())
            new = rescore_anat(old)
            if new.get("S_anat_mesh") != old.get("S_anat_mesh") or new.get(
                "anat_formula"
            ) != old.get("anat_formula"):
                n_upd += 1
            p.write_text(json.dumps(new, ensure_ascii=False, indent=2))
            recs.append(new)
        summary = summarize(recs)
        # enrich summary with residual / penalty means
        def mean(key: str) -> Optional[float]:
            xs = [
                float(r[key])
                for r in recs
                if r.get("ok") and r.get(key) is not None
            ]
            return float(sum(xs) / len(xs)) if xs else None

        summary["S_anat_residual_mean"] = mean("S_anat_residual")
        summary["P_anat_resid_mean"] = mean("P_anat_resid")
        summary["P_anat_struct_mean"] = mean("P_anat_struct")
        summary["P_anat_detect_mean"] = mean("P_anat_detect")
        (sub / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            sub.name,
            {
                "n_upd": n_upd,
                "S_anat_mesh": summary.get("S_anat_mesh"),
                "S_anat_residual_mean": summary.get("S_anat_residual_mean"),
                "S_inter_mesh": summary.get("S_inter_mesh"),
                "recon_fail_rate": summary.get("recon_fail_rate"),
            },
        )


if __name__ == "__main__":
    main(
        Path(
            sys.argv[1]
            if len(sys.argv) > 1
            else Path.home() / "mpie_testset_pack/judgments/mesh_v3"
        )
    )
