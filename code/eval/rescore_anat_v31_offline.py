#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Anat recompose with anat_v3.1 (A1 overdetect + A2 struct/ownership).

No GPU / Multi-HMR. Reads existing mesh_v3 JSON fields and recomposes S_anat_mesh.

Examples:
  # fig2 gallery rows (GT + gens in pack)
  python rescore_anat_v31_offline.py \\
    --pack ~/mpie_testset_pack \\
    --sample-ids hug__ece68b23998b__T5,piggyback__946806af5ed9__T4,piggyback__1469fe1a6428__T2,wrestle_grapple__28f54bd775cc__T9 \\
    --models gpt-image-2,gemini-3-pro-image,flux1-kontext-dev,omnigen2,uno,ace,_gt \\
    --out $MPIE_ROOT/analysis/figures/fig_fail_assets/anat_v31_compare.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from anat_extended import compose_anat_score


def _recompose(rec: Dict[str, Any]) -> Dict[str, Any]:
    n_exp = rec.get("n_expected")
    n_raw = rec.get("n_detected_raw")
    composed = compose_anat_score(
        s_residual=rec.get("S_anat_residual"),
        s_overcount=rec.get("S_anat_overcount"),
        s_scale=rec.get("S_anat_scale"),
        s_ownership=rec.get("S_anat_ownership"),
        s_part_mesh=rec.get("S_anat_part_mesh"),
        s_person=rec.get("S_anat_person"),
        s_abhuman=rec.get("S_anat_abhuman"),
        under_detect=bool(rec.get("under_detect")),
        recon_fail=bool(rec.get("recon_fail")) or not bool(rec.get("ok", True)),
        n_detected_raw=int(n_raw) if n_raw is not None else None,
        n_expected=int(n_exp) if n_exp is not None else None,
        leftover_frac=rec.get("anat_leftover_frac"),
        n_leftover_blobs=rec.get("anat_n_leftover_blobs"),
    )
    return composed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--sample-ids", required=True, help="comma-separated")
    ap.add_argument(
        "--models",
        default="gpt-image-2,gemini-3-pro-image,flux1-kontext-dev,omnigen2,uno,ace,_gt",
    )
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--write-json-dir",
        default="",
        help="optional: write recomposed copies under this root/<model>/<sid>.json",
    )
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    sids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    rows_out: List[Dict[str, Any]] = []

    print(
        f"{'sid':40s} {'model':22s} {'Anat_old':>8s} {'Anat_v31':>8s} {'Δ':>7s} "
        f"{'n_raw':>5s} {'ratio':>5s} {'Pdet':>5s} {'Pstr':>5s} {'Pext':>5s}"
    )
    for sid in sids:
        for mid in models:
            p = pack / "judgments" / "mesh_v3" / mid / f"{sid}.json"
            if not p.is_file():
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            old = rec.get("S_anat_mesh")
            inter = rec.get("S_inter_mesh")
            composed = _recompose(rec)
            new = composed["S_anat_mesh"]
            try:
                dlt = float(new) - float(old)
            except (TypeError, ValueError):
                dlt = None
            n_raw = composed.get("n_detected_raw")
            n_exp = composed.get("n_expected")
            ratio = composed.get("detect_ratio")
            print(
                f"{sid:40s} {mid:22s} "
                f"{float(old):8.3f} {float(new):8.3f} "
                f"{(dlt if dlt is not None else 0):+7.3f} "
                f"{str(n_raw):>5s} "
                f"{(f'{ratio:.2f}' if ratio is not None else '—'):>5s} "
                f"{composed['P_anat_detect']:5.2f} "
                f"{composed['P_anat_struct']:5.2f} "
                f"{composed['P_anat_extra']:5.2f}"
            )
            rows_out.append(
                {
                    "sample_id": sid,
                    "model": mid,
                    "S_anat_old": old,
                    "S_anat_v31": new,
                    "delta_anat": dlt,
                    "S_inter_mesh": inter,
                    "n_detected_raw": n_raw,
                    "n_expected": n_exp,
                    "detect_ratio": ratio,
                    "P_anat_detect": composed["P_anat_detect"],
                    "P_anat_struct": composed["P_anat_struct"],
                    "P_anat_extra": composed["P_anat_extra"],
                    "P_anat_resid": composed["P_anat_resid"],
                    "anat_protocol": composed.get("anat_protocol"),
                    "anat_formula": composed.get("anat_formula"),
                }
            )
            if args.write_json_dir:
                out_p = Path(args.write_json_dir).expanduser() / mid / f"{sid}.json"
                out_p.parent.mkdir(parents=True, exist_ok=True)
                out_rec = dict(rec)
                out_rec.update(composed)
                out_rec["S_anat_mesh_v3_legacy"] = old
                out_p.write_text(
                    json.dumps(out_rec, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        if rows_out:
            with out.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
                w.writeheader()
                w.writerows(rows_out)
            print(f"wrote {out} ({len(rows_out)} rows)")
    else:
        print(f"n={len(rows_out)} (pass --out to save csv)")


if __name__ == "__main__":
    main()
