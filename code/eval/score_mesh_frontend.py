#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score Anat/Inter with a selectable mesh frontend (Multi-HMR / HMR2 / SMPLer-X).

Writes the same JSON schema as score_mesh_v3.py into:
  $PACK/judgments/<judge-dir>/<model_id>/<sample_id>.json

Examples:
  # Multi-HMR baseline (same as score_mesh_v3)
  python score_mesh_frontend.py --pack $PACK --frontend multi_hmr \\
    --multihmr-repo ~/models/multi-hmr --model-id flux1-kontext-dev

  # HMR 2.0 second frontend
  conda activate hmr2
  python score_mesh_frontend.py --pack $PACK --frontend hmr2 \\
    --hmr2-repo ~/models/4D-Humans --model-id flux1-kontext-dev

  # SMPLer-X third frontend
  conda activate smplerx
  python score_mesh_frontend.py --pack $PACK --frontend smpler_x \\
    --smplerx-repo ~/models/SMPLer-X --smplerx-ckpt ~/mpie_weights/smpler_x/xxx.pth \\
    --model-id flux1-kontext-dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from mesh_frontends import build_frontend
from score_mesh_v3 import (
    atomic_write,
    calibrate_inter_params,
    load_manifest,
    rescore_inter_scalars,
    score_one,
    select_top_k_humans,  # noqa: F401  (kept for import side-effects / debug)
)
from rescore_mesh_inter import summarize


def default_judge_dir(frontend: str) -> str:
    f = frontend.strip().lower().replace("-", "_")
    if f in ("multi_hmr", "multihmr", "mhmr"):
        return "mesh_v3"
    if f in ("hmr2", "hmr_2", "4dhumans"):
        return "mesh_hmr2"
    if f in ("smpler_x", "smplerx"):
        return "mesh_smplerx"
    return f"mesh_{f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument(
        "--frontend",
        default="multi_hmr",
        help="multi_hmr | hmr2 | smpler_x",
    )
    ap.add_argument("--judge-dir", default="", help="default mesh_v3 / mesh_hmr2 / mesh_smplerx")
    ap.add_argument("--multihmr-repo", default=os.environ.get("MULTIHMR_REPO", ""))
    ap.add_argument("--hmr2-repo", default=os.environ.get("HMR2_REPO", ""))
    ap.add_argument("--smplerx-repo", default=os.environ.get("SMPLERX_REPO", ""))
    ap.add_argument("--smplerx-ckpt", default=os.environ.get("SMPLERX_CKPT", ""))
    ap.add_argument("--backend-model", default="multiHMR_896_L")
    ap.add_argument("--model-id", default="", help="outputs/<model_id>/")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--gt-only", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--det-thresh", type=float, default=0.2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-volume", action="store_true")
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    judge = args.judge_dir or default_judge_dir(args.frontend)
    rows = load_manifest(pack)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
        print(f"[{args.frontend}] shard {args.shard_id}/{args.num_shards} → {len(rows)}", flush=True)

    # reuse Multi-HMR calibration when present (shared geometry thresholds)
    cal_path = pack / "judgments" / "mesh_v3" / "_calibration.json"
    cal: Dict[str, Any] = {
        "tau_pen": 0.05,
        "tau_vol": 0.05,
        "tau_contact": 0.02,
        "vol_ok": 0.05,
        "vol_bad": 0.15,
        "d_good": 0.05,
        "d_fail": 0.40,
    }
    if cal_path.is_file():
        raw = json.loads(cal_path.read_text())
        for k in ("tau_pen", "tau_vol", "tau_contact", "vol_ok", "vol_bad", "d_good", "d_fail"):
            if k in raw and raw[k] is not None:
                cal[k] = float(raw[k])

    backend = build_frontend(
        args.frontend,
        multihmr_repo=args.multihmr_repo or None,
        hmr2_repo=args.hmr2_repo or None,
        smplerx_repo=args.smplerx_repo or None,
        smplerx_ckpt=args.smplerx_ckpt or None,
        backend_model=args.backend_model,
        det_thresh=args.det_thresh,
    )
    use_volume = not args.no_volume

    def score_and_tag(img: Path, row: dict, model_id: str) -> dict:
        r = score_one(
            backend, img, row, cal=cal, model_id=model_id, use_volume=use_volume
        )
        r["backend"] = getattr(backend, "name", args.frontend)
        r["backend_model"] = getattr(backend, "model_name", args.frontend)
        r["judge_dir"] = judge
        return r

    # --- GT ---
    if args.gt_only or args.calibrate:
        out_dir = pack / "judgments" / judge / "_gt"
        recs = []
        for row in rows:
            rel = row.get("gt_relpath")
            if not rel:
                continue
            img = pack / rel
            out_p = out_dir / f"{row['sample_id']}.json"
            if out_p.exists() and not args.force:
                recs.append(json.loads(out_p.read_text()))
                continue
            print(f"[{args.frontend}] GT {row['sample_id']}", flush=True)
            r = score_and_tag(img, row, "_gt")
            atomic_write(out_p, r)
            recs.append(r)
        summary = summarize(recs)
        atomic_write(out_dir / "_summary.json", summary)
        if args.calibrate and judge == "mesh_v3":
            ok_recs = [r for r in recs if r.get("ok") and r.get("pen_vert_ratio") is not None]
            cal2 = calibrate_inter_params(ok_recs)
            cal.update(cal2)
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write(
                cal_path,
                {
                    **cal,
                    "n_gt": len(ok_recs),
                    "frontend": args.frontend,
                    "written_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
            for r in recs:
                if r.get("ok"):
                    rr = rescore_inter_scalars(r, cal)
                    atomic_write(out_dir / f"{rr['sample_id']}.json", rr)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    # --- models ---
    if args.all_models:
        model_ids = sorted(
            p.name
            for p in (pack / "outputs").iterdir()
            if p.is_dir() and not p.name.startswith("_")
        )
    elif args.model_id:
        model_ids = [args.model_id]
    else:
        raise SystemExit("pass --model-id or --all-models (or --gt-only)")

    for mid in model_ids:
        out_dir = pack / "judgments" / judge / mid
        img_dir = pack / "outputs" / mid
        if not img_dir.is_dir():
            print(f"SKIP missing outputs/{mid}", flush=True)
            continue
        recs: List[dict] = []
        for row in rows:
            sid = row["sample_id"]
            img = img_dir / f"{sid}.png"
            out_p = out_dir / f"{sid}.json"
            if out_p.exists() and not args.force:
                try:
                    recs.append(json.loads(out_p.read_text()))
                    continue
                except Exception:
                    pass
            if not img.is_file():
                continue
            print(f"[{args.frontend}] {mid} {sid}", flush=True)
            r = score_and_tag(img, row, mid)
            atomic_write(out_p, r)
            recs.append(r)
            print(
                json.dumps(
                    {
                        "mid": mid,
                        "sid": sid,
                        "ok": r.get("ok"),
                        "S_anat": r.get("S_anat_mesh"),
                        "S_inter": r.get("S_inter_mesh"),
                        "n": r.get("n_humans"),
                        "err": (r.get("error") or "")[:120],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        # only summarize this shard's records; full summary via aggregate later
        if args.num_shards <= 1:
            atomic_write(out_dir / "_summary.json", summarize(recs))
        else:
            atomic_write(
                out_dir / f"_summary_shard{args.shard_id}.json", summarize(recs)
            )


if __name__ == "__main__":
    # ensure code/eval on path when launched from elsewhere
    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    main()
