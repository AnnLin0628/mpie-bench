#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full-pack offline recompose: mesh_v3 → mesh_v31 (Anat+Inter v3.1).

Does NOT overwrite judgments/mesh_v3. Writes copies under judgments/mesh_v31/
with legacy scores retained as S_*_mesh_v3_legacy.

No GPU / Multi-HMR. Requires intermediate fields already in mesh_v3 JSON
(anat_leftover_frac, pen_inside_ratio, n_detected_raw, …).

Example:
  python rescore_mesh_v31_full.py \\
    --pack ~/mpie_testset_pack \\
    --models gpt-image-2,gemini-3-pro-image,seedream-5-pro,flux1-kontext-dev,dreamo,omnigen2,uno,ace,bagel,firered
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anat_extended import compose_anat_score  # noqa: E402
from mesh_metrics import compose_inter_score  # noqa: E402
from rescore_mesh_inter import summarize  # noqa: E402

PAPER_MODELS = [
    "gpt-image-2",
    "gemini-3-pro-image",
    "seedream-5-pro",
    "flux1-kontext-dev",
    "dreamo",
    "omnigen2",
    "uno",
    "ace",
    "bagel",
    "firered",
]


def _intent_from_rec(rec: Dict[str, Any]) -> Optional[str]:
    if rec.get("contact_intent"):
        return str(rec["contact_intent"])
    regime = str(rec.get("inter_regime") or "")
    if "required" in regime:
        return "required"
    if "forbidden" in regime:
        return "forbidden"
    if "unspecified" in regime:
        return "unspecified"
    return None


def recompose_v31(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Return full judgment dict with v3.1 Anat/Inter filled in."""
    out = dict(rec)
    # freeze legacy scores once
    if "S_anat_mesh_v3_legacy" not in out and rec.get("S_anat_mesh") is not None:
        out["S_anat_mesh_v3_legacy"] = rec.get("S_anat_mesh")
    if "S_inter_mesh_v3_legacy" not in out and rec.get("S_inter_mesh") is not None:
        out["S_inter_mesh_v3_legacy"] = rec.get("S_inter_mesh")

    anat = compose_anat_score(
        s_residual=rec.get("S_anat_residual"),
        s_overcount=rec.get("S_anat_overcount"),
        s_scale=rec.get("S_anat_scale"),
        s_ownership=rec.get("S_anat_ownership"),
        s_part_mesh=rec.get("S_anat_part_mesh"),
        s_person=rec.get("S_anat_person"),
        s_abhuman=rec.get("S_anat_abhuman"),
        under_detect=bool(rec.get("under_detect")),
        recon_fail=bool(rec.get("recon_fail")) or not bool(rec.get("ok", True)),
        n_detected_raw=int(rec["n_detected_raw"])
        if rec.get("n_detected_raw") is not None
        else None,
        n_expected=int(rec["n_expected"]) if rec.get("n_expected") is not None else None,
        leftover_frac=rec.get("anat_leftover_frac"),
        n_leftover_blobs=rec.get("anat_n_leftover_blobs"),
    )
    intent = _intent_from_rec(rec)
    inter = compose_inter_score(
        needs_contact=bool(rec.get("needs_contact"))
        if rec.get("needs_contact") is not None
        else (intent == "required"),
        contact_intent=intent,
        min_surf_dist=float(
            rec["min_surf_dist"]
            if rec.get("min_surf_dist") is not None
            else float("nan")
        ),
        pen_volume_m3=rec.get("pen_volume_m3"),
        pen_vert_ratio=rec.get("pen_vert_ratio")
        if rec.get("pen_volume_m3") is None
        else None,
        pen_inside_ratio=rec.get("pen_inside_ratio"),
        under_detect=bool(rec.get("under_detect")),
        vol_ok=float(rec.get("vol_ok") or 0.05),
        vol_bad=float(rec.get("vol_bad") or 0.15),
        d_good=float(rec.get("d_good") or 0.05),
        d_fail=float(rec.get("d_fail") or 0.40),
        n_detected_raw=rec.get("n_detected_raw"),
        n_expected=rec.get("n_expected"),
        s_ownership=rec.get("S_anat_ownership"),
    )
    out.update(anat)
    out.update(inter)
    out["mesh_score_protocol"] = "anat_v3.1+inter_v3.1"
    return out


def _list_json(d: Path) -> List[Path]:
    return sorted(p for p in d.glob("*.json") if not p.name.startswith("_"))


def _worker_one(args: Tuple[str, str, bool]) -> Tuple[str, dict, str]:
    """Process one json. Returns (name, rec, status) status in {wrote,skip,err}."""
    src_s, dst_s, force = args
    src_p, dst_p = Path(src_s), Path(dst_s)
    try:
        if dst_p.is_file() and not force:
            return dst_p.name, json.loads(dst_p.read_text(encoding="utf-8")), "skip"
        rec = json.loads(src_p.read_text(encoding="utf-8"))
        new = recompose_v31(rec)
        dst_p.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
        return dst_p.name, new, "wrote"
    except Exception as e:  # noqa: BLE001
        return src_p.name, {"ok": False, "error": str(e)}, "err"


def process_model(
    src_dir: Path,
    dst_dir: Path,
    *,
    force: bool,
    limit: int,
    workers: int = 1,
) -> Dict[str, Any]:
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = _list_json(src_dir)
    if limit > 0:
        files = files[:limit]
    jobs = [(str(p), str(dst_dir / p.name), force) for p in files]
    recs: List[dict] = []
    n_write = 0
    n_skip = 0
    n_err = 0
    t0 = time.time()
    if workers <= 1:
        for i, job in enumerate(jobs, 1):
            _, rec, st = _worker_one(job)
            if st == "wrote":
                n_write += 1
            elif st == "skip":
                n_skip += 1
            else:
                n_err += 1
            if st != "err":
                recs.append(rec)
            if i % 500 == 0:
                print(
                    f"  [{src_dir.name}] {i}/{len(jobs)} "
                    f"wrote={n_write} skip={n_skip} err={n_err}",
                    flush=True,
                )
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_worker_one, job) for job in jobs]
            for fut in as_completed(futs):
                _, rec, st = fut.result()
                done += 1
                if st == "wrote":
                    n_write += 1
                elif st == "skip":
                    n_skip += 1
                else:
                    n_err += 1
                if st != "err":
                    recs.append(rec)
                if done % 500 == 0:
                    print(
                        f"  [{src_dir.name}] {done}/{len(jobs)} "
                        f"wrote={n_write} skip={n_skip} err={n_err} "
                        f"workers={workers}",
                        flush=True,
                    )
    summary = summarize(recs) if recs else {"n": 0}
    summary["n_write"] = n_write
    summary["n_skip_existing"] = n_skip
    summary["n_err"] = n_err
    summary["elapsed_s"] = round(time.time() - t0, 1)
    # keep legacy means for quick A/B
    def _mean_legacy(key: str) -> Optional[float]:
        xs = [
            float(r[key])
            for r in recs
            if r.get("ok") and r.get(key) is not None
        ]
        return float(sum(xs) / len(xs)) if xs else None

    summary["S_anat_mesh_v3_legacy_mean"] = _mean_legacy("S_anat_mesh_v3_legacy")
    summary["S_inter_mesh_v3_legacy_mean"] = _mean_legacy("S_inter_mesh_v3_legacy")
    (dst_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument(
        "--models",
        default=",".join(PAPER_MODELS),
        help="comma-separated model ids; default = paper 9",
    )
    ap.add_argument("--src-name", default="mesh_v3", help="source judgments subdir")
    ap.add_argument("--dst-name", default="mesh_v31", help="dest judgments subdir")
    ap.add_argument("--force", action="store_true", help="overwrite existing mesh_v31 json")
    ap.add_argument("--limit", type=int, default=0, help="per-model cap (0=all)")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="CPU process workers per model (default 8; set 1 to disable)",
    )
    ap.add_argument(
        "--include-gt",
        action="store_true",
        help="also recompose _gt if present",
    )
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    src_root = pack / "judgments" / args.src_name
    dst_root = pack / "judgments" / args.dst_name
    if not src_root.is_dir():
        raise SystemExit(f"missing source: {src_root}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.include_gt and "_gt" not in models:
        models = ["_gt"] + models

    print(f"pack={pack}")
    print(f"src={src_root}  dst={dst_root}")
    print(
        f"models={models}  force={args.force}  limit={args.limit}  "
        f"workers={args.workers}"
    )

    # copy calibration if present
    cal = src_root / "_calibration.json"
    if cal.is_file():
        dst_root.mkdir(parents=True, exist_ok=True)
        (dst_root / "_calibration.json").write_text(cal.read_text(encoding="utf-8"))

    rows: List[Dict[str, Any]] = []
    for mid in models:
        src = src_root / mid
        if not src.is_dir():
            print(f"SKIP missing {src}")
            continue
        n_src = len(_list_json(src))
        print(f"==== {mid}  src_json={n_src} ====", flush=True)
        s = process_model(
            src,
            dst_root / mid,
            force=args.force,
            limit=args.limit,
            workers=max(1, int(args.workers)),
        )
        rows.append(
            {
                "model": mid,
                "n": s.get("n"),
                "n_ok": s.get("n_ok"),
                "S_anat_v3": s.get("S_anat_mesh_v3_legacy_mean"),
                "S_anat_v31": s.get("S_anat_mesh"),
                "S_inter_v3": s.get("S_inter_mesh_v3_legacy_mean"),
                "S_inter_v31": s.get("S_inter_mesh"),
                "n_write": s.get("n_write"),
                "elapsed_s": s.get("elapsed_s"),
            }
        )
        print(
            f"  done n={s.get('n')} write={s.get('n_write')} "
            f"Anat {s.get('S_anat_mesh_v3_legacy_mean')}→{s.get('S_anat_mesh')} "
            f"Inter {s.get('S_inter_mesh_v3_legacy_mean')}→{s.get('S_inter_mesh')}",
            flush=True,
        )

    table_path = dst_root / "_rescore_table.json"
    dst_root.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {table_path}")
    print("next: python compare_mesh_v31_full.py --pack ... --out .../mesh_v31_ab_full")


if __name__ == "__main__":
    main()
