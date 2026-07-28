#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU path: Multi-HMR reconstruct → mesh-space corruptions → rescore.

Run on a CUDA GPU machine (needs CUDA + Multi-HMR). Complements the CPU synthetic
script corrupt_mesh_validate.py.

For each sample:
  1) Reconstruct GT or a model generation
  2) Score baseline
  3) Apply: force_penetration / separate_far / drop_person / duplicate_person
  4) Rescore without re-running HMR

Example:
  python corrupt_mesh_on_recon.py \\
    --pack ~/mpie_testset_pack \\
    --multihmr-repo ~/multi-hmr \\
    --source gt --limit 80 \\
    --out ./analysis/out/analysis_corruption_recon.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_metrics import score_humans  # noqa: E402
from score_mesh_v3 import (  # noqa: E402
    MultiHMRBackend,
    load_manifest,
    n_expected,
    needs_contact,
    select_top_k_humans,
)


def _translate(verts: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.asarray(verts, dtype=np.float64) + np.asarray(delta, dtype=np.float64)


def _centroid(verts: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(verts, dtype=np.float64), axis=0)


def _score_pack(
    verts: Sequence[np.ndarray],
    faces: np.ndarray,
    row: dict,
    *,
    poses=None,
    j3ds=None,
    shapes=None,
    j2ds=None,
    cal: Dict[str, Any],
    img_path: Optional[Path],
    n_raw: int,
    intent_override: Optional[str] = None,
) -> Dict[str, Any]:
    prompt = row.get("prompt")
    intent = intent_override
    return score_humans(
        list(verts),
        faces,
        n_expected(row),
        body_poses=poses,
        j3ds=j3ds,
        shapes=shapes,
        j2ds=j2ds,
        img_path=img_path,
        abhuman_weights=cal.get("abhuman_weights"),
        hmr_img_size=896,
        needs_contact=needs_contact(row) if intent is None else (intent == "required"),
        prompt=prompt,
        contact_intent=intent,
        tau_pen=float(cal.get("tau_pen", 0.15)),
        tau_contact=float(cal.get("tau_contact", 0.02)),
        tau_vol=float(cal.get("tau_vol", 0.05)),
        vol_ok=float(cal.get("vol_ok", 0.05)),
        vol_bad=float(cal.get("vol_bad", 0.15)),
        d_good=float(cal.get("d_good", 0.05)),
        d_fail=float(cal.get("d_fail", 0.40)),
        use_volume=True,
        use_anat_extended=True,
        n_detected_raw=n_raw,
    )


def _apply_corruptions(
    verts: List[np.ndarray],
    poses,
    j3ds,
    shapes,
    j2ds,
    scores,
) -> Dict[str, Dict[str, Any]]:
    """Return name -> {verts, poses, j3ds, shapes, j2ds, n_raw}."""
    out: Dict[str, Dict[str, Any]] = {}
    out["baseline"] = {
        "verts": verts,
        "poses": poses,
        "j3ds": j3ds,
        "shapes": shapes,
        "j2ds": j2ds,
        "n_raw": len(verts),
    }
    if len(verts) >= 2:
        c0, c1 = _centroid(verts[0]), _centroid(verts[1])
        mid = 0.5 * (c0 + c1)
        # pull both to mid → force penetration
        v_pen = [
            _translate(verts[0], mid - c0),
            _translate(verts[1], mid - c1),
        ]
        out["force_penetration"] = {
            "verts": v_pen,
            "poses": poses[:2] if poses else None,
            "j3ds": (
                [np.asarray(j3ds[0]) + (mid - c0), np.asarray(j3ds[1]) + (mid - c1)]
                if j3ds
                else None
            ),
            "shapes": shapes[:2] if shapes else None,
            "j2ds": j2ds[:2] if j2ds else None,
            "n_raw": 2,
        }
        # push apart along pelvis axis
        axis = c1 - c0
        nrm = float(np.linalg.norm(axis)) + 1e-8
        axis = axis / nrm
        v_far = [
            _translate(verts[0], -0.8 * axis),
            _translate(verts[1], 0.8 * axis),
        ]
        out["separate_far"] = {
            "verts": v_far,
            "poses": poses[:2] if poses else None,
            "j3ds": (
                [
                    np.asarray(j3ds[0]) - 0.8 * axis,
                    np.asarray(j3ds[1]) + 0.8 * axis,
                ]
                if j3ds
                else None
            ),
            "shapes": shapes[:2] if shapes else None,
            "j2ds": j2ds[:2] if j2ds else None,
            "n_raw": 2,
        }
        out["drop_person"] = {
            "verts": [verts[0]],
            "poses": poses[:1] if poses else None,
            "j3ds": j3ds[:1] if j3ds else None,
            "shapes": shapes[:1] if shapes else None,
            "j2ds": j2ds[:1] if j2ds else None,
            "n_raw": 1,
        }
        # duplicate first person offset slightly (extra structure)
        dup = _translate(verts[0], np.array([0.05, 0.0, 0.0]))
        out["duplicate_person"] = {
            "verts": [verts[0], verts[1], dup],
            "poses": (poses + [poses[0]]) if poses else None,
            "j3ds": (
                list(j3ds) + [np.asarray(j3ds[0]) + np.array([0.05, 0, 0])]
                if j3ds
                else None
            ),
            "shapes": (shapes + [shapes[0]]) if shapes else None,
            "j2ds": (j2ds + [j2ds[0]]) if j2ds else None,
            "n_raw": 3,
        }
    return out


def resolve_image(pack: Path, row: dict, source: str, model_id: str) -> Optional[Path]:
    if source == "gt":
        rel = row.get("gt_relpath")
        return (pack / rel) if rel else None
    # model generation
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = pack / "outputs" / model_id / f"{row['sample_id']}{ext}"
        if p.is_file():
            return p
    return None


def summarize_per_sample(
    per_sample: List[Dict[str, Any]],
    *,
    pack: str,
    source: str,
    model_id: str,
    limit: int,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agg: Dict[str, List[float]] = {}
    for rec in per_sample:
        if not rec.get("ok"):
            continue
        for name, sc in (rec.get("scores") or {}).items():
            for ax in ("S_anat_mesh", "S_inter_mesh"):
                val = sc.get(ax)
                if val is not None and np.isfinite(float(val)):
                    agg.setdefault(f"{name}:{ax}", []).append(float(val))
    summary_means = {k: float(np.mean(v)) for k, v in agg.items()}
    deltas: Dict[str, Any] = {}
    if "baseline:S_inter_mesh" in summary_means:
        bi = summary_means["baseline:S_inter_mesh"]
        ba = summary_means["baseline:S_anat_mesh"]
        for name in (
            "force_penetration",
            "separate_far",
            "drop_person",
            "duplicate_person",
        ):
            if f"{name}:S_inter_mesh" in summary_means:
                deltas[name] = {
                    "mean_Anat": summary_means.get(f"{name}:S_anat_mesh"),
                    "mean_Inter": summary_means.get(f"{name}:S_inter_mesh"),
                    "delta_Anat": summary_means.get(f"{name}:S_anat_mesh", ba) - ba,
                    "delta_Inter": summary_means.get(f"{name}:S_inter_mesh", bi) - bi,
                    "n": len(agg.get(f"{name}:S_inter_mesh", [])),
                }
    out_obj: Dict[str, Any] = {
        "pack": pack,
        "source": source,
        "model_id": model_id,
        "limit": limit,
        "n_ok": sum(1 for r in per_sample if r.get("ok")),
        "n_total_attempted": len(per_sample),
        "mean_scores": summary_means,
        "delta_vs_baseline": deltas,
        "monotonic_ok": {
            "force_penetration_Inter_down": (
                deltas.get("force_penetration", {}).get("delta_Inter", 0) < 0
            ),
            "separate_far_Inter_down": (
                deltas.get("separate_far", {}).get("delta_Inter", 0) < 0
            ),
            "drop_person_Inter_down": (
                deltas.get("drop_person", {}).get("delta_Inter", 0) < 0
            ),
        },
        "per_sample": per_sample,
    }
    if extra_meta:
        out_obj.update(extra_meta)
    return out_obj


def merge_shard_files(paths: Sequence[Path], out: Path) -> Dict[str, Any]:
    per_sample: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        per_sample.extend(d.get("per_sample") or [])
        meta.setdefault("pack", d.get("pack"))
        meta.setdefault("source", d.get("source"))
        meta.setdefault("model_id", d.get("model_id"))
        meta["limit"] = max(int(meta.get("limit") or 0), int(d.get("limit") or 0))
    # de-dupe by sample_id (keep first)
    seen = set()
    uniq: List[Dict[str, Any]] = []
    for r in per_sample:
        sid = r.get("sample_id")
        if sid in seen:
            continue
        seen.add(sid)
        uniq.append(r)
    obj = summarize_per_sample(
        uniq,
        pack=str(meta.get("pack") or ""),
        source=str(meta.get("source") or "gt"),
        model_id=str(meta.get("model_id") or "_gt"),
        limit=int(meta.get("limit") or 0),
        extra_meta={"merged_from": [str(p) for p in paths]},
    )
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2) + "\n")
    print(json.dumps({"delta_vs_baseline": obj["delta_vs_baseline"], "out": str(out),
                      "n_ok": obj["n_ok"], "n_total_attempted": obj["n_total_attempted"]},
                     indent=2))
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--multihmr-repo", default=os.environ.get("MULTIHMR_REPO", ""))
    ap.add_argument("--backend-model", default="multiHMR_896_L")
    ap.add_argument("--source", choices=["gt", "model"], default="gt")
    ap.add_argument("--model-id", default="flux1-kontext-dev")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--det-thresh", type=float, default=0.2)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split first --limit rows by index %% num_shards == shard_id")
    ap.add_argument(
        "--merge-shards",
        nargs="+",
        default=[],
        help="merge shard JSON paths and write --out (no GPU)",
    )
    ap.add_argument(
        "--out",
        default=str(
            Path(".")
            / "analysis" / "out"
            / "analysis_corruption_recon.json"
        ),
    )
    args = ap.parse_args()

    if args.merge_shards:
        merge_shard_files([Path(p) for p in args.merge_shards], Path(args.out))
        return

    pack = Path(args.pack).expanduser().resolve()
    repo = Path(args.multihmr_repo).expanduser() if args.multihmr_repo else None
    if not repo or not repo.is_dir():
        raise SystemExit("pass --multihmr-repo /path/to/multi-hmr (CUDA machine)")

    cal_path = pack / "judgments" / "mesh_v3" / "_calibration.json"
    cal: Dict[str, Any] = {}
    if cal_path.is_file():
        raw = json.loads(cal_path.read_text())
        for k in ("tau_pen", "tau_vol", "tau_contact", "vol_ok", "vol_bad", "d_good", "d_fail"):
            if k in raw and raw[k] is not None:
                cal[k] = float(raw[k])
        if raw.get("abhuman_weights"):
            cal["abhuman_weights"] = raw["abhuman_weights"]
    for k, v in {
        "tau_pen": 0.05,
        "tau_vol": 0.05,
        "tau_contact": 0.02,
        "vol_ok": 0.05,
        "vol_bad": 0.15,
        "d_good": 0.05,
        "d_fail": 0.40,
    }.items():
        cal.setdefault(k, v)

    rows = load_manifest(pack)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
        print(
            f"[corrupt] shard {args.shard_id}/{args.num_shards} → {len(rows)} samples",
            flush=True,
        )

    backend = MultiHMRBackend(
        repo, model_name=args.backend_model, det_thresh=args.det_thresh
    )

    per_sample: List[Dict[str, Any]] = []
    agg: Dict[str, List[float]] = {}

    for row in rows:
        img = resolve_image(pack, row, args.source, args.model_id)
        if img is None or not img.is_file():
            continue
        try:
            verts_all, poses_all, j3ds_all, shapes_all, j2ds_all, scores_all, _ms = (
                backend.infer(img)
            )
        except Exception as e:
            per_sample.append(
                {"sample_id": row["sample_id"], "ok": False, "error": repr(e)}
            )
            continue
        n_exp = n_expected(row)
        verts, poses, j3ds, shapes, j2ds, scores, _keep = select_top_k_humans(
            verts_all,
            poses_all,
            j3ds_all,
            shapes_all,
            j2ds_all,
            scores_all,
            n_exp,
        )
        if len(verts) < 1:
            per_sample.append(
                {"sample_id": row["sample_id"], "ok": False, "error": "no_humans"}
            )
            continue

        corruptions = _apply_corruptions(verts, poses, j3ds, shapes, j2ds, scores)
        sample_rec: Dict[str, Any] = {
            "sample_id": row["sample_id"],
            "ok": True,
            "source": args.source,
            "model_id": args.model_id if args.source == "model" else "_gt",
            "n_baseline": len(verts),
            "scores": {},
        }
        for name, pack_c in corruptions.items():
            geo = _score_pack(
                pack_c["verts"],
                backend.faces,
                row,
                poses=pack_c["poses"],
                j3ds=pack_c["j3ds"],
                shapes=pack_c["shapes"],
                j2ds=pack_c["j2ds"],
                cal=cal,
                img_path=img if name == "baseline" else None,
                # image residual only meaningful on baseline; corruptions are mesh-only
                n_raw=pack_c["n_raw"],
            )
            # re-score Anat extended without image for non-baseline to avoid false residual
            if name != "baseline":
                geo = score_humans(
                    pack_c["verts"],
                    backend.faces,
                    n_exp,
                    body_poses=pack_c["poses"],
                    j3ds=pack_c["j3ds"],
                    shapes=pack_c["shapes"],
                    j2ds=None,
                    img_path=None,
                    needs_contact=needs_contact(row),
                    prompt=row.get("prompt"),
                    tau_pen=float(cal["tau_pen"]),
                    tau_contact=float(cal["tau_contact"]),
                    tau_vol=float(cal["tau_vol"]),
                    vol_ok=float(cal["vol_ok"]),
                    vol_bad=float(cal["vol_bad"]),
                    d_good=float(cal["d_good"]),
                    d_fail=float(cal["d_fail"]),
                    use_volume=True,
                    use_anat_extended=False,
                    n_detected_raw=pack_c["n_raw"],
                )
            sample_rec["scores"][name] = {
                "S_anat_mesh": geo.get("S_anat_mesh"),
                "S_inter_mesh": geo.get("S_inter_mesh"),
                "P_fuse": geo.get("P_fuse"),
                "P_miss": geo.get("P_miss"),
                "min_surf_dist": geo.get("min_surf_dist"),
                "under_detect": geo.get("under_detect"),
                "n_humans": geo.get("n_humans"),
            }
            for ax in ("S_anat_mesh", "S_inter_mesh"):
                key = f"{name}:{ax}"
                val = geo.get(ax)
                if val is not None and np.isfinite(float(val)):
                    agg.setdefault(key, []).append(float(val))
        per_sample.append(sample_rec)
        print(
            f"[shard{args.shard_id} {len(per_sample)}] {row['sample_id']} "
            f"base Inter={sample_rec['scores']['baseline']['S_inter_mesh']:.3f}",
            flush=True,
        )

    out_obj = summarize_per_sample(
        per_sample,
        pack=str(pack),
        source=args.source,
        model_id=args.model_id if args.source == "model" else "_gt",
        limit=args.limit,
        extra_meta={
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
        },
    )
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_obj, indent=2) + "\n")
    print(
        json.dumps(
            {
                "delta_vs_baseline": out_obj["delta_vs_baseline"],
                "out": str(out),
                "n_ok": out_obj["n_ok"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
