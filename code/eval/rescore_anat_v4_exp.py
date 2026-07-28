#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline Anat v4 experiment: mesh_v3 → judgments/mesh_anat_exp/ (never overwrites mesh_v3).

Example:
  python rescore_anat_v4_exp.py --pack ~/mpie_testset_pack --fig2
  python rescore_anat_v4_exp.py --pack ~/mpie_testset_pack --models omnigen2 --limit 50
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
from anat_extended import compose_anat_score_v4  # noqa: E402
from compare_mesh_v31_subset import recompose as recompose_v31  # noqa: E402

PAPER_MODELS = [
    "gpt-image-2",
    "gemini-3-pro-image",
    "seedream-5-pro",
    "flux1-kontext-dev",
    "dreamo",
    "omnigen2",
    "uno",
    "ace",
    "qwen-image-edit-2511",
]

FIG2_SIDS = [
    "hug__ece68b23998b__T5",
    "piggyback__946806af5ed9__T4",
    "piggyback__1469fe1a6428__T2",
    "wrestle_grapple__28f54bd775cc__T9",
]

DEFAULT_PROTOCOL: Dict[str, Any] = {
    "protocol": "anat_v4_exp",
    "w_attach": 0.40,
    "w_orphan": 0.20,
    "w_struct": 0.25,
    "w_resid": 0.15,
    "ownership_amp": 2.0,
    "leftover_ok": 0.55,
    "leftover_bad": 0.80,
    "inside_ok": 0.20,
    "inside_bad": 0.50,
    "overdetect_ok": 2.5,
    "overdetect_span": 2.0,
    "gate_leftover": 0.60,
    "gate_inside": 0.25,
    "gate_ownership": 0.98,
    "orphan_blob_ok": 0.0,
    "orphan_blob_bad": 3.0,
    "orphan_frac_ok": 0.02,
    "orphan_frac_bad": 0.15,
    "leftover_alone_scale": 0.20,
    "fuse_alone_scale": 0.25,
    "attached_blobs_max": 0,
    "attached_leftover_min": 0.65,
    "extreme_ratio": 3.5,
    "extreme_span": 1.5,
    "extreme_cap": 0.85,
}


def load_protocol(path: Optional[Path]) -> Dict[str, Any]:
    proto = dict(DEFAULT_PROTOCOL)
    if path is not None and path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        proto.update({k: v for k, v in loaded.items() if not k.startswith("_")})
    return proto


def _orphan_frac_from_rec(rec: Dict[str, Any]) -> Optional[float]:
    if rec.get("anat_orphan_frac") is not None:
        return float(rec["anat_orphan_frac"])
    scene = rec.get("anat_scene") or {}
    over = scene.get("structure_overcount") or {}
    if over.get("orphan_frac") is not None:
        return float(over["orphan_frac"])
    return None


def compose_v4_from_rec(
    rec: Dict[str, Any], proto: Dict[str, Any]
) -> Dict[str, Any]:
    kw = {
        k: proto[k]
        for k in (
            "w_attach",
            "w_orphan",
            "w_struct",
            "w_resid",
            "ownership_amp",
            "leftover_ok",
            "leftover_bad",
            "inside_ok",
            "inside_bad",
            "overdetect_ok",
            "overdetect_span",
            "gate_leftover",
            "gate_inside",
            "gate_ownership",
            "orphan_blob_ok",
            "orphan_blob_bad",
            "orphan_frac_ok",
            "orphan_frac_bad",
            "leftover_alone_scale",
            "fuse_alone_scale",
            "attached_blobs_max",
            "attached_leftover_min",
            "extreme_ratio",
            "extreme_span",
            "extreme_cap",
            "protocol",
        )
        if k in proto
    }
    return compose_anat_score_v4(
        s_residual=rec.get("S_anat_residual"),
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
        orphan_frac=_orphan_frac_from_rec(rec),
        pen_inside_ratio=rec.get("pen_inside_ratio"),
        p_fuse=rec.get("P_fuse"),
        **kw,
    )


def enrich_exp_record(rec: Dict[str, Any], proto: Dict[str, Any]) -> Dict[str, Any]:
    """Copy of rec + S_anat_v3_legacy + S_anat_v31 + S_anat_v4 (S_anat_mesh untouched)."""
    out = dict(rec)
    out["S_anat_v3_legacy"] = rec.get("S_anat_mesh")
    try:
        a31, _, _ = recompose_v31(rec)
        out["S_anat_v31"] = float(a31)
    except Exception as e:  # noqa: BLE001
        out["S_anat_v31"] = None
        out["S_anat_v31_error"] = repr(e)
    v4 = compose_v4_from_rec(rec, proto)
    # keep paper field frozen in the copy too
    paper_anat = rec.get("S_anat_mesh")
    out.update({k: v for k, v in v4.items() if k != "S_anat_mesh"})
    out["S_anat_mesh"] = paper_anat  # never overwrite with v4 in exp copies
    out["S_anat_v4"] = v4["S_anat_v4"]
    out["mesh_anat_exp"] = True
    return out


def _worker(args: Tuple[str, str, Dict[str, Any], bool]) -> Tuple[str, str]:
    src_s, dst_s, proto, force = args
    src_p, dst_p = Path(src_s), Path(dst_s)
    if dst_p.is_file() and not force:
        return src_p.name, "skip"
    rec = json.loads(src_p.read_text(encoding="utf-8"))
    out = enrich_exp_record(rec, proto)
    dst_p.parent.mkdir(parents=True, exist_ok=True)
    dst_p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return src_p.name, "wrote"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--models", default=",".join(PAPER_MODELS))
    ap.add_argument("--src-name", default="mesh_v3")
    ap.add_argument("--dst-name", default="mesh_anat_exp")
    ap.add_argument("--protocol-json", default="", help="optional tuned protocol")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--fig2",
        action="store_true",
        help="only Fig2 sample ids across models",
    )
    ap.add_argument("--sample-ids", default="", help="comma-separated; overrides --fig2")
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    src_root = pack / "judgments" / args.src_name
    dst_root = pack / "judgments" / args.dst_name
    proto_path = (
        Path(args.protocol_json).expanduser()
        if args.protocol_json
        else dst_root / "_protocol.json"
    )
    proto = load_protocol(proto_path if proto_path.is_file() else None)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.sample_ids.strip():
        sids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
    elif args.fig2:
        sids = list(FIG2_SIDS)
    else:
        sids = None

    dst_root.mkdir(parents=True, exist_ok=True)
    (dst_root / "_protocol.json").write_text(
        json.dumps(proto, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"pack={pack} src={src_root} dst={dst_root}")
    print(f"protocol leftover_ok/bad={proto['leftover_ok']}/{proto['leftover_bad']}")

    for mid in models:
        src = src_root / mid
        if not src.is_dir():
            print(f"SKIP missing {src}")
            continue
        if sids is not None:
            files = [src / f"{sid}.json" for sid in sids if (src / f"{sid}.json").is_file()]
        else:
            files = sorted(p for p in src.glob("*.json") if not p.name.startswith("_"))
            if args.limit > 0:
                files = files[: args.limit]
        jobs = [
            (str(p), str(dst_root / mid / p.name), proto, args.force) for p in files
        ]
        print(f"==== {mid} n={len(jobs)} ====", flush=True)
        t0 = time.time()
        n_write = n_skip = 0
        if args.workers <= 1:
            for job in jobs:
                _, st = _worker(job)
                n_write += int(st == "wrote")
                n_skip += int(st == "skip")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(_worker, j) for j in jobs]
                for fut in as_completed(futs):
                    _, st = fut.result()
                    n_write += int(st == "wrote")
                    n_skip += int(st == "skip")
        print(
            f"  wrote={n_write} skip={n_skip} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    print(f"done → {dst_root}")


if __name__ == "__main__":
    main()
