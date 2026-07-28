#!/usr/bin/env python3
"""Rescore S_inter with prompt-conditioned penalty formula (no Multi-HMR).

Joins prompts from pack/manifest.jsonl when judgment jsons lack them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_metrics import (  # noqa: E402
    compose_inter_score,
    infer_needs_contact,
    prompt_contact_intent,
)


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    arr = np.sort(np.asarray(xs, dtype=np.float64))
    idx = float(np.clip(q, 0.0, 1.0)) * (len(arr) - 1)
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    if lo == hi:
        return float(arr[lo])
    w = idx - lo
    return float(arr[lo] * (1 - w) + arr[hi] * w)


def load_prompts(pack: Optional[Path]) -> Dict[str, str]:
    if pack is None:
        return {}
    man = pack / "manifest.jsonl"
    if not man.is_file():
        return {}
    out = {}
    with man.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["sample_id"]] = r.get("prompt") or ""
    return out


def calibrate_from_gt(gt_dir: Path, prompts: Dict[str, str]) -> Dict[str, float]:
    vols, dists = [], []
    for p in gt_dir.glob("*.json"):
        if p.name.startswith("_"):
            continue
        r = json.loads(p.read_text())
        if not r.get("ok") or r.get("recon_fail"):
            continue
        if r.get("pen_volume_m3") is not None:
            vols.append(float(r["pen_volume_m3"]))
        prompt = r.get("prompt") or prompts.get(r.get("sample_id", ""), "")
        intent = r.get("contact_intent") or prompt_contact_intent(prompt)
        if (
            intent == "required"
            and r.get("min_surf_dist") is not None
            and r["min_surf_dist"] == r["min_surf_dist"]
        ):
            dists.append(float(r["min_surf_dist"]))
    vol_ok = max(0.01, _percentile(vols, 0.55)) if vols else 0.05
    vol_bad = max(vol_ok * 1.8, _percentile(vols, 0.95) * 1.15) if vols else 0.15
    d_good, d_fail = 0.05, 0.40
    if dists:
        d_good = float(max(0.03, min(0.12, _percentile(dists, 0.80))))
        d_fail = float(max(d_good + 0.15, min(0.60, _percentile(dists, 0.98) + 0.15)))
    return {
        "vol_ok": vol_ok,
        "vol_bad": vol_bad,
        "tau_vol": max(vol_ok, _percentile(vols, 0.85) * 1.05) if vols else 0.05,
        "tau_pen": 0.05,
        "d_good": d_good,
        "d_fail": d_fail,
        "tau_contact": 0.02,
    }


def rescore(rec: dict, cal: Dict[str, float], prompts: Dict[str, str]) -> dict:
    if not rec.get("ok") or rec.get("recon_fail"):
        return rec
    sid = rec.get("sample_id", "")
    prompt = rec.get("prompt") or prompts.get(sid, "")
    intent = rec.get("contact_intent") or prompt_contact_intent(prompt)
    vol = rec.get("pen_volume_m3")
    inter = compose_inter_score(
        contact_intent=intent,
        prompt=prompt,
        needs_contact=(intent == "required"),
        min_surf_dist=float(
            rec["min_surf_dist"]
            if rec.get("min_surf_dist") is not None
            else float("nan")
        ),
        pen_volume_m3=vol,
        pen_vert_ratio=rec.get("pen_vert_ratio") if vol is None else None,
        under_detect=bool(rec.get("under_detect")),
        vol_ok=float(cal.get("vol_ok", 0.05)),
        vol_bad=float(cal.get("vol_bad", 0.15)),
        tau_pen=float(cal.get("tau_pen", 0.15)),
        d_good=float(cal.get("d_good", 0.05)),
        d_fail=float(cal.get("d_fail", 0.40)),
    )
    out = dict(rec)
    if prompt and not out.get("prompt"):
        out["prompt"] = prompt
    out.update(inter)
    return out


def summarize(recs: List[dict]) -> dict:
    ok = [r for r in recs if r.get("ok")]

    def mean(key, subset=None):
        xs = subset if subset is not None else ok
        vals = [float(r[key]) for r in xs if r.get(key) is not None]
        if not vals:
            return None
        arr = np.asarray(vals, dtype=np.float64)
        if not np.any(np.isfinite(arr)):
            return None
        return float(np.nanmean(arr))

    def by_intent(name):
        return [r for r in ok if (r.get("contact_intent") or "") == name]

    return {
        "n": len(recs),
        "n_ok": len(ok),
        "recon_fail_rate": float(
            np.mean([1.0 if r.get("recon_fail") else 0.0 for r in recs])
        )
        if recs
        else None,
        "S_count_mesh": mean("S_count_mesh"),
        "S_anat_mesh": mean("S_anat_mesh"),
        "S_anat_person_mean": mean("S_anat_person"),
        "S_anat_joint_mean": mean("S_anat_joint"),
        "S_anat_bone_mean": mean("S_anat_bone"),
        "S_anat_self_mean": mean("S_anat_self"),
        "S_anat_shape_mean": mean("S_anat_shape"),
        "S_anat_hand_mean": mean("S_anat_hand"),
        "S_anat_part_mesh_mean": mean("S_anat_part_mesh"),
        "S_anat_scale_mean": mean("S_anat_scale"),
        "S_anat_contact_region_mean": mean("S_anat_contact_region"),
        "S_anat_ownership_mean": mean("S_anat_ownership"),
        "S_anat_residual_mean": mean("S_anat_residual"),
        "S_anat_overcount_mean": mean("S_anat_overcount"),
        "P_anat_extra_mean": mean("P_anat_extra"),
        "anat_leftover_frac_mean": mean("anat_leftover_frac"),
        "anat_orphan_frac_mean": mean("anat_orphan_frac"),
        "anat_n_leftover_blobs_mean": mean("anat_n_leftover_blobs"),
        "S_anat_abhuman_mean": mean("S_anat_abhuman"),
        "S_inter_mesh": mean("S_inter_mesh"),
        "S_inter_required": mean("S_inter_mesh", by_intent("required")),
        "S_inter_unspecified": mean("S_inter_mesh", by_intent("unspecified")),
        "S_inter_forbidden": mean("S_inter_mesh", by_intent("forbidden")),
        "P_fuse_mean": mean("P_fuse"),
        "P_miss_mean": mean("P_miss"),
        "S_prox_mean": mean("S_prox"),
        "S_pen_mean": mean("S_pen"),
        "pen_vert_ratio_mean": mean("pen_vert_ratio"),
        "pen_inside_ratio_mean": mean("pen_inside_ratio"),
        "pen_volume_m3_mean": mean("pen_volume_m3"),
        "min_surf_dist_mean": mean("min_surf_dist"),
        "n_humans_mean": mean("n_humans"),
        "n_detected_raw_mean": mean("n_detected_raw"),
        "n_intent_required": len(by_intent("required")),
        "n_intent_unspecified": len(by_intent("unspecified")),
        "n_intent_forbidden": len(by_intent("forbidden")),
    }


def main(root: Path, pack: Optional[Path] = None) -> None:
    root = Path(root)
    if pack is None:
        # judgments/mesh_v3 -> pack root
        pack = root.parent.parent if root.name == "mesh_v3" else None
    prompts = load_prompts(pack)
    cal_path = root / "_calibration.json"
    gt_dir = root / "_gt"
    cal: Dict[str, float] = {}
    if cal_path.is_file():
        raw = json.loads(cal_path.read_text())
        for k in ("vol_ok", "vol_bad", "d_good", "d_fail", "tau_pen", "tau_vol", "tau_contact"):
            if k in raw and raw[k] is not None:
                cal[k] = float(raw[k])
    derived = calibrate_from_gt(gt_dir, prompts)
    # always refresh d_*/vol_* from GT + prompt intent (safe)
    derived.update({k: v for k, v in cal.items() if k in ("tau_contact",)})
    cal = derived
    cal_path.write_text(
        json.dumps(
            {
                **cal,
                "inter_formula": (
                    "penalty S=1-P_fuse[-P_miss|-P_unwanted]; "
                    "intent from prompt (required/forbidden/unspecified)"
                ),
                "note": "rescore_mesh_inter prompt-conditioned penalty",
            },
            indent=2,
        )
    )
    print("calibration", cal)

    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        recs = []
        for p in sorted(sub.glob("*.json")):
            if p.name.startswith("_"):
                continue
            r = rescore(json.loads(p.read_text()), cal, prompts)
            p.write_text(json.dumps(r, ensure_ascii=False, indent=2))
            recs.append(r)
        summary = summarize(recs)
        (sub / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(sub.name, summary)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Rescore S_inter_mesh (no Multi-HMR).")
    ap.add_argument(
        "root",
        nargs="?",
        default="",
        help="judgments/mesh_v3 dir (default: <pack>/judgments/mesh_v3)",
    )
    ap.add_argument(
        "--pack",
        default=str(Path.home() / "mpie_testset_pack"),
        help="pack root with manifest.jsonl (for prompts)",
    )
    ap.add_argument(
        "--root",
        dest="root_opt",
        default="",
        help="same as positional root; use if you prefer flags only",
    )
    args = ap.parse_args()
    pack = Path(args.pack).expanduser().resolve()
    root_s = (args.root_opt or args.root or "").strip()
    root = (
        Path(root_s).expanduser().resolve()
        if root_s
        else (pack / "judgments" / "mesh_v3")
    )
    if not root.is_dir():
        raise SystemExit(f"mesh_v3 dir not found: {root}")
    main(root, pack)
