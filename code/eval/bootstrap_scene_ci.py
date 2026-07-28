#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scene-clustered bootstrap CIs for mesh Anat/Inter.

Samples nest in scenes (middle token of sample_id). Resample scenes with
replacement; mean over all units in drawn scenes.

CPU-only; no GPU / Multi-HMR. Needs existing judgments/mesh_v3/<model>/*.json.

Example:
  python bootstrap_scene_ci.py \\
    --pack ~/mpie_testset_pack \\
    --out ./analysis/out/analysis_scene_ci.json \\
    --n-boot 2000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

MODEL_ORDER = [
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

DISPLAY = {
    "gpt-image-2": "GPT-Image-2",
    "gemini-3-pro-image": "Gemini-3-Pro-Image",
    "seedream-5-pro": "Seedream-5-Pro",
    "flux1-kontext-dev": "FLUX.1-Kontext",
    "dreamo": "DreamO",
    "omnigen2": "OmniGen2",
    "uno": "UNO",
    "ace": "ACE++",
    "bagel": "BAGEL",
    "firered": "FireRed-Image-Edit",
}


def scene_id(sample_id: str) -> str:
    parts = sample_id.split("__")
    return parts[1] if len(parts) >= 2 else sample_id


def load_mesh_scores(model_dir: Path) -> Dict[str, Dict[str, float]]:
    """Match paper summarize(): keep ok==True rows; recon_fail contributes 0 scores."""
    out: Dict[str, Dict[str, float]] = {}
    for p in model_dir.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if d.get("ok") is False:
            continue
        # Align with rescore_mesh_inter.summarize: ok rows include recon_fail (S=0).
        if d.get("recon_fail"):
            anat, inter = 0.0, 0.0
        else:
            anat, inter = d.get("S_anat_mesh"), d.get("S_inter_mesh")
            if anat is None or inter is None:
                continue
            if not (math.isfinite(float(anat)) and math.isfinite(float(inter))):
                continue
            anat, inter = float(anat), float(inter)
        sid = d.get("sample_id") or p.stem
        out[sid] = {"Anat": anat, "Inter": inter}
    return out


def group_by_scene(
    scores: Dict[str, Dict[str, float]],
) -> Dict[str, List[Dict[str, float]]]:
    by: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for sid, rec in scores.items():
        by[scene_id(sid)].append(rec)
    return dict(by)


def scene_bootstrap(
    by_scene: Dict[str, List[Dict[str, float]]],
    *,
    n_boot: int,
    seed: int,
    axes: Sequence[str] = ("Anat", "Inter"),
) -> Dict[str, Any]:
    scenes = list(by_scene.keys())
    n_sc = len(scenes)
    if n_sc == 0:
        return {"error": "no scenes"}

    rng = np.random.default_rng(seed)
    all_units = [u for sc in scenes for u in by_scene[sc]]
    point = {ax: float(np.mean([u[ax] for u in all_units])) for ax in axes}
    n_units = len(all_units)

    boots = {ax: np.empty(n_boot, dtype=np.float64) for ax in axes}
    for b in range(n_boot):
        draw = rng.choice(scenes, size=n_sc, replace=True)
        vals = {ax: [] for ax in axes}
        for sc in draw:
            for u in by_scene[sc]:
                for ax in axes:
                    vals[ax].append(u[ax])
        for ax in axes:
            boots[ax][b] = float(np.mean(vals[ax])) if vals[ax] else float("nan")

    ci = {}
    for ax in axes:
        lo, hi = np.quantile(boots[ax], [0.025, 0.975])
        ci[ax] = {
            "mean": point[ax],
            "ci95": [float(lo), float(hi)],
            "boot_std": float(np.std(boots[ax], ddof=1)),
        }
    return {
        "n_units": n_units,
        "n_scenes": n_sc,
        "n_boot": n_boot,
        "axes": ci,
    }


def paired_scene_delta(
    by_a: Dict[str, List[Dict[str, float]]],
    by_b: Dict[str, List[Dict[str, float]]],
    *,
    axis: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    shared = sorted(set(by_a) & set(by_b))
    if not shared:
        return {"error": "no shared scenes"}
    sc_delta = []
    for sc in shared:
        ma = float(np.mean([u[axis] for u in by_a[sc]]))
        mb = float(np.mean([u[axis] for u in by_b[sc]]))
        sc_delta.append(ma - mb)
    sc_delta = np.asarray(sc_delta, dtype=np.float64)
    point = float(np.mean(sc_delta))
    rng = np.random.default_rng(seed)
    n = len(sc_delta)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        boots[b] = float(np.mean(rng.choice(sc_delta, size=n, replace=True)))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = 2.0 * min(float(np.mean(boots >= 0)), float(np.mean(boots <= 0)))
    p = min(1.0, max(0.0, p))
    return {
        "axis": axis,
        "n_shared_scenes": n,
        "delta_mean": point,
        "ci95": [float(lo), float(hi)],
        "p_boot": p,
        "sign_ci_excludes_0": not (lo <= 0.0 <= hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument(
        "--out",
        default=str(
            Path(".")
            / "analysis" / "out"
            / "analysis_scene_ci.json"
        ),
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", default=",".join(MODEL_ORDER))
    ap.add_argument(
        "--pair",
        default="seedream-5-pro,gemini-3-pro-image",
        help="A,B model_ids for paired scene-delta (Inter/Anat)",
    )
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    mesh_root = pack / "judgments" / "mesh_v3"
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    by_model_scene: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    results: Dict[str, Any] = {
        "pack": str(pack),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "clustering": "scene_token_from_sample_id",
        "models": {},
    }

    for mid in models:
        mdir = mesh_root / mid
        if not mdir.is_dir():
            results["models"][mid] = {"error": f"missing {mdir}"}
            continue
        scores = load_mesh_scores(mdir)
        by_sc = group_by_scene(scores)
        by_model_scene[mid] = by_sc
        stats = scene_bootstrap(by_sc, n_boot=args.n_boot, seed=args.seed)
        stats["display"] = DISPLAY.get(mid, mid)
        results["models"][mid] = stats
        anat, inter = stats["axes"]["Anat"], stats["axes"]["Inter"]
        print(
            f"{DISPLAY.get(mid, mid):22s}  "
            f"Anat {anat['mean']:.3f} [{anat['ci95'][0]:.3f},{anat['ci95'][1]:.3f}]  "
            f"Inter {inter['mean']:.3f} [{inter['ci95'][0]:.3f},{inter['ci95'][1]:.3f}]  "
            f"n={stats['n_units']} scenes={stats['n_scenes']}",
            flush=True,
        )

    gt_sum = mesh_root / "_gt" / "_summary.json"
    smoke_gt = (
        Path.home()
        / "mpie_testset_pack"
        / "judgments"
        / "mesh_v3"
        / "_gt"
        / "_summary.json"
    )
    if gt_sum.is_file():
        results["gt_summary"] = json.loads(gt_sum.read_text())
        results["gt_source"] = str(gt_sum)
    elif smoke_gt.is_file():
        results["gt_summary"] = json.loads(smoke_gt.read_text())
        results["gt_source"] = str(smoke_gt)

    if args.pair and "," in args.pair:
        a, b = [x.strip() for x in args.pair.split(",", 1)]
        if a in by_model_scene and b in by_model_scene:
            results["paired"] = {}
            for axis in ("Anat", "Inter"):
                key = f"{a}_minus_{b}_{axis}"
                results["paired"][key] = paired_scene_delta(
                    by_model_scene[a],
                    by_model_scene[b],
                    axis=axis,
                    n_boot=args.n_boot,
                    seed=args.seed + 17,
                )
            print("paired:", json.dumps(results["paired"], indent=2), flush=True)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}", flush=True)

    tex_lines = []
    for mid in models:
        st = results["models"].get(mid) or {}
        if "axes" not in st:
            continue
        a, i = st["axes"]["Anat"], st["axes"]["Inter"]
        a_lo, a_hi = a["ci95"][0], a["ci95"][1]
        i_lo, i_hi = i["ci95"][0], i["ci95"][1]
        tex_lines.append(
            "{} & {:.2f} [{:.2f},{:.2f}] & {:.2f} [{:.2f},{:.2f}] \\\\".format(
                st.get("display", mid),
                a["mean"],
                a_lo,
                a_hi,
                i["mean"],
                i_lo,
                i_hi,
            )
        )
    tex_path = out.with_suffix(".texrows.txt")
    tex_path.write_text("\n".join(tex_lines) + "\n")
    print(f"wrote {tex_path}", flush=True)


if __name__ == "__main__":
    main()
