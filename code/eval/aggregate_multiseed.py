#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate multi-seed mesh_v3 judgments → mean±std for paper.

Expects:
  $PACK/judgments/mesh_v3/<base>_s0|s1|s2/<sample_id>.json

Writes:
  $PACK/judgments/mesh_v3/_multiseed/<base>_mean_std.json
  $PACK/judgments/mesh_v3/_multiseed/summary_table.md

Example:
  python aggregate_multiseed.py \\
    --pack ~/mpie_testset_pack_seed150 \\
    --bases flux1-kontext-dev ace omnigen2 \\
    --seeds 0,1,2
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


METRICS = ("S_anat_mesh", "S_inter_mesh")


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v


def load_seed_scores(
    pack: Path, mid: str
) -> Dict[str, Dict[str, float]]:
    """sample_id -> {metric: value} for ok records."""
    d = pack / "judgments" / "mesh_v3" / mid
    out: Dict[str, Dict[str, float]] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not rec.get("ok"):
            continue
        sid = rec.get("sample_id") or p.stem
        vals = {}
        for m in METRICS:
            v = _f(rec.get(m))
            if v is not None:
                vals[m] = v
        if vals:
            out[sid] = vals
    return out


def mean_std(xs: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not xs:
        return None, None
    if len(xs) == 1:
        return float(xs[0]), 0.0
    return float(statistics.mean(xs)), float(statistics.stdev(xs))


def aggregate_base(
    pack: Path, base: str, seeds: List[int]
) -> Dict[str, Any]:
    per_seed = {s: load_seed_scores(pack, f"{base}_s{s}") for s in seeds}
    common = None
    for s in seeds:
        ids = set(per_seed[s].keys())
        common = ids if common is None else (common & ids)
    common = sorted(common or [])

    per_sample: Dict[str, Any] = {}
    metric_pools: Dict[str, List[float]] = {m: [] for m in METRICS}
    # also pool of per-sample means across seeds
    sample_means: Dict[str, List[float]] = {m: [] for m in METRICS}
    sample_stds: Dict[str, List[float]] = {m: [] for m in METRICS}

    for sid in common:
        entry: Dict[str, Any] = {"sample_id": sid, "seeds": {}}
        for m in METRICS:
            xs = [per_seed[s][sid][m] for s in seeds if m in per_seed[s][sid]]
            mu, sd = mean_std(xs)
            entry[m] = {"values": xs, "mean": mu, "std": sd}
            if mu is not None:
                sample_means[m].append(mu)
            if sd is not None:
                sample_stds[m].append(sd)
            for s in seeds:
                if m in per_seed[s].get(sid, {}):
                    entry["seeds"].setdefault(str(s), {})[m] = per_seed[s][sid][m]
        per_sample[sid] = entry

    # seed-level means (mean over samples that exist in that seed)
    seed_means: Dict[str, Dict[str, Optional[float]]] = {}
    for s in seeds:
        sm: Dict[str, Optional[float]] = {}
        for m in METRICS:
            xs = [per_seed[s][sid][m] for sid in common if m in per_seed[s][sid]]
            mu, _ = mean_std(xs)
            sm[m] = mu
            if mu is not None:
                metric_pools[m].append(mu)
        seed_means[str(s)] = sm

    summary_metrics: Dict[str, Any] = {}
    for m in METRICS:
        # primary paper number: mean±std of *seed-level* means
        mu_s, sd_s = mean_std(metric_pools[m])
        # secondary: mean of per-sample std (typical sample-level variance)
        mu_ps, _ = mean_std(sample_stds[m])
        summary_metrics[m] = {
            "seed_means": metric_pools[m],
            "mean_of_seed_means": mu_s,
            "std_across_seeds": sd_s,  # ← paper mean±std
            "mean_per_sample_std": mu_ps,
            "n_common_samples": len(common),
        }

    return {
        "base_model_id": base,
        "seeds": seeds,
        "n_common_samples": len(common),
        "n_per_seed": {str(s): len(per_seed[s]) for s in seeds},
        "seed_means": seed_means,
        "metrics": summary_metrics,
        "per_sample": per_sample,
    }


def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument(
        "--bases",
        default="flux1-kontext-dev,ace,omnigen2",
        help="comma-separated base model ids (without _sN)",
    )
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument(
        "--out-dir",
        default="",
        help="default $PACK/judgments/mesh_v3/_multiseed",
    )
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else (
        pack / "judgments" / "mesh_v3" / "_multiseed"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_md = [
        "# Multi-seed mesh variance",
        "",
        f"Pack: `{pack}`",
        f"Seeds: {seeds}",
        "",
        "| model | n | Anat mean±std | Inter mean±std | mean sample-std Anat | mean sample-std Inter |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    all_out: Dict[str, Any] = {"pack": str(pack), "seeds": seeds, "models": {}}

    for base in bases:
        agg = aggregate_base(pack, base, seeds)
        path = out_dir / f"{base}_mean_std.json"
        path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")
        all_out["models"][base] = {
            k: agg[k]
            for k in (
                "n_common_samples",
                "n_per_seed",
                "seed_means",
                "metrics",
            )
        }
        ma = agg["metrics"]["S_anat_mesh"]
        mi = agg["metrics"]["S_inter_mesh"]
        rows_md.append(
            f"| {base} | {agg['n_common_samples']} | "
            f"{fmt(ma['mean_of_seed_means'])}±{fmt(ma['std_across_seeds'])} | "
            f"{fmt(mi['mean_of_seed_means'])}±{fmt(mi['std_across_seeds'])} | "
            f"{fmt(ma['mean_per_sample_std'])} | {fmt(mi['mean_per_sample_std'])} |"
        )
        print(
            f"[ok] {base}: Anat {fmt(ma['mean_of_seed_means'])}±{fmt(ma['std_across_seeds'])}  "
            f"Inter {fmt(mi['mean_of_seed_means'])}±{fmt(mi['std_across_seeds'])}  "
            f"n={agg['n_common_samples']}",
            flush=True,
        )

    rows_md += [
        "",
        "Notes:",
        "- `mean±std` = mean and std of **seed-level** means (paper-facing).",
        "- `mean sample-std` = average over samples of std across seeds (typical per-image noise).",
        "- Only samples with ok judgments in **all** seeds are kept.",
        "",
    ]
    (out_dir / "summary_table.md").write_text("\n".join(rows_md), encoding="utf-8")
    (out_dir / "all_models.json").write_text(
        json.dumps(all_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
