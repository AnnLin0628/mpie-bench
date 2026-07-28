#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full A/B: stored mesh_v3 vs anat/inter v3.1 (all samples × paper models).

Reads judgments/mesh_v3 (old scores in JSON) and optionally judgments/mesh_v31
(if already written). If mesh_v31 missing, recomposes on the fly (no write).

Outputs per_sample.csv, per_model.csv, report.json, REPORT.md + verdict lean.

Example:
  python compare_mesh_v31_full.py \\
    --pack ~/mpie_testset_pack \\
    --out ~/mpie_testset_pack/judgments/mesh_v31_ab_full
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mesh_v31_subset import (  # noqa: E402
    mean_std,
    recompose,
    spearman,
)
from rescore_mesh_v31_full import PAPER_MODELS  # noqa: E402


def _list_json(d: Path) -> List[Path]:
    return sorted(p for p in d.glob("*.json") if not p.name.startswith("_"))


def _spread(xs: List[float]) -> float:
    return (max(xs) - min(xs)) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(PAPER_MODELS))
    ap.add_argument("--old-name", default="mesh_v3")
    ap.add_argument("--new-name", default="mesh_v31")
    ap.add_argument(
        "--prefer-written-v31",
        action="store_true",
        help="if mesh_v31/<m>/<sid>.json exists, use its scores instead of recompose",
    )
    ap.add_argument(
        "--probe-bad",
        default="omnigen2:hug__ece68b23998b__T5,ace:hug__ece68b23998b__T5",
    )
    ap.add_argument(
        "--probe-ok",
        default="gpt-image-2:hug__ece68b23998b__T5,flux1-kontext-dev:hug__ece68b23998b__T5",
    )
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    old_root = pack / "judgments" / args.old_name
    new_root = pack / "judgments" / args.new_name

    per_model: Dict[str, Dict[str, List[float]]] = {
        m: {"anat_old": [], "anat_new": [], "inter_old": [], "inter_new": []}
        for m in models
    }
    rows_csv: List[Dict[str, Any]] = []

    for m in models:
        d_old = old_root / m
        if not d_old.is_dir():
            print(f"SKIP missing {d_old}")
            continue
        files = _list_json(d_old)
        print(f"==== {m}  n_json={len(files)} ====", flush=True)
        for p in files:
            rec = json.loads(p.read_text(encoding="utf-8"))
            if not rec.get("ok") or rec.get("recon_fail"):
                continue
            try:
                a_old = float(rec["S_anat_mesh"])
                i_old = float(rec["S_inter_mesh"])
            except (KeyError, TypeError, ValueError):
                continue

            used = "recompose"
            a_new: float
            i_new: float
            if args.prefer_written_v31:
                p_new = new_root / m / p.name
                if p_new.is_file():
                    rec_n = json.loads(p_new.read_text(encoding="utf-8"))
                    try:
                        a_new = float(rec_n["S_anat_mesh"])
                        i_new = float(rec_n["S_inter_mesh"])
                        used = "mesh_v31"
                    except (KeyError, TypeError, ValueError):
                        a_new, i_new, _ = recompose(rec)
                else:
                    a_new, i_new, _ = recompose(rec)
            else:
                a_new, i_new, _ = recompose(rec)

            per_model[m]["anat_old"].append(a_old)
            per_model[m]["anat_new"].append(a_new)
            per_model[m]["inter_old"].append(i_old)
            per_model[m]["inter_new"].append(i_new)
            rows_csv.append(
                {
                    "model": m,
                    "sample_id": p.stem,
                    "S_anat_old": a_old,
                    "S_anat_v31": a_new,
                    "delta_anat": a_new - a_old,
                    "S_inter_old": i_old,
                    "S_inter_v31": i_new,
                    "delta_inter": i_new - i_old,
                    "n_detected_raw": rec.get("n_detected_raw"),
                    "inter_regime": rec.get("inter_regime"),
                    "source_new": used,
                }
            )

    summary_rows: List[Dict[str, Any]] = []
    print(
        f"{'model':28s} {'n':>5s} | "
        f"{'Anat_old':>8s} {'Anat_new':>8s} {'ΔA':>7s} | "
        f"{'Inter_old':>9s} {'Inter_new':>9s} {'ΔI':>7s}"
    )
    for m in models:
        d = per_model[m]
        n = len(d["anat_old"])
        if n == 0:
            continue
        ao, as_ = mean_std(d["anat_old"])
        an, ans = mean_std(d["anat_new"])
        io, is_ = mean_std(d["inter_old"])
        inn, ins = mean_std(d["inter_new"])
        summary_rows.append(
            {
                "model": m,
                "n": n,
                "Anat_old": ao,
                "Anat_v31": an,
                "delta_Anat": an - ao,
                "Inter_old": io,
                "Inter_v31": inn,
                "delta_Inter": inn - io,
                "std_Anat_old": as_,
                "std_Anat_v31": ans,
                "std_Inter_old": is_,
                "std_Inter_v31": ins,
                "frac_Anat_ge095_old": sum(1 for x in d["anat_old"] if x >= 0.95) / n,
                "frac_Anat_ge095_v31": sum(1 for x in d["anat_new"] if x >= 0.95) / n,
                "frac_Inter_ge095_old": sum(1 for x in d["inter_old"] if x >= 0.95) / n,
                "frac_Inter_ge095_v31": sum(1 for x in d["inter_new"] if x >= 0.95) / n,
            }
        )
        print(
            f"{m:28s} {n:5d} | "
            f"{ao:8.3f} {an:8.3f} {an - ao:+7.3f} | "
            f"{io:9.3f} {inn:9.3f} {inn - io:+7.3f}"
        )

    if not summary_rows:
        raise SystemExit("no scored rows — check pack/judgments/mesh_v3")

    means_ao = [r["Anat_old"] for r in summary_rows]
    means_an = [r["Anat_v31"] for r in summary_rows]
    means_io = [r["Inter_old"] for r in summary_rows]
    means_in = [r["Inter_v31"] for r in summary_rows]
    rho_a = spearman(means_ao, means_an)
    rho_i = spearman(means_io, means_in)

    report: Dict[str, Any] = {
        "n_scored_rows": len(rows_csv),
        "n_models": len(summary_rows),
        "model_rank_spearman_Anat_old_vs_v31": rho_a,
        "model_rank_spearman_Inter_old_vs_v31": rho_i,
        "model_mean_spread_Anat_old": _spread(means_ao),
        "model_mean_spread_Anat_v31": _spread(means_an),
        "model_mean_spread_Inter_old": _spread(means_io),
        "model_mean_spread_Inter_v31": _spread(means_in),
        "mean_sample_std_Anat_old": sum(r["std_Anat_old"] for r in summary_rows)
        / len(summary_rows),
        "mean_sample_std_Anat_v31": sum(r["std_Anat_v31"] for r in summary_rows)
        / len(summary_rows),
        "mean_sample_std_Inter_old": sum(r["std_Inter_old"] for r in summary_rows)
        / len(summary_rows),
        "mean_sample_std_Inter_v31": sum(r["std_Inter_v31"] for r in summary_rows)
        / len(summary_rows),
        "mean_frac_Anat_ge095_old": sum(r["frac_Anat_ge095_old"] for r in summary_rows)
        / len(summary_rows),
        "mean_frac_Anat_ge095_v31": sum(r["frac_Anat_ge095_v31"] for r in summary_rows)
        / len(summary_rows),
        "mean_frac_Inter_ge095_old": sum(r["frac_Inter_ge095_old"] for r in summary_rows)
        / len(summary_rows),
        "mean_frac_Inter_ge095_v31": sum(r["frac_Inter_ge095_v31"] for r in summary_rows)
        / len(summary_rows),
        "by_model": summary_rows,
        "probes": {},
    }

    def parse_probes(s: str) -> List[Tuple[str, str]]:
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            mm, sid = part.split(":", 1)
            out.append((mm.strip(), sid.strip()))
        return out

    for label, spec in (("bad", args.probe_bad), ("ok", args.probe_ok)):
        items = []
        for m, sid in parse_probes(spec):
            hit = next(
                (r for r in rows_csv if r["model"] == m and r["sample_id"] == sid),
                None,
            )
            if hit:
                items.append(hit)
        report["probes"][label] = items

    # ranking tables (high→low)
    rank_old_a = sorted(summary_rows, key=lambda r: r["Anat_old"], reverse=True)
    rank_new_a = sorted(summary_rows, key=lambda r: r["Anat_v31"], reverse=True)
    rank_old_i = sorted(summary_rows, key=lambda r: r["Inter_old"], reverse=True)
    rank_new_i = sorted(summary_rows, key=lambda r: r["Inter_v31"], reverse=True)
    report["rank_Anat_old"] = [r["model"] for r in rank_old_a]
    report["rank_Anat_v31"] = [r["model"] for r in rank_new_a]
    report["rank_Inter_old"] = [r["model"] for r in rank_old_i]
    report["rank_Inter_v31"] = [r["model"] for r in rank_new_i]

    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)
    with (out_dir / "per_model.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    votes_new = 0
    votes_old = 0
    reasons: List[str] = []

    def vote(cond_new: bool, why: str) -> None:
        nonlocal votes_new, votes_old
        if cond_new:
            votes_new += 1
            reasons.append(f"+v3.1: {why}")
        else:
            votes_old += 1
            reasons.append(f"+v3: {why}")

    vote(
        report["model_mean_spread_Anat_v31"] > report["model_mean_spread_Anat_old"],
        f"Anat model spread {report['model_mean_spread_Anat_old']:.3f}→{report['model_mean_spread_Anat_v31']:.3f}",
    )
    vote(
        report["model_mean_spread_Inter_v31"] > report["model_mean_spread_Inter_old"],
        f"Inter model spread {report['model_mean_spread_Inter_old']:.3f}→{report['model_mean_spread_Inter_v31']:.3f}",
    )
    vote(
        report["mean_frac_Anat_ge095_v31"] < report["mean_frac_Anat_ge095_old"],
        f"Anat≥0.95 {report['mean_frac_Anat_ge095_old']:.3f}→{report['mean_frac_Anat_ge095_v31']:.3f}",
    )
    vote(
        report["mean_frac_Inter_ge095_v31"] < report["mean_frac_Inter_ge095_old"],
        f"Inter≥0.95 {report['mean_frac_Inter_ge095_old']:.3f}→{report['mean_frac_Inter_ge095_v31']:.3f}",
    )
    vote(
        report["mean_sample_std_Anat_v31"] > report["mean_sample_std_Anat_old"],
        f"Anat sample-std {report['mean_sample_std_Anat_old']:.3f}→{report['mean_sample_std_Anat_v31']:.3f}",
    )
    vote(
        report["mean_sample_std_Inter_v31"] > report["mean_sample_std_Inter_old"],
        f"Inter sample-std {report['mean_sample_std_Inter_old']:.3f}→{report['mean_sample_std_Inter_v31']:.3f}",
    )

    bad_drop = [
        h["delta_anat"] + h["delta_inter"] for h in report["probes"].get("bad", [])
    ]
    ok_drop = [
        h["delta_anat"] + h["delta_inter"] for h in report["probes"].get("ok", [])
    ]
    if bad_drop and ok_drop:
        vote(
            (sum(bad_drop) / len(bad_drop)) < (sum(ok_drop) / len(ok_drop) - 0.05),
            f"probe Δ(bad)={sum(bad_drop)/len(bad_drop):+.3f} vs Δ(ok)={sum(ok_drop)/len(ok_drop):+.3f}",
        )

    lean = (
        "v3.1"
        if votes_new > votes_old
        else ("v3 (old)" if votes_old > votes_new else "tie")
    )
    report["verdict"] = {
        "votes_new": votes_new,
        "votes_old": votes_old,
        "lean": lean,
        "reasons": reasons,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Full mesh_v3 vs v3.1 (Anat+Inter)",
        "",
        f"- scored rows: **{len(rows_csv)}** across **{len(summary_rows)}** models",
        f"- rank Spearman Anat / Inter old↔new: "
        f"**{rho_a if rho_a is not None else float('nan'):.3f}** / "
        f"**{rho_i if rho_i is not None else float('nan'):.3f}**",
        "",
        "## Model means",
        "",
        "| model | n | Anat_v3 | Anat_v31 | ΔA | Inter_v3 | Inter_v31 | ΔI |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        md.append(
            f"| {r['model']} | {r['n']} | {r['Anat_old']:.3f} | {r['Anat_v31']:.3f} | "
            f"{r['delta_Anat']:+.3f} | {r['Inter_old']:.3f} | {r['Inter_v31']:.3f} | "
            f"{r['delta_Inter']:+.3f} |"
        )
    md += [
        "",
        "## Rankings (high→low)",
        "",
        f"- Anat v3:   {' ≻ '.join(report['rank_Anat_old'])}",
        f"- Anat v31:  {' ≻ '.join(report['rank_Anat_v31'])}",
        f"- Inter v3:  {' ≻ '.join(report['rank_Inter_old'])}",
        f"- Inter v31: {' ≻ '.join(report['rank_Inter_v31'])}",
        "",
        "## Diagnostics",
        "",
        "| metric | v3 | v3.1 | prefer |",
        "|---|---:|---:|:---|",
        f"| Anat model spread | {report['model_mean_spread_Anat_old']:.3f} | {report['model_mean_spread_Anat_v31']:.3f} | larger |",
        f"| Inter model spread | {report['model_mean_spread_Inter_old']:.3f} | {report['model_mean_spread_Inter_v31']:.3f} | larger |",
        f"| mean Anat sample-std | {report['mean_sample_std_Anat_old']:.3f} | {report['mean_sample_std_Anat_v31']:.3f} | larger |",
        f"| mean Inter sample-std | {report['mean_sample_std_Inter_old']:.3f} | {report['mean_sample_std_Inter_v31']:.3f} | larger |",
        f"| frac Anat≥0.95 | {report['mean_frac_Anat_ge095_old']:.3f} | {report['mean_frac_Anat_ge095_v31']:.3f} | smaller |",
        f"| frac Inter≥0.95 | {report['mean_frac_Inter_ge095_old']:.3f} | {report['mean_frac_Inter_ge095_v31']:.3f} | smaller |",
        "",
        "## Verdict (heuristic)",
        "",
        f"- votes new/old: **{votes_new}/{votes_old}** → lean **{lean}**",
        "",
    ]
    for why in reasons:
        md.append(f"- {why}")
    md.append("")
    md.append(
        "> Heuristic only. For paper cutover also re-map consistency Checklist_M "
        "and recompute ρ(H,M)."
    )
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nwrote {out_dir}/REPORT.md")


if __name__ == "__main__":
    main()
