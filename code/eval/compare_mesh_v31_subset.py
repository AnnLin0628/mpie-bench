#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline A/B: mesh_v3 (stored) vs anat_v3.1 + inter_v3.1 on a sample subset.

No GPU. Reads judgments/mesh_v3/<model>/*.json and recomposes Anat/Inter.

Example:
  python compare_mesh_v31_subset.py \\
    --pack ~/mpie_testset_pack \\
    --sample-ids-file ~/mpie_testset_pack_seed150/manifest.jsonl \\
    --out /tmp/mesh_v31_ab
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from anat_extended import compose_anat_score
from mesh_metrics import compose_inter_score


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None

    def ranks(a: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: a[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and a[order[j + 1]] == a[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    deny = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if denx < 1e-12 or deny < 1e-12:
        return None
    return float(num / (denx * deny))


def load_sids(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            out.append(json.loads(line)["sample_id"])
        else:
            out.append(line)
    return out


def recompose(rec: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
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
    # Prefer stored calibration knobs when present
    inter = compose_inter_score(
        needs_contact=bool(rec.get("needs_contact")),
        contact_intent=rec.get("contact_intent")
        or (
            "required"
            if "required" in str(rec.get("inter_regime", ""))
            else (
                "forbidden"
                if "forbidden" in str(rec.get("inter_regime", ""))
                else "unspecified"
            )
        ),
        min_surf_dist=float(rec.get("min_surf_dist") or float("nan")),
        pen_volume_m3=rec.get("pen_volume_m3"),
        pen_vert_ratio=rec.get("pen_vert_ratio"),
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
    return float(anat["S_anat_mesh"]), float(inter["S_inter_mesh"]), {
        **anat,
        **{f"inter_{k}": v for k, v in inter.items() if k.startswith("P_") or k in ("S_pen", "S_prox", "pen_signal")},
    }


def mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--sample-ids-file", required=True)
    ap.add_argument(
        "--models",
        default="gpt-image-2,gemini-3-pro-image,seedream-5-pro,flux1-kontext-dev,dreamo,omnigen2,uno,ace,bagel,firered",
    )
    ap.add_argument("--out", required=True, help="output directory")
    # fig2 "visibly bad" probes (row1 fused-limb cases)
    ap.add_argument(
        "--probe-bad",
        default="omnigen2:hug__ece68b23998b__T5,ace:hug__ece68b23998b__T5",
        help="model:sid pairs expected to drop under v3.1",
    )
    ap.add_argument(
        "--probe-ok",
        default="gpt-image-2:hug__ece68b23998b__T5,flux1-kontext-dev:hug__ece68b23998b__T5",
        help="model:sid pairs that should not collapse",
    )
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sids = load_sids(Path(args.sample_ids_file).expanduser())
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    per_model: Dict[str, Dict[str, List[float]]] = {
        m: {"anat_old": [], "anat_new": [], "inter_old": [], "inter_new": []}
        for m in models
    }
    rows_csv: List[Dict[str, Any]] = []

    for m in models:
        d = pack / "judgments" / "mesh_v3" / m
        if not d.is_dir():
            print(f"SKIP missing {d}")
            continue
        for sid in sids:
            p = d / f"{sid}.json"
            if not p.is_file():
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            if not rec.get("ok") or rec.get("recon_fail"):
                continue
            try:
                a_old = float(rec["S_anat_mesh"])
                i_old = float(rec["S_inter_mesh"])
                a_new, i_new, meta = recompose(rec)
            except Exception as e:
                print(f"WARN {m}/{sid}: {e}")
                continue
            per_model[m]["anat_old"].append(a_old)
            per_model[m]["anat_new"].append(a_new)
            per_model[m]["inter_old"].append(i_old)
            per_model[m]["inter_new"].append(i_new)
            rows_csv.append(
                {
                    "model": m,
                    "sample_id": sid,
                    "S_anat_old": a_old,
                    "S_anat_v31": a_new,
                    "delta_anat": a_new - a_old,
                    "S_inter_old": i_old,
                    "S_inter_v31": i_new,
                    "delta_inter": i_new - i_old,
                    "n_detected_raw": rec.get("n_detected_raw"),
                    "inter_regime": rec.get("inter_regime"),
                }
            )

    # --- model-level summary ---
    summary_rows = []
    print(
        f"{'model':28s} {'n':>4s} | "
        f"{'Anat_old':>8s} {'Anat_new':>8s} {'ΔA':>7s} | "
        f"{'Inter_old':>9s} {'Inter_new':>9s} {'ΔI':>7s} | "
        f"{'stdA_o':>6s} {'stdA_n':>6s} {'stdI_o':>6s} {'stdI_n':>6s}"
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
            f"{m:28s} {n:4d} | "
            f"{ao:8.3f} {an:8.3f} {an-ao:+7.3f} | "
            f"{io:9.3f} {inn:9.3f} {inn-io:+7.3f} | "
            f"{as_:6.3f} {ans:6.3f} {is_:6.3f} {ins:6.3f}"
        )

    # ranking correlation (model means)
    means_ao = [r["Anat_old"] for r in summary_rows]
    means_an = [r["Anat_v31"] for r in summary_rows]
    means_io = [r["Inter_old"] for r in summary_rows]
    means_in = [r["Inter_v31"] for r in summary_rows]
    rho_a = spearman(means_ao, means_an)
    rho_i = spearman(means_io, means_in)

    def spread(xs: List[float]) -> float:
        return (max(xs) - min(xs)) if xs else float("nan")

    # paper-support metrics
    report = {
        "n_samples_filter": len(sids),
        "n_scored_rows": len(rows_csv),
        "n_models": len(summary_rows),
        "model_rank_spearman_Anat_old_vs_v31": rho_a,
        "model_rank_spearman_Inter_old_vs_v31": rho_i,
        "model_mean_spread_Anat_old": spread(means_ao),
        "model_mean_spread_Anat_v31": spread(means_an),
        "model_mean_spread_Inter_old": spread(means_io),
        "model_mean_spread_Inter_v31": spread(means_in),
        "mean_sample_std_Anat_old": sum(r["std_Anat_old"] for r in summary_rows)
        / max(len(summary_rows), 1),
        "mean_sample_std_Anat_v31": sum(r["std_Anat_v31"] for r in summary_rows)
        / max(len(summary_rows), 1),
        "mean_sample_std_Inter_old": sum(r["std_Inter_old"] for r in summary_rows)
        / max(len(summary_rows), 1),
        "mean_sample_std_Inter_v31": sum(r["std_Inter_v31"] for r in summary_rows)
        / max(len(summary_rows), 1),
        "mean_frac_Anat_ge095_old": sum(r["frac_Anat_ge095_old"] for r in summary_rows)
        / max(len(summary_rows), 1),
        "mean_frac_Anat_ge095_v31": sum(r["frac_Anat_ge095_v31"] for r in summary_rows)
        / max(len(summary_rows), 1),
        "mean_frac_Inter_ge095_old": sum(r["frac_Inter_ge095_old"] for r in summary_rows)
        / max(len(summary_rows), 1),
        "mean_frac_Inter_ge095_v31": sum(r["frac_Inter_ge095_v31"] for r in summary_rows)
        / max(len(summary_rows), 1),
        "by_model": summary_rows,
        "probes": {},
    }

    def parse_probes(s: str) -> List[Tuple[str, str]]:
        out = []
        for part in s.split(","):
            part = part.strip()
            if not part or ":" not in part:
                continue
            m, sid = part.split(":", 1)
            out.append((m.strip(), sid.strip()))
        return out

    for label, spec in (("bad", args.probe_bad), ("ok", args.probe_ok)):
        items = []
        for m, sid in parse_probes(spec):
            hit = next(
                (
                    r
                    for r in rows_csv
                    if r["model"] == m and r["sample_id"] == sid
                ),
                None,
            )
            if hit:
                items.append(hit)
        report["probes"][label] = items

    # write outputs
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

    # markdown verdict
    md = [
        "# mesh_v3 vs v3.1 (Anat+Inter) subset A/B",
        "",
        f"- filter samples: **{len(sids)}**",
        f"- scored rows: **{len(rows_csv)}** across **{len(summary_rows)}** models",
        "",
        "## Paper-support diagnostics",
        "",
        "| metric | old (v3) | new (v3.1) | prefer |",
        "|---|---:|---:|:---|",
        f"| model-mean Anat spread (max−min) | {report['model_mean_spread_Anat_old']:.3f} | {report['model_mean_spread_Anat_v31']:.3f} | larger (separates editors) |",
        f"| model-mean Inter spread | {report['model_mean_spread_Inter_old']:.3f} | {report['model_mean_spread_Inter_v31']:.3f} | larger |",
        f"| mean per-sample Anat std | {report['mean_sample_std_Anat_old']:.3f} | {report['mean_sample_std_Anat_v31']:.3f} | larger (less flat) |",
        f"| mean per-sample Inter std | {report['mean_sample_std_Inter_old']:.3f} | {report['mean_sample_std_Inter_v31']:.3f} | larger |",
        f"| frac Anat≥0.95 (mean over models) | {report['mean_frac_Anat_ge095_old']:.3f} | {report['mean_frac_Anat_ge095_v31']:.3f} | smaller (less ceiling) |",
        f"| frac Inter≥0.95 | {report['mean_frac_Inter_ge095_old']:.3f} | {report['mean_frac_Inter_ge095_v31']:.3f} | smaller |",
        f"| model-rank Spearman Anat old↔new | — | {rho_a if rho_a is not None else float('nan'):.3f} | high = ranking stable |",
        f"| model-rank Spearman Inter old↔new | — | {rho_i if rho_i is not None else float('nan'):.3f} | high = ranking stable |",
        "",
        "## Fig2 probes (row1 embrace)",
        "",
    ]
    for label in ("bad", "ok"):
        md.append(f"### {label}")
        md.append("| model | sid | ΔAnat | ΔInter | Anat_new | Inter_new |")
        md.append("|---|---|---:|---:|---:|---:|")
        for h in report["probes"].get(label, []):
            md.append(
                f"| {h['model']} | `{h['sample_id']}` | {h['delta_anat']:+.3f} | "
                f"{h['delta_inter']:+.3f} | {h['S_anat_v31']:.3f} | {h['S_inter_v31']:.3f} |"
            )
        md.append("")

    # automatic lean
    votes_new = 0
    votes_old = 0
    if report["model_mean_spread_Anat_v31"] > report["model_mean_spread_Anat_old"]:
        votes_new += 1
    else:
        votes_old += 1
    if report["model_mean_spread_Inter_v31"] > report["model_mean_spread_Inter_old"]:
        votes_new += 1
    else:
        votes_old += 1
    if report["mean_frac_Anat_ge095_v31"] < report["mean_frac_Anat_ge095_old"]:
        votes_new += 1
    else:
        votes_old += 1
    if report["mean_frac_Inter_ge095_v31"] < report["mean_frac_Inter_ge095_old"]:
        votes_new += 1
    else:
        votes_old += 1
    bad_drop = [
        h["delta_anat"] + h["delta_inter"]
        for h in report["probes"].get("bad", [])
    ]
    ok_drop = [
        h["delta_anat"] + h["delta_inter"]
        for h in report["probes"].get("ok", [])
    ]
    if bad_drop and ok_drop and (sum(bad_drop) / len(bad_drop) < sum(ok_drop) / len(ok_drop) - 0.05):
        votes_new += 1
        probe_note = "bad probes drop more than ok probes → supports qualitative claim"
    else:
        votes_old += 1
        probe_note = "probe separation unclear / ok probes also hit hard"

    lean = "v3.1" if votes_new > votes_old else ("v3 (old)" if votes_old > votes_new else "tie")
    md += [
        "## Verdict (heuristic)",
        "",
        f"- votes new/old: **{votes_new}/{votes_old}** → lean **{lean}**",
        f"- probe note: {probe_note}",
        f"- rank stability Anat/Inter Spearman: "
        f"**{rho_a if rho_a is not None else float('nan'):.3f}** / "
        f"**{rho_i if rho_i is not None else float('nan'):.3f}** "
        f"(keep high if you do not want main-table reshuffle)",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nwrote {out_dir}/REPORT.md")


if __name__ == "__main__":
    main()
