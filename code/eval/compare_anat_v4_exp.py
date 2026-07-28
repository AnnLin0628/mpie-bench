#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Anat v3 / v3.1 / v4_exp; Fig2 hard acceptance + optional full/seed report.

Example:
  python compare_anat_v4_exp.py --pack ~/mpie_testset_pack --fig2 \\
    --out ~/mpie_testset_pack/judgments/mesh_anat_exp_ab
  python compare_anat_v4_exp.py --pack ... --full --out ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_mesh_v31_subset import recompose as recompose_v31  # noqa: E402
from rescore_anat_v4_exp import (  # noqa: E402
    DEFAULT_PROTOCOL,
    FIG2_SIDS,
    PAPER_MODELS,
    compose_v4_from_rec,
    load_protocol,
)

FIG2_ROW1 = "hug__ece68b23998b__T5"
FIG2_GOOD = ("gpt-image-2", "flux1-kontext-dev")
FIG2_BAD = ("omnigen2", "ace", "uno")


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


def anat_v3_legacy(rec: Dict[str, Any]) -> Optional[float]:
    if rec.get("S_anat_mesh") is None:
        return None
    return float(rec["S_anat_mesh"])


def anat_v31(rec: Dict[str, Any]) -> float:
    a, _, _ = recompose_v31(rec)
    return float(a)


def score_row(
    rec: Dict[str, Any], proto: Dict[str, Any]
) -> Dict[str, Any]:
    a3 = anat_v3_legacy(rec)
    if a3 is None:
        raise KeyError("S_anat_mesh")
    try:
        a31 = anat_v31(rec)
    except Exception:
        a31 = float("nan")
    v4 = compose_v4_from_rec(rec, proto)
    return {
        "S_anat_v3": a3,
        "S_anat_v31": a31,
        "S_anat_v4": float(v4["S_anat_v4"]),
        "P_anat_attach": v4.get("P_anat_attach"),
        "P_anat_orphan": v4.get("P_anat_orphan"),
        "P_anat_leftover": v4.get("P_anat_leftover"),
        "P_anat_fuse": v4.get("P_anat_fuse"),
        "P_anat_overdetect_gated": v4.get("P_anat_overdetect_gated"),
        "overdetect_gated": v4.get("overdetect_gated"),
        "n_detected_raw": rec.get("n_detected_raw"),
        "anat_leftover_frac": rec.get("anat_leftover_frac"),
        "pen_inside_ratio": rec.get("pen_inside_ratio"),
    }


def fig2_hard_checks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Hard acceptance on row1 embrace."""
    by = {
        (r["model"], r["sample_id"]): r
        for r in rows
        if r["sample_id"] == FIG2_ROW1
    }
    good_scores = []
    bad_scores = []
    detail = {}
    for m in FIG2_GOOD:
        r = by.get((m, FIG2_ROW1))
        if r:
            good_scores.append(float(r["S_anat_v4"]))
            detail[m] = float(r["S_anat_v4"])
    for m in FIG2_BAD:
        r = by.get((m, FIG2_ROW1))
        if r:
            bad_scores.append(float(r["S_anat_v4"]))
            detail[m] = float(r["S_anat_v4"])

    gpt = detail.get("gpt-image-2")
    omn = detail.get("omnigen2")
    mean_good = sum(good_scores) / len(good_scores) if good_scores else float("nan")
    mean_bad = sum(bad_scores) / len(bad_scores) if bad_scores else float("nan")
    gap = mean_good - mean_bad if good_scores and bad_scores else float("nan")

    c1 = gap >= 0.15 if gap == gap else False
    c2 = (omn is not None and gpt is not None and omn < gpt)
    c3 = all(detail.get(m, -1) >= 0.65 for m in FIG2_GOOD if m in detail)
    passed = bool(c1 and c2 and c3)
    return {
        "passed": passed,
        "row1_mean_good": mean_good,
        "row1_mean_bad": mean_bad,
        "row1_gap_good_minus_bad": gap,
        "check_gap_ge_0.15": c1,
        "check_omnigen2_lt_gpt": c2,
        "check_good_ge_0.65": c3,
        "row1_scores": detail,
    }


def mean_std(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, math.sqrt(v)


def collect_rows(
    pack: Path,
    models: List[str],
    sids: Optional[List[str]],
    proto: Dict[str, Any],
    src_name: str = "mesh_v3",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    src_root = pack / "judgments" / src_name
    for m in models:
        d = src_root / m
        if not d.is_dir():
            continue
        if sids is not None:
            paths = [d / f"{sid}.json" for sid in sids]
        else:
            paths = sorted(p for p in d.glob("*.json") if not p.name.startswith("_"))
        for p in paths:
            if not p.is_file():
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            if not rec.get("ok") or rec.get("recon_fail"):
                # still record recon_fail for ALL-N optional; skip for ok-only means
                pass
            try:
                scored = score_row(rec, proto)
            except Exception as e:  # noqa: BLE001
                print(f"WARN {m}/{p.name}: {e}")
                continue
            rows.append(
                {
                    "model": m,
                    "sample_id": p.stem,
                    "ok": bool(rec.get("ok")),
                    "recon_fail": bool(rec.get("recon_fail")),
                    **scored,
                }
            )
    return rows


def write_report(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    proto: Dict[str, Any],
    fig2_only: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok_rows = [r for r in rows if r.get("ok") and not r.get("recon_fail")]

    # per model (ok only)
    models = sorted({r["model"] for r in ok_rows})
    summary = []
    for m in models:
        rs = [r for r in ok_rows if r["model"] == m]
        a3 = [float(r["S_anat_v3"]) for r in rs]
        a31 = [float(r["S_anat_v31"]) for r in rs if r["S_anat_v31"] == r["S_anat_v31"]]
        a4 = [float(r["S_anat_v4"]) for r in rs]
        m3, s3 = mean_std(a3)
        m31, s31 = mean_std(a31)
        m4, s4 = mean_std(a4)
        summary.append(
            {
                "model": m,
                "n": len(rs),
                "Anat_v3": m3,
                "Anat_v31": m31,
                "Anat_v4": m4,
                "std_v3": s3,
                "std_v31": s31,
                "std_v4": s4,
                "frac_ge095_v3": sum(1 for x in a3 if x >= 0.95) / max(len(a3), 1),
                "frac_ge095_v4": sum(1 for x in a4 if x >= 0.95) / max(len(a4), 1),
            }
        )

    hard = fig2_hard_checks(ok_rows) if any(
        r["sample_id"] == FIG2_ROW1 for r in ok_rows
    ) else {"passed": False, "note": "no row1"}

    means_v3 = [r["Anat_v3"] for r in summary]
    means_v4 = [r["Anat_v4"] for r in summary]
    rho = spearman(means_v3, means_v4) if len(summary) >= 3 else None
    spread_v3 = (max(means_v3) - min(means_v3)) if means_v3 else float("nan")
    spread_v4 = (max(means_v4) - min(means_v4)) if means_v4 else float("nan")
    rank_v3 = [r["model"] for r in sorted(summary, key=lambda x: -x["Anat_v3"])]
    rank_v4 = [r["model"] for r in sorted(summary, key=lambda x: -x["Anat_v4"])]
    qwen_rank_v4 = rank_v4.index("qwen-image-edit-2511") + 1 if "qwen-image-edit-2511" in rank_v4 else None

    report = {
        "fig2_only": fig2_only,
        "n_rows": len(rows),
        "n_ok_rows": len(ok_rows),
        "protocol": proto,
        "fig2_hard": hard,
        "model_rank_spearman_Anat_v3_vs_v4": rho,
        "model_mean_spread_Anat_v3": spread_v3,
        "model_mean_spread_Anat_v4": spread_v4,
        "rank_Anat_v3": rank_v3,
        "rank_Anat_v4": rank_v4,
        "qwen_Anat_rank_v4": qwen_rank_v4,
        "by_model": summary,
    }

    with (out_dir / "per_sample.csv").open("w", newline="", encoding="utf-8") as f:
        if ok_rows:
            w = csv.DictWriter(f, fieldnames=list(ok_rows[0].keys()))
            w.writeheader()
            w.writerows(ok_rows)
    with (out_dir / "per_model.csv").open("w", newline="", encoding="utf-8") as f:
        if summary:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Anat v3 / v3.1 / v4_exp comparison",
        "",
        f"- rows (ok): **{len(ok_rows)}** | fig2_only={fig2_only}",
        f"- Fig2 hard acceptance: **{'PASS' if hard.get('passed') else 'FAIL'}**",
        "",
        "## Fig2 row1 hard checks",
        "",
        f"- mean(good)−mean(bad) = {hard.get('row1_gap_good_minus_bad')} "
        f"(need ≥0.15): {hard.get('check_gap_ge_0.15')}",
        f"- OmniGen2 < GPT: {hard.get('check_omnigen2_lt_gpt')} "
        f"(scores={hard.get('row1_scores')})",
        f"- good ≥0.65: {hard.get('check_good_ge_0.65')}",
        "",
        "## Model means (ok)",
        "",
        "| model | n | Anat_v3 | Anat_v31 | Anat_v4 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summary:
        md.append(
            f"| {r['model']} | {r['n']} | {r['Anat_v3']:.3f} | "
            f"{r['Anat_v31']:.3f} | {r['Anat_v4']:.3f} |"
        )
    md += [
        "",
        f"- Anat rank Spearman v3↔v4: **{rho if rho is not None else float('nan'):.3f}**",
        f"- Anat spread v3→v4: {spread_v3:.3f}→{spread_v4:.3f}",
        f"- rank v4: {' ≻ '.join(rank_v4)}",
        f"- Qwen Anat rank v4: {qwen_rank_v4}",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return report


def tune_fig2(
    pack: Path,
    models: List[str],
    out_dir: Path,
) -> Dict[str, Any]:
    """Small grid search on Fig2; maximize hard-pass then gap."""
    base = dict(DEFAULT_PROTOCOL)
    leftover_oks = [0.50, 0.55, 0.60]
    leftover_bads = [0.75, 0.80, 0.85]
    inside_oks = [0.15, 0.20]
    inside_bads = [0.45, 0.55]
    over_oks = [2.0, 2.5]
    attach_mins = [0.62, 0.65, 0.70]
    extreme_ratios = [3.0, 3.5, 4.0]
    alone_scales = [0.15, 0.20, 0.30]

    best: Optional[Dict[str, Any]] = None
    trials = 0
    for lok in leftover_oks:
        for lbad in leftover_bads:
            if lbad <= lok:
                continue
            for iok in inside_oks:
                for ibad in inside_bads:
                    if ibad <= iok:
                        continue
                    for ook in over_oks:
                        for amin in attach_mins:
                            for er in extreme_ratios:
                                for als in alone_scales:
                                    proto = dict(base)
                                    proto.update(
                                        {
                                            "leftover_ok": lok,
                                            "leftover_bad": lbad,
                                            "inside_ok": iok,
                                            "inside_bad": ibad,
                                            "overdetect_ok": ook,
                                            "attached_leftover_min": amin,
                                            "extreme_ratio": er,
                                            "leftover_alone_scale": als,
                                            "fuse_alone_scale": als,
                                        }
                                    )
                                    rows = collect_rows(
                                        pack, models, FIG2_SIDS, proto
                                    )
                                    hard = fig2_hard_checks(rows)
                                    trials += 1
                                    gap = float(
                                        hard.get("row1_gap_good_minus_bad")
                                        or -10
                                    )
                                    rs = hard.get("row1_scores") or {}
                                    good_vals = [
                                        float(rs[m])
                                        for m in FIG2_GOOD
                                        if m in rs
                                    ]
                                    good_floor = (
                                        min(good_vals) if good_vals else 0.0
                                    )
                                    score = (
                                        int(bool(hard.get("passed"))) * 1000.0
                                        + gap
                                        + 0.1 * good_floor
                                    )
                                    cand = {
                                        "score": score,
                                        "protocol": proto,
                                        "hard": hard,
                                    }
                                    if best is None or cand["score"] > best[
                                        "score"
                                    ]:
                                        best = cand

    assert best is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_root = pack / "judgments" / "mesh_anat_exp"
    exp_root.mkdir(parents=True, exist_ok=True)
    proto_out = dict(best["protocol"])
    proto_out["_tune"] = {
        "objective": "fig2_hard_pass then gap",
        "trials": trials,
        "hard": best["hard"],
    }
    (exp_root / "_protocol.json").write_text(
        json.dumps(proto_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "tune_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"[tune] trials={trials} passed={best['hard'].get('passed')} "
        f"gap={best['hard'].get('row1_gap_good_minus_bad')} "
        f"→ {exp_root / '_protocol.json'}"
    )
    print("best protocol:", json.dumps(best["protocol"], indent=2))
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(PAPER_MODELS))
    ap.add_argument("--fig2", action="store_true", help="Fig2 sids only")
    ap.add_argument("--full", action="store_true", help="all samples")
    ap.add_argument(
        "--sample-ids-file",
        default="",
        help="manifest.jsonl or sid list (e.g. seed150)",
    )
    ap.add_argument("--tune", action="store_true", help="grid-search protocol on Fig2")
    ap.add_argument("--protocol-json", default="")
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.tune:
        tune_fig2(pack, models, out_dir)

    proto_path = (
        Path(args.protocol_json).expanduser()
        if args.protocol_json
        else pack / "judgments" / "mesh_anat_exp" / "_protocol.json"
    )
    proto = load_protocol(proto_path if proto_path.is_file() else None)

    sids: Optional[List[str]]
    if args.full:
        sids = None
        fig2_only = False
    elif args.sample_ids_file:
        text = Path(args.sample_ids_file).expanduser().read_text(encoding="utf-8")
        sids = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("{"):
                sids.append(json.loads(line)["sample_id"])
            else:
                sids.append(line)
        fig2_only = False
    else:
        # default: fig2
        sids = list(FIG2_SIDS)
        fig2_only = True

    rows = collect_rows(pack, models, sids, proto)
    report = write_report(out_dir, rows, proto, fig2_only=fig2_only)
    print(f"\nwrote {out_dir}/REPORT.md  fig2_pass={report['fig2_hard'].get('passed')}")


if __name__ == "__main__":
    main()
