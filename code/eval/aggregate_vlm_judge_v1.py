#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate VLM Judge v1 judgments into summary tables + HTML report.

Protocol: docs/02_pipeline_design/eval_vlm_judge_v1.md
  S_count / S_id / S_anat / S_inter / S_instr / S_qual ∈ [0,1]
  low confidence → excluded from main-table means (still counted in coverage)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

AXES = ("count", "id", "anat", "inter", "instr", "qual")
YN = {"yes": 1.0, "partial": 0.5, "no": 0.0}


def _yn(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().lower()
    return YN.get(s)


def score_axes(j: dict) -> Dict[str, Optional[float]]:
    """Map one judgment JSON → six scalars (may be None if field missing)."""
    s: Dict[str, Optional[float]] = {a: None for a in AXES}

    if "count_pass" in j:
        s["count"] = 1.0 if j["count_pass"] else 0.0
    elif j.get("count_detected") is not None and j.get("count_expected") is not None:
        s["count"] = 1.0 if j["count_detected"] == j["count_expected"] else 0.0

    id_rows = j.get("id") or []
    yes_vals = []
    for row in id_rows:
        m = str((row or {}).get("match", "")).lower()
        if m == "uncertain":
            continue
        if m == "yes":
            yes_vals.append(1.0)
        elif m == "no":
            yes_vals.append(0.0)
    if yes_vals:
        s["id"] = sum(yes_vals) / len(yes_vals)
    elif id_rows:
        s["id"] = None  # all uncertain
    else:
        s["id"] = None

    errs = j.get("anat_errors") or []
    n_err = len(errs) if isinstance(errs, list) else 0
    if "anat_pass" in j and j["anat_pass"] and n_err == 0:
        s["anat"] = 1.0
    elif "anat_pass" in j and (not j["anat_pass"]) and n_err == 0:
        s["anat"] = 0.0  # flagged fail without typed errors
    elif isinstance(errs, list):
        s["anat"] = 1.0 - min(1.0, 0.25 * n_err)
    else:
        s["anat"] = None

    inter = j.get("inter") or {}
    knives = [
        _yn(inter.get("semantic")),
        _yn(inter.get("contact_points")),
        _yn(inter.get("no_pathological_penetration")),
    ]
    knives_ok = [x for x in knives if x is not None]
    s["inter"] = sum(knives_ok) / len(knives_ok) if knives_ok else None

    qa = j.get("instr_qa") or []
    qa_vals = [_yn((q or {}).get("a")) for q in qa]
    qa_ok = [x for x in qa_vals if x is not None]
    s["instr"] = sum(qa_ok) / len(qa_ok) if qa_ok else None

    if "qual_pass" in j:
        qp = j["qual_pass"]
        if isinstance(qp, bool):
            s["qual"] = 1.0 if qp else 0.0
        else:
            s["qual"] = _yn(qp)
    return s


def anat_pass(j: dict) -> Optional[bool]:
    if "anat_pass" in j:
        return bool(j["anat_pass"])
    errs = j.get("anat_errors")
    if errs is None:
        return None
    return len(errs) == 0


def mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def fmt(x: Optional[float], digits: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{digits}f}"


def pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%"


def load_pack(pack: Path) -> Tuple[Dict[str, dict], List[str]]:
    manifest = {}
    order = []
    with open(pack / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sid = row["sample_id"]
            manifest[sid] = row
            order.append(sid)
    return manifest, order


def load_judgments(pack: Path, model_id: str) -> Dict[str, dict]:
    d = pack / "judgments" / "vlm_judge_v1" / model_id
    out = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        sid = (j.get("_meta") or {}).get("sample_id") or p.stem
        out[sid] = j
    return out


# Main table display order: closed source first, open source last (see eval_model_zoo.md); within the group, the order is fixed according to the list
CLOSED_MODEL_ORDER = (
    "gpt-image-2",
    "gemini-3-pro-image",
    "seedream-5-pro",
)
OPEN_MODEL_ORDER = (
    "flux1-kontext-dev",
    "qwen-image-edit-2511",
    "omnigen2",
    "uno",
    "ace",
    "bagel",
    "dreamo",
)
MODEL_DISPLAY_ORDER = CLOSED_MODEL_ORDER + OPEN_MODEL_ORDER


def order_models_closed_then_open(models: List[str]) -> List[str]:
    """Closed source → Open source; the list is in fixed order, and the items outside the list are appended at the end (stable dictionary order). """
    rank = {mid: i for i, mid in enumerate(MODEL_DISPLAY_ORDER)}
    known = [m for m in MODEL_DISPLAY_ORDER if m in models]
    unknown = sorted(m for m in models if m not in rank)
    return known + unknown


def list_models(pack: Path) -> List[str]:
    """Discover models from outputs and any judgments tree (VLM / ArcFace / HPS / mesh / Instr)."""
    names: set = set()
    roots = [
        pack / "outputs",
        pack / "judgments" / "vlm_judge_v1",
        pack / "judgments" / "arcface_v1",
        pack / "judgments" / "hpsv2",
        pack / "judgments" / "mesh_v3",
        pack / "judgments" / "instr_v2",
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.iterdir():
            if p.is_dir() and not p.name.startswith("_"):
                names.add(p.name)
    return sorted(names)


def aggregate(pack: Path, models: Optional[List[str]] = None) -> dict:
    manifest, order = load_pack(pack)
    models = models or list_models(pack)
    per_sample = []  # flat rows
    by_model: Dict[str, dict] = {}

    for mid in models:
        outs = {p.stem for p in (pack / "outputs" / mid).glob("*.png")} if (pack / "outputs" / mid).is_dir() else set()
        juds = load_judgments(pack, mid)
        axis_vals = {a: [] for a in AXES}
        axis_vals_all = {a: [] for a in AXES}  # including low conf
        pass_count = pass_anat = pass_qual = 0
        n_main = n_low = n_jud = 0
        anat_err_types = Counter()
        inter_knife = {k: Counter() for k in ("semantic", "contact_points", "no_pathological_penetration")}
        conf_hist = Counter()
        by_cat_axis = defaultdict(lambda: {a: [] for a in AXES})

        for sid in order:
            j = juds.get(sid)
            if not j:
                continue
            n_jud += 1
            scores = score_axes(j)
            conf = str(j.get("confidence") or "unknown").lower()
            conf_hist[conf] += 1
            low = conf == "low"
            if low:
                n_low += 1
            else:
                n_main += 1

            for a in AXES:
                v = scores[a]
                if v is None:
                    continue
                axis_vals_all[a].append(v)
                if not low:
                    axis_vals[a].append(v)
                    by_cat_axis[manifest[sid]["cat"]][a].append(v)

            if not low:
                if scores["count"] == 1.0:
                    pass_count += 1
                ap = anat_pass(j)
                if ap is True:
                    pass_anat += 1
                if scores["qual"] == 1.0:
                    pass_qual += 1

            for e in j.get("anat_errors") or []:
                anat_err_types[str((e or {}).get("type") or "unknown")] += 1
            inter = j.get("inter") or {}
            for k in inter_knife:
                inter_knife[k][str(inter.get(k) or "missing").lower()] += 1

            per_sample.append({
                "sample_id": sid,
                "cat": manifest[sid]["cat"],
                "model_id": mid,
                "confidence": conf,
                "in_main": not low,
                "has_output": sid in outs,
                **{f"S_{a}": scores[a] for a in AXES},
                "anat_pass": anat_pass(j),
                "count_pass": bool(j.get("count_pass")) if "count_pass" in j else None,
                "qual_pass": bool(j.get("qual_pass")) if "qual_pass" in j else None,
                "overall_notes": j.get("overall_notes") or "",
                "gen_relpath": (j.get("_meta") or {}).get("gen_relpath") or f"outputs/{mid}/{sid}.png",
                "judge_model": (j.get("_meta") or {}).get("judge_model"),
            })

        def m(a):
            return mean(axis_vals[a])

        by_model[mid] = {
            "model_id": mid,
            "n_manifest": len(order),
            "n_outputs": len(outs),
            "n_judgments": n_jud,
            "n_main": n_main,
            "n_low_conf": n_low,
            "coverage_output": len(outs) / len(order) if order else 0.0,
            "coverage_judge": n_jud / len(order) if order else 0.0,
            "means_main": {a: m(a) for a in AXES},
            "means_all": {a: mean(axis_vals_all[a]) for a in AXES},
            "pass_rate_main": {
                "count": (pass_count / n_main) if n_main else None,
                "anat": (pass_anat / n_main) if n_main else None,
                "qual": (pass_qual / n_main) if n_main else None,
            },
            "struct_triplet_main": {  # Anat / Inter / Count focus
                "anat": m("anat"),
                "inter": m("inter"),
                "count": m("count"),
            },
            "confidence": dict(conf_hist),
            "anat_error_types": dict(anat_err_types),
            "inter_knives": {k: dict(v) for k, v in inter_knife.items()},
            "by_cat_means_main": {
                cat: {a: mean(vals[a]) for a in AXES}
                for cat, vals in sorted(by_cat_axis.items())
            },
            "missing_outputs": sorted(set(order) - outs),
            "missing_judgments": sorted(outs - set(juds)),
        }

    # Table order: closed source first, open source last (no longer ranked by score)
    ranking = order_models_closed_then_open(list(models))

    return {
        "pack": str(pack),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "vlm_judge_v1",
        "n_samples": len(order),
        "models": models,
        "ranking": ranking,
        "by_model": by_model,
        "per_sample": per_sample,
        "cats": sorted({manifest[s]["cat"] for s in order}),
        "note": "Main-table means exclude confidence=low (protocol §2.4).",
    }


def write_csv(rows: List[dict], path: Path, fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def render_html(summary: dict, pack_url: str = "pack") -> str:
    """Clean public page: metric glossary + table + bar charts (same layout as smoke100_v2)."""
    del pack_url  # kept for call-site compat; images not shown on this page
    metrics = ["Count", "ID", "Anat", "Inter", "Instr", "Qual", "Overall"]
    bm = summary["by_model"]
    rows_data = []
    for mid in summary["ranking"]:
        m = bm[mid]
        means = m["means_main"]
        vals = {
            "Count": means["count"],
            "ID": means["id"],
            "Anat": means["anat"],
            "Inter": means["inter"],
            "Instr": means["instr"],
            "Qual": means["qual"],
        }
        present = [v for v in vals.values() if v is not None]
        overall = mean(present) if present else None
        rows_data.append({"model": mid, **vals, "Overall": overall})

    trs = []
    for r in rows_data:
        cells = "".join(f'<td class="num">{fmt(r[k])}</td>' for k in metrics)
        trs.append(f'<tr><td class="model">{r["model"]}</td>{cells}</tr>')
    table_body = "\n".join(trs)

    colors = {
        "Count": "#3b82f6", "ID": "#8b5cf6", "Anat": "#ef4444",
        "Inter": "#f59e0b", "Instr": "#10b981", "Qual": "#06b6d4", "Overall": "#1f2937",
    }
    chart_blocks = []
    for k in metrics:
        vals = [(r["model"], r[k]) for r in rows_data if r[k] is not None]
        if not vals:
            continue
        vmax = max(abs(v) for _, v in vals) or 1.0
        scale = 1.0 if vmax <= 1.05 else vmax
        bars = []
        for name, v in vals:
            pct_w = max(0.0, min(100.0, 100.0 * float(v) / scale))
            bars.append(
                f'<div class="bar-row"><span class="bn">{name}</span>'
                f'<div class="track"><div class="fill" style="width:{pct_w:.1f}%;background:{colors[k]}"></div></div>'
                f'<span class="bv">{fmt(v)}</span></div>'
            )
        chart_blocks.append(f'<div class="chart"><h3>{k}</h3>{"".join(bars)}</div>')

    return f"""<!DOCTYPE html>
<html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MPIE smoke100 · plan① All axis VQA</title>
<style>
:root {{ --bg:#f7f8fa; --card:#fff; --line:#e8eaee; --text:#1f2430; --muted:#6b7280; }}
* {{ box-sizing:border-box }}
body {{ margin:0; padding:28px 24px 48px; font-family:-apple-system,"PingFang SC",Helvetica,sans-serif;
  background:var(--bg); color:var(--text); }}
h1 {{ font-size:22px; margin:0 0 6px; font-weight:700; }}
.sub {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
h2 {{ font-size:15px; margin:28px 0 10px; font-weight:600; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:8px 16px;
  background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:22px; }}
.metrics dt {{ font-weight:600; font-size:13px; }}
.metrics dd {{ margin:0 0 6px; color:var(--muted); font-size:12.5px; line-height:1.45; }}
table {{ width:100%; max-width:960px; border-collapse:collapse; background:var(--card);
  border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:13.5px; }}
th, td {{ padding:10px 12px; border-bottom:1px solid #f0f2f5; }}
th {{ background:#fafbfc; color:var(--muted); font-size:12px; font-weight:600; text-align:right; }}
th:first-child, td.model {{ text-align:left; font-weight:600; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr:last-child td {{ border-bottom:0; }}
tr:hover td {{ background:#fafbff; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; margin-top:8px; max-width:1100px; }}
.chart {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.chart h3 {{ margin:0 0 10px; font-size:13px; font-weight:600; }}
.bar-row {{ display:grid; grid-template-columns:140px 1fr 48px; gap:8px; align-items:center; margin:5px 0; }}
.bn {{ font-size:11px; color:#374151; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.track {{ height:10px; background:#eef0f3; border-radius:5px; overflow:hidden; }}
.fill {{ height:100%; border-radius:5px; }}
.bv {{ font-size:11px; color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }}
.note {{ color:var(--muted); font-size:12px; margin-top:16px; }}
</style></head><body>
<h1>MPIE smoke100 · plan①(all axes VQA · Judge v1）</h1>
<p class="sub">N = {summary['n_samples']} · {summary['generated_at'][:10]} ·
  <a href="/">← Review home page</a> ·
  <a href="/eval_outputs/smoke100_v2/">Compare: Scheme② VQA+ArcFace+HPS</a>
</p>

<h2>Indicator description</h2>
<dl class="metrics">
  <div><dt>Count</dt><dd>Is the number of people correct? Number of people generating graphs = Expected number of people → 1,otherwise 0。</dd></div>
  <div><dt>ID</dt><dd>Does the character look like the reference picture? Each reference yes=1 / no=0 Take the mean (uncertain eliminated).</dd></div>
  <div><dt>Anat</dt><dd>limbs/Is the anatomy normal? Closed set error list, the more mistakes, the lower the score:1 − min(1, 0.25·n_err)。</dd></div>
  <div><dt>Inter</dt><dd>Whether the interaction is established. Triple Mean: Semantics / contact position / No pathological mold wear (yes=1 / partial=0.5 / no=0）。</dd></div>
  <div><dt>Instr</dt><dd>Are you keeping up with the details of the instructions? Average several atomic questions and answers.</dd></div>
  <div><dt>Qual</dt><dd>Image quality/Perception.VLM pass/fail(plan② Change to HPSv2）。</dd></div>
  <div><dt>Overall</dt><dd>A simple average of the six available scores above. Sort first Anat/Inter/Count。</dd></div>
</dl>

<h2>Summary table</h2>
<table>
<thead><tr><th>Model</th>{''.join(f'<th>{k}</th>' for k in metrics)}</tr></thead>
<tbody>
{table_body}
</tbody>
</table>

<h2>Comparison of items</h2>
<div class="charts">
{''.join(chart_blocks)}
</div>

<p class="note">The higher the score, the better. Main table mean elimination confidence=low. See the same catalog for details summary.json / leaderboard.csv / per_sample.csv。</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--out", default="", help="report output dir (default: <bench>/data/eval_outputs/smoke100)")
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    out = Path(args.out).expanduser() if args.out else (
        Path("data") / "eval_outputs" / "smoke100"
    )
    out.mkdir(parents=True, exist_ok=True)

    summary = aggregate(pack, args.models)
    # slim json for web (full per_sample kept; drop huge notes already short)
    (out / "summary.json").write_text(
        json.dumps({k: v for k, v in summary.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lb_rows = []
    for i, mid in enumerate(summary["ranking"], 1):
        m = summary["by_model"][mid]
        means = m["means_main"]
        st = m["struct_triplet_main"]
        lb_rows.append({
            "rank": i,
            "model_id": mid,
            "n_outputs": m["n_outputs"],
            "n_judgments": m["n_judgments"],
            "n_main": m["n_main"],
            "struct": mean([st[k] for k in ("anat", "inter", "count") if st[k] is not None]),
            **{f"S_{a}": means[a] for a in AXES},
            **{f"pass_{k}": m["pass_rate_main"][k] for k in ("count", "anat", "qual")},
        })
    write_csv(lb_rows, out / "leaderboard.csv", list(lb_rows[0].keys()) if lb_rows else ["rank"])

    ps_fields = [
        "sample_id", "cat", "model_id", "confidence", "in_main",
        "S_count", "S_id", "S_anat", "S_inter", "S_instr", "S_qual",
        "anat_pass", "count_pass", "qual_pass", "gen_relpath", "overall_notes", "judge_model",
    ]
    write_csv(summary["per_sample"], out / "per_sample.csv", ps_fields)

    # symlink pack for static image serving
    link = out / "pack"
    if link.is_symlink() or link.exists():
        if link.is_symlink():
            link.unlink()
        elif link.is_dir():
            pass  # don't delete real dir
        else:
            link.unlink()
    if not link.exists():
        link.symlink_to(pack)

    html = render_html(summary, pack_url="pack")
    (out / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "n_samples": summary["n_samples"],
        "models": summary["models"],
        "ranking": summary["ranking"],
        "leaderboard": [
            {"model": r["model_id"], "struct": r["struct"], "S_anat": r["S_anat"], "S_inter": r["S_inter"], "S_count": r["S_count"]}
            for r in lb_rows
        ],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
