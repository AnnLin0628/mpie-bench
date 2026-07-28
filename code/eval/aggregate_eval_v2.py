#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate MPIE eval protocol v2 tables.

Main table = Count + ArcFace ID + VLM/mesh Anat/Inter + Instr v2 + HPSv2 Qual.
No Overall (HPSv2 scale differs).

See: docs/02_pipeline_design/eval_protocol_v2.md
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from aggregate_vlm_judge_v1 import (  # noqa: E402
    aggregate,
    fmt,
    mean,
    order_models_closed_then_open,
)


VLM_KEEP = ("count", "anat", "inter", "instr")


def load_side_judgments(pack: Path, kind: str, model_id: str) -> dict:
    """kind in {arcface_v1, hpsv2, instr_v2}."""
    d = pack / "judgments" / kind / model_id
    out = {}
    if not d.is_dir():
        return out
    if kind.startswith("arcface"):
        score_key = "S_id"
    elif kind.startswith("hps"):
        score_key = "hpsv2"
    elif kind.startswith("instr"):
        score_key = "S_instr"
    else:
        score_key = "S_id"
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j.get("ok") is False:
            continue
        if j.get("confidence") == "low" and kind.startswith("instr"):
            continue
        if j.get(score_key) is None:
            continue
        sid = (j.get("sample_id") or (j.get("_meta") or {}).get("sample_id") or p.stem)
        out[sid] = j
    return out


def aggregate_v2(pack: Path, models=None) -> dict:
    base = aggregate(pack, models)
    models = base["models"]
    by_model = {}
    any_instr_v2 = False
    for mid in models:
        m = base["by_model"][mid]
        means = {a: m["means_main"][a] for a in VLM_KEEP}
        st = m["struct_triplet_main"]
        struct = mean([st[k] for k in ("anat", "inter", "count") if st[k] is not None])

        arc = load_side_judgments(pack, "arcface_v1", mid)
        hps = load_side_judgments(pack, "hpsv2", mid)
        instr2 = load_side_judgments(pack, "instr_v2", mid)
        id_scores, fvr, mrate, mcp = [], [], [], []
        for j in arc.values():
            if j.get("S_id") is not None:
                id_scores.append(float(j["S_id"]))
            if j.get("face_visible_rate") is not None:
                fvr.append(float(j["face_visible_rate"]))
            if j.get("match_rate") is not None:
                mrate.append(float(j["match_rate"]))
            if j.get("M_CP") is not None:
                mcp.append(float(j["M_CP"]))
        hps_scores = [float(j["hpsv2"]) for j in hps.values() if j.get("hpsv2") is not None]
        instr_v2_scores = [
            float(j["S_instr"]) for j in instr2.values() if j.get("S_instr") is not None
        ]
        asymm_scores = [
            float(j["S_instr_asymm"])
            for j in instr2.values()
            if j.get("S_instr_asymm") is not None
        ]
        role_duty_scores = [
            float(j["S_instr_role_duty"])
            for j in instr2.values()
            if j.get("S_instr_role_duty") is not None
        ]
        prop_obj_scores = [
            float(j["S_instr_prop_object"])
            for j in instr2.values()
            if j.get("S_instr_prop_object") is not None
        ]
        if instr_v2_scores:
            any_instr_v2 = True
            s_instr = mean(instr_v2_scores)
            instr_src = "instr_v2"
            s_asymm = mean(asymm_scores)
            s_role_duty = mean(role_duty_scores)
            s_prop_object = mean(prop_obj_scores)
            p_perfect = sum(1 for s in instr_v2_scores if s >= 0.999) / len(
                instr_v2_scores
            )
        else:
            s_instr = means["instr"]
            instr_src = "vlm_v1"
            s_asymm = None
            s_role_duty = None
            s_prop_object = None
            p_perfect = None

        n_out = m["n_outputs"] or max(len(arc), len(hps), 0)
        by_model[mid] = {
            "model_id": mid,
            "n_outputs": n_out,
            "n_vlm": m["n_judgments"],
            "n_main_vlm": m["n_main"],
            "n_arcface": len(arc),
            "n_hpsv2": len(hps),
            "n_instr_v2": len(instr2),
            "struct": struct,
            "S_count": means["count"],
            "S_anat": means["anat"],
            "S_inter": means["inter"],
            "S_instr": s_instr,
            "S_instr_asymm": s_asymm,
            "S_instr_role_duty": s_role_duty,
            "S_instr_prop_object": s_prop_object,
            "p_perfect": p_perfect,
            "S_instr_source": instr_src,
            "S_instr_v1": means["instr"],
            "S_id": mean(id_scores),
            "face_visible_rate": mean(fvr),
            "match_rate": mean(mrate),
            "M_CP": mean(mcp),
            "HPSv2": mean(hps_scores),
            "status": {
                "vlm_four_axis": m["n_judgments"] > 0,
                "arcface": len(arc) > 0,
                "hpsv2": len(hps) > 0,
                "instr_v2": len(instr2) > 0,
            },
            "appendix_vlm_id": m["means_main"]["id"],
            "appendix_vlm_qual": m["means_main"]["qual"],
        }

    ranking = order_models_closed_then_open(list(models))
    note = (
        "Main columns: Count/ID/Anat/Inter/Instr/Qual. "
        "Instr = weighted S_instr (asymm 0.50 / role_duty 0.35 / prop_object 0.15)."
    )
    if any_instr_v2:
        note += " Instr from judgments/instr_v2."
    return {
        "protocol": "eval_protocol_v2",
        "pack": str(pack),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": base["n_samples"],
        "models": models,
        "ranking": ranking,
        "by_model": by_model,
        "instr_protocol": "instr_v2_weighted" if any_instr_v2 else "vlm_v1",
        "note": note,
    }


def _display_row(m: dict) -> dict:
    """Public table metrics. Prefer ArcFace/HPSv2; Instr = weighted total."""
    sid = m["S_id"] if m["S_id"] is not None else m.get("appendix_vlm_id")
    qual = m["HPSv2"] if m["HPSv2"] is not None else m.get("appendix_vlm_qual")
    return {
        "Count": m["S_count"],
        "ID": sid,
        "Anat": m["S_anat"],
        "Inter": m["S_inter"],
        "Instr": m["S_instr"],
        "Instr_source": m.get("S_instr_source") or "vlm_v1",
        "Instr_asymm": m.get("S_instr_asymm"),
        "Instr_role_duty": m.get("S_instr_role_duty"),
        "Instr_prop_object": m.get("S_instr_prop_object"),
        "p_perfect": m.get("p_perfect"),
        "Qual": qual,
    }


def render_html(summary: dict) -> str:
    metrics = ["Count", "ID", "Anat", "Inter", "Instr", "Qual"]
    metric_headers = {
        "Count": "Count",
        "ID": "ID",
        "Anat": "Anat",
        "Inter": "Inter",
        "Instr": 'Instr<br><span class="th-hint">weighted total score</span>',
        "Qual": 'Qual<br><span class="th-hint">HPS≈0.20–0.35</span>',
    }
    rows_data = []
    for mid in summary["ranking"]:
        m = summary["by_model"][mid]
        d = _display_row(m)
        rows_data.append({"model": mid, **d, "n": m["n_outputs"], "status": m["status"]})

    trs = []
    for r in rows_data:
        cells = "".join(f'<td class="num">{fmt(r[k])}</td>' for k in metrics)
        trs.append(f'<tr><td class="model">{r["model"]}</td>{cells}</tr>')
    table_body = "\n".join(trs)

    chart_blocks = []
    colors = {
        "Count": "#3b82f6",
        "ID": "#8b5cf6",
        "Anat": "#ef4444",
        "Inter": "#f59e0b",
        "Instr": "#10b981",
        "Qual": "#06b6d4",
    }
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

    id_src = (
        "ArcFace"
        if any(summary["by_model"][m]["status"]["arcface"] for m in summary["models"])
        else "VLM(temporary)"
    )
    qual_src = (
        "HPSv2"
        if any(summary["by_model"][m]["status"]["hpsv2"] for m in summary["models"])
        else "VLM(temporary)"
    )

    return f"""<!DOCTYPE html>
<html lang=zh><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>MPIE smoke100 · plan② VQA+ArcFace+HPS</title>
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
th .th-hint {{ display:block; font-weight:500; font-size:10px; color:#9ca3af; margin-top:2px; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; margin-top:8px; }}
.chart {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.bar-row {{ display:grid; grid-template-columns:140px 1fr 48px; gap:8px; align-items:center; margin:5px 0; }}
.track {{ height:10px; background:#eef0f3; border-radius:5px; overflow:hidden; }}
.fill {{ height:100%; border-radius:5px; }}
.note {{ color:var(--muted); font-size:12px; margin-top:16px; }}
</style></head><body>
<h1>MPIE smoke100 · plan②（VQA Anat/Inter + ArcFace + HPS）</h1>
<p class="sub">N = {summary['n_samples']} · {summary['generated_at'][:10]} ·
  <a href="/">← Review home page</a> ·
  <a href="/eval_outputs/smoke100_v3/">Compare: Scheme③ Mesh</a>
</p>
<h2>Indicator description</h2>
<dl class="metrics">
  <div><dt>Count</dt><dd>Number of people.VLM。</dd></div>
  <div><dt>ID</dt><dd>Recognize your face. source:{id_src}。</dd></div>
  <div><dt>Anat / Inter</dt><dd>plan②still VLM(plan③Just changed mesh）。</dd></div>
  <div><dt>Instr</dt><dd>priority Instr v2 Weighted total score.</dd></div>
  <div><dt>Qual</dt><dd>HPSv2（{qual_src}),typical 0.20–0.35。</dd></div>
</dl>
<h2>Summary table</h2>
<table>
<thead><tr><th>Model</th>{''.join(f'<th>{metric_headers[k]}</th>' for k in metrics)}</tr></thead>
<tbody>
{table_body}
</tbody>
</table>
<h2>Comparison of items</h2>
<div class="charts">{''.join(chart_blocks)}</div>
<p class="note">{summary.get('note','')}</p>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from pack_io import pack_root

    pack = pack_root(args.pack or None)
    out = (
        Path(args.out).expanduser()
        if args.out
        else Path("data") / "eval_outputs" / "smoke100_v2"
    )
    out.mkdir(parents=True, exist_ok=True)
    summary = aggregate_v2(pack)
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "index.html").write_text(render_html(summary), encoding="utf-8")
    print(json.dumps({"out": str(out), "ranking": summary["ranking"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
