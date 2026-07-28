#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPIE-Bench Pilot case: Build self-contained HTML Comparison report.

read pilot_cases/pilot_generation_results/pilot_scores three json,output
docs/01_research/pilot_case_study_report.html：
- top summary matrix(Case×Model×six dimensions)
- One piece per case: reference picture + instruction → Each model generates diagrams side by side + Six-dimensional fraction(Low score mark red) + VLM in accordance with
The image is compressed to base64 inline(≤640px), the report is completely self-contained and can be used directly as Artifact release.
usage: python3 build_report.py
"""
import base64
import html as html_mod
import io
import json
from pathlib import Path

from PIL import Image

BENCH = Path(".")
CASES_FILE = BENCH / "data/manifests/pilot_cases.json"
RESULTS_FILE = BENCH / "data/manifests/pilot_generation_results.json"
SCORES_FILE = BENCH / "data/manifests/pilot_scores.json"
IMG_DIR = BENCH / "data/crops/pilot_case_study"
REF_CACHE = IMG_DIR / "_refs"
OUT = BENCH / "docs/01_research/pilot_case_study_report.html"

MODEL_LABELS = {"gpt-image-2": "GPT-Image-2", "nano-banana-pro": "Nano Banana Pro (Gemini)",
                "seedream-5": "Seedream 5"}
DIMS = [("count", "Count Number of people"), ("id", "ID Keep"), ("anat", "Anat anatomy"),
        ("inter", "Inter Interaction"), ("instr", "Instr instruction"), ("qual", "Qual image quality")]


def thumb_b64(path: Path, max_side: int = 640, q: int = 78) -> str:
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=q)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def dim_values(s: dict) -> dict:
    """Put one VLM Scoring is normalized into six dimensions 1-5 numerical value(count Use right and wrong mapping 5/1)。"""
    if not s or s.get("error"):
        return {}
    ids = s.get("id_scores") or []
    return {
        "count": 5 if s.get("count_correct") else 1,
        "id": round(sum(ids) / len(ids), 1) if ids else None,
        "anat": s.get("anat_score"),
        "inter": s.get("inter_score"),
        "instr": s.get("instr_score"),
        "qual": s.get("qual_score"),
    }


def cell(v) -> str:
    if v is None:
        return "<td class='na'>—</td>"
    cls = "bad" if v <= 2 else ("mid" if v <= 3 else "good")
    return f"<td class='{cls}'>{v}</td>"


def esc(t) -> str:
    return html_mod.escape(str(t or ""))


def main():
    cases = json.load(open(CASES_FILE))["cases"]
    runs = json.load(open(RESULTS_FILE))["runs"]
    sc = json.load(open(SCORES_FILE)) if SCORES_FILE.exists() else {"scores": {}}
    scores = sc.get("scores", {})
    judge = sc.get("judge_model", "?")
    models = sorted({r["model"] for r in runs.values()})

    # ---- summary matrix ----
    rows = []
    for c in cases:
        tds = [f"<td class='case'>{esc(c['case_id'])}<span class='tag'>{c['density']}·{c['n_person']}people</span></td>"]
        for m in models:
            key = f"{c['case_id']}::{m}"
            run = runs.get(key)
            if run is None:
                tds.append("<td class='na' colspan='6'>—</td>")
                continue
            if not run.get("ok"):
                tds.append(f"<td class='na' colspan='6'>{esc((run.get('error') or 'not run')[:60])}</td>")
                continue
            dv = dim_values(scores.get(key, {}))
            if not dv:
                tds.append("<td class='na' colspan='6'>Not rated</td>")
                continue
            tds.extend(cell(dv.get(k)) for k, _ in DIMS)
        rows.append("<tr>" + "".join(tds) + "</tr>")

    mhead = "".join(f"<th colspan='6'>{esc(MODEL_LABELS.get(m, m))}</th>" for m in models)
    dhead = "".join(f"<th class='dim'>{lbl.split()[0]}</th>" for m in models for _, lbl in DIMS)
    matrix = f"""<table class='matrix'>
<tr><th rowspan='2'>Case</th>{mhead}</tr><tr>{dhead}</tr>{''.join(rows)}</table>"""

    # ---- Model equalization(Binning by density) ----
    agg_rows = []
    for m in models:
        by_density = {}
        for c in cases:
            dv = dim_values(scores.get(f"{c['case_id']}::{m}", {}))
            if dv:
                by_density.setdefault(c["density"], []).append(dv)
        parts = []
        for d in ("C0", "C1", "C2", "C3"):
            if d == "C3":
                lst = list(by_density.get("C3", [])) + list(by_density.get("C4", []))
            else:
                lst = by_density.get(d, [])
            if lst:
                a = round(sum(x['anat'] for x in lst if x.get('anat')) / len(lst), 1)
                i = round(sum(x['inter'] for x in lst if x.get('inter')) / len(lst), 1)
                parts.append(f"<td>{d}: anat <b>{a}</b> / inter <b>{i}</b> (n={len(lst)})</td>")
        agg_rows.append(f"<tr><td>{esc(MODEL_LABELS.get(m, m))}</td>{''.join(parts)}</tr>")
    agg = f"<table class='agg'><tr><th>Model</th><th colspan='4'>According to contact density Anat/Inter Divide equally (C3=high contact)</th></tr>{''.join(agg_rows)}</table>"

    # ---- case-by-case block ----
    blocks = []
    for c in cases:
        refs_html = "".join(
            f"<figure><img src='{thumb_b64(REF_CACHE / r['url'].rstrip('/').split('/')[-1], 320)}'>"
            f"<figcaption>{esc(r['name'])}</figcaption></figure>"
            for r in c["refs"] if (REF_CACHE / r['url'].rstrip('/').split('/')[-1]).exists())
        gen_html = []
        for m in models:
            key = f"{c['case_id']}::{m}"
            if key not in runs:
                continue
            run = runs[key]
            s = scores.get(key, {})
            dv = dim_values(s)
            if run.get("ok") and run.get("local_file") and (IMG_DIR / run["local_file"]).exists():
                img = (f"<a href='/images/{run['local_file']}' target='_blank'>"
                       f"<img src='{thumb_b64(IMG_DIR / run['local_file'])}'></a>")
            else:
                img = f"<div class='fail'>Build failed<br><small>{esc((run.get('error') or '')[:120])}</small></div>"
            score_line = " · ".join(
                f"{lbl.split()[0]} <b class='{'r' if (dv.get(k) or 5) <= 2 else ''}'>{dv.get(k, '—')}</b>"
                for k, lbl in DIMS) if dv else "<i>Not rated</i>"
            issues = ""
            for field, tag in (("anat_issues", "Anat"), ("inter_issues", "Inter")):
                for it in (s.get(field) or []):
                    issues += f"<li><b>{tag}:</b> {esc(it)}</li>"
            notes = esc(s.get("overall_notes", ""))
            gen_html.append(f"""<div class='gen'>
<h4>{esc(MODEL_LABELS.get(m, m))}</h4>{img}
<p class='scores'>{score_line}</p>
{f"<ul class='issues'>{issues}</ul>" if issues else ""}
{f"<p class='notes'>{notes}</p>" if notes else ""}</div>""")
        blocks.append(f"""<section>
<h3>{esc(c['case_id'])} <span class='tag'>{c['density']} · {c['n_person']}people · {esc(c['interaction_type'])}</span></h3>
<p class='inst'>「{esc(c['prompt_zh'])}」</p>
<p class='inst en'>{esc(c['prompt'])}</p>
<div class='refs'>{refs_html}</div>
<div class='gens'>{''.join(gen_html)}</div>
</section>""")

    n_ok = sum(1 for r in runs.values() if r.get("ok"))
    html = f"""<title>MPIE-Bench Pilot case study · 10Complex multi-person interaction</title>
<style>
:root {{ --fg:#1a1a1e; --bg:#fff; --muted:#667; --line:#e4e4ea; --card:#f7f7fa;
  --bad:#c62828; --badbg:#fdecea; --mid:#b26a00; --midbg:#fff3e0; --good:#1b5e20; --goodbg:#e8f5e9; }}
@media (prefers-color-scheme: dark) {{ :root {{ --fg:#e8e8ec; --bg:#131318; --muted:#99a; --line:#2c2c34; --card:#1c1c23;
  --badbg:#3a1715; --midbg:#332508; --goodbg:#12290f; --bad:#ff8a80; --mid:#ffcc80; --good:#a5d6a7; }} }}
:root[data-theme=light] {{ --fg:#1a1a1e; --bg:#fff; --muted:#667; --line:#e4e4ea; --card:#f7f7fa;
  --bad:#c62828; --badbg:#fdecea; --mid:#b26a00; --midbg:#fff3e0; --good:#1b5e20; --goodbg:#e8f5e9; }}
:root[data-theme=dark] {{ --fg:#e8e8ec; --bg:#131318; --muted:#99a; --line:#2c2c34; --card:#1c1c23;
  --badbg:#3a1715; --midbg:#332508; --goodbg:#12290f; --bad:#ff8a80; --mid:#ffcc80; --good:#a5d6a7; }}
body {{ font: 15px/1.65 -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  color: var(--fg); background: var(--bg); max-width: 1100px; margin: 0 auto; padding: 24px 20px 80px; }}
h1 {{ font-size: 26px; margin-bottom: 4px; }}
.sub {{ color: var(--muted); margin-bottom: 24px; }}
.note {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; margin: 16px 0; font-size: 14px; }}
.tblwrap {{ overflow-x: auto; margin: 16px 0; }}
table {{ border-collapse: collapse; font-size: 13px; white-space: nowrap; }}
th, td {{ border: 1px solid var(--line); padding: 4px 8px; text-align: center; }}
th {{ background: var(--card); }} th.dim {{ font-weight: 500; font-size: 11px; color: var(--muted); }}
td.case {{ text-align: left; font-weight: 600; }}
td.bad {{ background: var(--badbg); color: var(--bad); font-weight: 700; }}
td.mid {{ background: var(--midbg); color: var(--mid); }}
td.good {{ background: var(--goodbg); color: var(--good); }}
td.na {{ color: var(--muted); font-size: 11px; }}
.tag {{ display: inline-block; font-size: 11px; font-weight: 500; color: var(--muted);
  border: 1px solid var(--line); border-radius: 20px; padding: 0 8px; margin-left: 8px; vertical-align: 2px; }}
section {{ border-top: 1px solid var(--line); margin-top: 36px; padding-top: 16px; }}
.inst {{ margin: 4px 0; }} .inst.en {{ color: var(--muted); font-size: 13px; }}
.refs {{ display: flex; gap: 10px; margin: 10px 0; }}
.refs figure {{ margin: 0; text-align: center; }}
.refs img {{ height: 110px; border-radius: 8px; }}
.refs figcaption {{ font-size: 11px; color: var(--muted); }}
.gens {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 230px)); gap: 14px; }}
.gen {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
.gen h4 {{ margin: 0 0 8px; font-size: 14px; }}
.gen img {{ width: 100%; border-radius: 8px; }}
.fail {{ aspect-ratio: 3/4; display: flex; flex-direction: column; justify-content: center; align-items: center;
  color: var(--bad); background: var(--badbg); border-radius: 8px; text-align: center; padding: 12px; }}
.scores {{ font-size: 13px; margin: 8px 0 4px; }} .scores b.r {{ color: var(--bad); }}
.issues {{ font-size: 12.5px; color: var(--muted); margin: 4px 0; padding-left: 18px; }}
.notes {{ font-size: 12.5px; color: var(--muted); font-style: italic; margin: 4px 0 0; }}
</style>
<h1>MPIE-Bench Pilot case study</h1>
<p class='sub'>10 A complex multi-person interactive editing test pair (all C3 High contact density, including 3 Group threesome scene)× {len(models)} models · success {n_ok}/{len(runs)} open · VLM trial: {esc(judge)}(Proxy indicator, informal ArcFace/SMPL-X index)</p>
<div class='note'>⚠️ Score as VLM Proxy indicators are used to pilot verify "whether the indicators can widen the gap between models"; officially benchmark will be replaced with ArcFace/DWPose/SMPL-X mesh objective indicators. The reference picture is AI Generate a character portrait (not a real person).Count List: 5=The number of people is correct, 1=mistake.</div>
<h2>summary matrix</h2>
<div class='tblwrap'>{matrix}</div>
<h2>Model equalization by contact density</h2>
<div class='tblwrap'>{agg}</div>
{''.join(blocks)}
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"report -> {OUT} ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
