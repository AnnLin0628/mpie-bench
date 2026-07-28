#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""from _split.json Generate local annotation preview HTML(picture + prompt + checklist empty table).

exist pack You can click on the static service from the root directory:
  cd "$MPIE_TEST_PACK" && python -m http.server 8080
  Open http://127.0.0.1:8080/judgments/human_consistency/annot_preview/guide/index.html

usage:
  python build_annot_preview.py --pack "$MPIE_TEST_PACK"
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from pack_io import pack_root  # noqa: E402

INTER_Q = [
    ("I0", "Are the number of people participating in the interaction consistent with the number of people in the command? (Ignore passers-by)", "always"),
    ("I1", "Whether there is pathological mold penetration/Fusion? (Abnormal adhesion)", "always"),
    ("I2", "Is the physical contact required by the directive basically established?", "required"),
    ("I3", "Is there any inappropriate adhesion?/Tangled?", "forbidden"),
    ("I4", "(Supplementary) Does the contact part generally comply with the instructions?", "required And it is recommended I2=1 back"),
]
ANAT_Q = [
    ("A0", "Are the number of main characters consistent with the instructions?"),
    ("A1", "Is there any excess?/Suspended limbs or broken bodies?"),
    ("A2", "Are there any obvious missing limbs (should be visible in the posture)?"),
    ("A3", "Are there any limbs attached to the wrong person?"),
    ("A4", "Is the relative size of the two seriously outrageous?"),
    ("A5", "Is the (auxiliary) hand severely unrecognizable?"),
    ("A6", "(auxiliary) Is it obviously impossible to break?"),
]


def card(item: dict, idx: int, n: int) -> str:
    sid = html.escape(item["sample_id"])
    mid = html.escape(item["model_id"])
    intent = html.escape(str(item.get("intent") or ""))
    cat = html.escape(str(item.get("cat") or ""))
    prompt = html.escape(item.get("prompt") or "")
    # The page is at judgments/human_consistency/annot_preview/<split>/
    # priority media/; Otherwise use outputs/. right pack rooted http.server Using absolute paths is the most stable.
    rel = item.get("img_relpath") or ""
    if rel.startswith("judgments/") or rel.startswith("outputs/"):
        img = html.escape("/" + rel.lstrip("/"))
    else:
        img = html.escape("/" + rel.lstrip("/")) if rel else ""

    def radios(name: str, disabled_null: bool = False) -> str:
        if disabled_null:
            return f'<span class="null">null(Book intent not applicable)</span>'
        return (
            f'<label><input type="radio" name="{name}" value="1">1</label> '
            f'<label><input type="radio" name="{name}" value="0">0</label> '
            f'<label><input type="radio" name="{name}" value="U">U</label>'
        )

    rows = []
    for code, q, cond in INTER_Q:
        dis = (code == "I2" and intent != "required") or (
            code == "I3" and intent != "forbidden"
        )
        rows.append(
            f"<tr><td><b>{code}</b></td><td>{html.escape(q)}<div class='cond'>{html.escape(cond)}</div></td>"
            f"<td>{radios(f'{sid}__{mid}__{code}', dis)}</td></tr>"
        )
    for code, q in ANAT_Q:
        rows.append(
            f"<tr><td><b>{code}</b></td><td>{html.escape(q)}</td>"
            f"<td>{radios(f'{sid}__{mid}__{code}')}</td></tr>"
        )

    return f"""
<section class="card" id="item-{idx}">
  <header>
    <div class="meta">{idx+1}/{n} · <code>{sid}</code> · <code>{mid}</code></div>
    <div class="chips"><span>intent=<b>{intent}</b></span><span>cat={cat}</span></div>
  </header>
  <div class="body">
    <a href="{img}" target="_blank"><img src="{img}" alt="{sid}" loading="lazy"/></a>
    <div class="side">
      <h3>Edit prompt</h3>
      <pre>{prompt}</pre>
      <table>
        <thead><tr><th>ID</th><th>Question stem</th><th>answer</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <p class="hint">Please fill in for formal submission CSV（annot_templates/) or after exporting import_human_checklist.py;This page only previews exercises.</p>
    </div>
  </div>
</section>
"""


CSS = """
:root { --bg:#f6f3ee; --ink:#1c1917; --muted:#78716c; --line:#e7e5e4; --accent:#0f766e; }
* { box-sizing: border-box; }
body { margin:0; font:15px/1.45 "Source Sans 3", "IBM Plex Sans", sans-serif; background:var(--bg); color:var(--ink); }
header.page { padding:20px 24px; border-bottom:1px solid var(--line); background:#fff; position:sticky; top:0; z-index:2; }
header.page h1 { margin:0 0 6px; font-size:1.25rem; }
header.page p { margin:0; color:var(--muted); }
nav { display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }
nav a { color:var(--accent); text-decoration:none; font-size:13px; }
.card { margin:16px 24px 28px; background:#fff; border:1px solid var(--line); border-radius:10px; overflow:hidden; }
.card header { padding:12px 16px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.meta code { font-size:12px; }
.chips span { display:inline-block; margin-right:10px; color:var(--muted); font-size:13px; }
.body { display:grid; grid-template-columns: minmax(280px, 42%) 1fr; gap:0; }
.body img { width:100%; display:block; background:#111; max-height:80vh; object-fit:contain; }
.side { padding:14px 16px 20px; overflow:auto; }
pre { white-space:pre-wrap; background:#fafaf9; border:1px solid var(--line); padding:10px; border-radius:6px; font-size:13px; max-height:160px; overflow:auto; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th, td { border-top:1px solid var(--line); padding:8px 6px; vertical-align:top; }
th { text-align:left; color:var(--muted); font-weight:600; }
.cond { color:var(--muted); font-size:11px; margin-top:2px; }
.null { color:var(--muted); font-style:italic; }
.hint { color:var(--muted); font-size:12px; }
@media (max-width: 900px) { .body { grid-template-columns: 1fr; } }
"""


def build_split(hc: Path, name: str, items: list) -> Path:
    out_dir = hc / "annot_preview" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    cards = "\n".join(card(it, i, len(items)) for i, it in enumerate(items))
    nav = " · ".join(f'<a href="#item-{i}">{i+1}</a>' for i in range(len(items)))
    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MPIE Checklist · {html.escape(name)}</title>
<style>{CSS}</style>
</head>
<body>
<header class="page">
  <h1>Anat / Inter Checklist Preview · {html.escape(name)}（n={len(items)}）</h1>
  <p>coding:1=pass/Correct · 0=fail/Wrong · U=Can't tell. Forbidden to view mesh/VLM Fraction. See guide ../GUIDELINES.md</p>
  <nav>{nav}</nav>
</header>
{cards}
</body>
</html>
"""
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="")
    args = ap.parse_args()
    root = pack_root(args.pack or None)
    hc = root / "judgments" / "human_consistency"
    split_path = hc / "_split.json"
    data = json.loads(split_path.read_text(encoding="utf-8"))
    wrote = {}
    for name, items in (data.get("splits") or {}).items():
        if not items:
            continue
        p = build_split(hc, name, items)
        wrote[name] = str(p)
    index = hc / "annot_preview" / "index.html"
    links = "".join(
        f'<li><a href="{n}/index.html">{n}</a> ({len(data["splits"].get(n) or [])})</li>'
        for n in ("guide", "pilot", "holdout", "main")
        if data.get("splits", {}).get(n)
    )
    index.write_text(
        f"""<!DOCTYPE html><html><head><meta charset="utf-8"/><title>Annot preview</title>
<style>body{{font:16px/1.5 sans-serif;margin:40px;background:#f6f3ee}} a{{color:#0f766e}}</style>
</head><body>
<h1>Human consistency annot preview</h1>
<p>Serve pack root, e.g. <code>python -m http.server 8080</code> then open these links.</p>
<ul>{links}</ul>
<p><a href="../GUIDELINES.md">GUIDELINES.md</a> · CSV templates in <code>../annot_templates/</code></p>
</body></html>
""",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": wrote, "index": str(index)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
