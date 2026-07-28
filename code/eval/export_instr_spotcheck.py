#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export a stratified human spot-check pack for Instr QA v2.

Default: 80 samples from smoke100 frozen bank → HTML + JSONL for annotators.

Usage:
  python export_instr_spotcheck.py --pack "$MPIE_TEST_PACK" --n 80
  # writes $PACK/instr_qa_v2/_spotcheck_v2.1/ and mirrors to the evaluation dashboard eval_outputs
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from instr_qa_common import REVISION
from pack_io import pack_root


def load_bank(pack: Path) -> list[dict]:
    d = pack / "instr_qa_v2"
    rows = []
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = j.get("_meta") or {}
        rev = j.get("revision") or meta.get("revision")
        if rev and rev != REVISION:
            continue
        qs = j.get("questions") or []
        if len(qs) < 2:
            continue
        cat = (meta.get("cat") or j.get("sample_id", "").split("__")[0] or "other")
        rows.append({**j, "_cat": cat})
    return rows


def stratified(rows: list[dict], n: int, seed: int) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["_cat"]].append(r)
    rng = random.Random(seed)
    for cat in by:
        rng.shuffle(by[cat])
    cats = sorted(by.keys())
    if not cats:
        return []
    # round-robin until n
    picked = []
    idx = {c: 0 for c in cats}
    while len(picked) < n and any(idx[c] < len(by[c]) for c in cats):
        for c in cats:
            if len(picked) >= n:
                break
            i = idx[c]
            if i < len(by[c]):
                picked.append(by[c][i])
                idx[c] = i + 1
    return picked


def render_html(samples: list[dict], *, pack: Path, n: int) -> str:
    blocks = []
    for i, s in enumerate(samples, 1):
        qs = s.get("questions") or []
        q_html = "".join(
            f'<li><span class="tag">{q.get("subtype") or q.get("bucket")}</span> '
            f'{_esc(q.get("q") or "")}'
            f'{" · <i>swap</i>" if q.get("swap_sensitive") else ""}</li>'
            for q in qs
        )
        blocks.append(
            f"""
<section class="card" id="s{i}">
  <h3>{i}. <code>{_esc(s.get("sample_id") or "")}</code>
    <span class="cat">{_esc(s.get("_cat") or "")}</span></h3>
  <p class="instr">{_esc(s.get("instruction") or "")}</p>
  <ol class="qs">{q_html}</ol>
  <div class="vote">
    Question bank quality:
    <label><input type="radio" name="q{i}" value="ok"> OK</label>
    <label><input type="radio" name="q{i}" value="weak"> Easy/empty</label>
    <label><input type="radio" name="q{i}" value="bad"> Bad question</label>
    · asymm Is swapping really wrong:
    <label><input type="radio" name="a{i}" value="yes"> meeting</label>
    <label><input type="radio" name="a{i}" value="no"> Won't</label>
    <label><input type="radio" name="a{i}" value="unsure"> uncertain</label>
  </div>
</section>"""
        )
    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Instr QA spotcheck · {REVISION}</title>
<style>
body{{font-family:-apple-system,"PingFang SC",sans-serif;margin:0;padding:24px;background:#f6f7f9;color:#1f2430}}
h1{{font-size:20px;margin:0 0 8px}}
.meta{{color:#6b7280;font-size:13px;margin-bottom:18px;max-width:900px;line-height:1.5}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px 16px;margin:0 0 12px;max-width:960px}}
h3{{margin:0 0 8px;font-size:14px}}
.cat{{display:inline-block;margin-left:8px;padding:1px 8px;background:#eef2ff;color:#3730a3;border-radius:999px;font-size:12px;font-weight:500}}
.instr{{font-size:13px;line-height:1.45;color:#374151;background:#f9fafb;border-radius:8px;padding:10px 12px}}
.qs{{margin:10px 0;padding-left:1.2em;font-size:13px;line-height:1.45}}
.tag{{display:inline-block;min-width:72px;font-size:11px;color:#047857;font-weight:600}}
.vote{{font-size:13px;color:#374151;border-top:1px solid #f3f4f6;padding-top:10px;margin-top:8px}}
.vote label{{margin-right:10px}}
code{{font-size:12px}}
</style></head><body>
<h1>Instr QA Manual inspection · {REVISION}</h1>
<p class="meta">Pack: <code>{pack}</code><br>
N = {n}(stratified sampling)· generated in {datetime.now().isoformat(timespec="seconds")}<br>
Please judge: Does the question test asymmetric responsibilities?/role binding; exchange R1↔R2 Should something go wrong. Results may be recorded in tables or summarized verbally.</p>
{"".join(blocks)}
</body></html>"""


def _esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="", help="output dir")
    args = ap.parse_args()
    pack = pack_root(args.pack or None)
    rows = load_bank(pack)
    if not rows:
        raise SystemExit(f"no frozen bank under {pack}/instr_qa_v2")
    picked = stratified(rows, args.n, args.seed)
    out = Path(args.out) if args.out else (pack / "instr_qa_v2" / f"_spotcheck_{REVISION}")
    out.mkdir(parents=True, exist_ok=True)

    # JSONL for programmatic use
    jsonl = out / "spotcheck.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for s in picked:
            rec = {
                "sample_id": s.get("sample_id"),
                "cat": s.get("_cat"),
                "instruction": s.get("instruction"),
                "questions": s.get("questions"),
                "revision": REVISION,
                "n_person": s.get("n_person"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    html = render_html(picked, pack=pack, n=len(picked))
    (out / "index.html").write_text(html, encoding="utf-8")
    meta = {
        "revision": REVISION,
        "pack": str(pack),
        "n": len(picked),
        "seed": args.seed,
        "cats": sorted({s["_cat"] for s in picked}),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # mirror to the evaluation dashboard
    pub = (
        Path("data") / "eval_outputs"
        / "instr_spotcheck_v2.1"
    )
    pub.mkdir(parents=True, exist_ok=True)
    (pub / "index.html").write_text(html, encoding="utf-8")
    (pub / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (pub / "spotcheck.jsonl").write_text(jsonl.read_text(encoding="utf-8"), encoding="utf-8")

    print(json.dumps({"out": str(out), "public": str(pub), **meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
