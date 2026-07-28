#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare penalty Anat/Inter vs additive (gate) rescoring on existing mesh_v3 JSONs.

Important: continuous add  Σ w·(1−P)  is affine-identical to  1−Σ w·P.
Only thresholded / binary credit can change discrimination.

Writes: $MPIE_ROOT/data/eval_outputs/full2500_additive_v1/
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from aggregate_vlm_judge_v1 import order_models_closed_then_open


def _f(j: dict, *keys: str) -> Optional[float]:
    for k in keys:
        v = j.get(k)
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x and math.isfinite(x):
            return x
    return None


def score_anat_variants(j: dict, tau: float) -> Dict[str, float]:
    """Return penalty (stored), add_cont, add_gate, add_binary for Anat."""
    p_extra = _f(j, "P_anat_extra") or 0.0
    p_resid = _f(j, "P_anat_resid") or 0.0
    p_struct = _f(j, "P_anat_struct") or 0.0
    p_detect = _f(j, "P_anat_detect") or 0.0
    we, wr, ws, wd = 0.40, 0.30, 0.15, 0.15

    penalty = _f(j, "S_anat_mesh")
    if penalty is None:
        penalty = float(np.clip(1.0 - we * p_extra - wr * p_resid - ws * p_struct - wd * p_detect, 0, 1))

    add_cont = float(np.clip(
        we * (1.0 - p_extra) + wr * (1.0 - p_resid) + ws * (1.0 - p_struct) + wd * (1.0 - p_detect),
        0, 1,
    ))

    credits = [
        (we, p_extra < tau),
        (wr, p_resid < tau),
        (ws, p_struct < tau),
        (wd, p_detect < tau),
    ]
    add_gate = float(sum(w for w, ok in credits if ok))

    n_pass = sum(1 for _, ok in credits if ok)
    add_binary = n_pass / 4.0

    return {
        "penalty": float(penalty),
        "add_cont": add_cont,
        "add_gate": add_gate,
        "add_binary": add_binary,
    }


def score_inter_variants(j: dict, tau: float) -> Dict[str, float]:
    p_fuse = _f(j, "P_fuse") or 0.0
    p_miss = _f(j, "P_miss") or 0.0
    p_unwanted = _f(j, "P_unwanted") or 0.0
    intent = (j.get("contact_intent") or "unspecified").strip()
    under = bool(j.get("under_detect"))
    w_fuse, w_miss, w_unw = 0.55, 0.45, 0.45

    penalty = _f(j, "S_inter_mesh")
    if penalty is None:
        if intent == "required":
            penalty = max(0.0, 1.0 - w_fuse * p_fuse - w_miss * p_miss)
        elif intent == "forbidden":
            penalty = max(0.0, 1.0 - w_fuse * p_fuse - w_unw * p_unwanted)
        else:
            penalty = max(0.0, 1.0 - p_fuse)
        if under:
            penalty *= 0.5

    # continuous add (equivalent to penalty when no under_detect; under handled same)
    if intent == "required":
        add_cont = w_fuse * (1.0 - p_fuse) + w_miss * (1.0 - p_miss)
        gates = [(w_fuse, p_fuse < tau), (w_miss, p_miss < tau)]
    elif intent == "forbidden":
        add_cont = w_fuse * (1.0 - p_fuse) + w_unw * (1.0 - p_unwanted)
        gates = [(w_fuse, p_fuse < tau), (w_unw, p_unwanted < tau)]
    else:
        add_cont = 1.0 - p_fuse
        gates = [(1.0, p_fuse < tau)]

    add_gate = float(sum(w for w, ok in gates if ok))
    add_binary = float(sum(1 for _, ok in gates if ok) / max(1, len(gates)))

    if under:
        add_cont *= 0.5
        add_gate *= 0.5
        add_binary *= 0.5

    return {
        "penalty": float(penalty),
        "add_cont": float(np.clip(add_cont, 0, 1)),
        "add_gate": float(np.clip(add_gate, 0, 1)),
        "add_binary": float(np.clip(add_binary, 0, 1)),
    }


def disc_stats(means: List[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([x for x in means if x is not None], dtype=np.float64)
    if arr.size == 0:
        return {"n_models": 0, "mean": None, "std": None, "range": None, "pairwise_mad": None, "cv": None}
    std = float(np.std(arr, ddof=0))
    rng = float(arr.max() - arr.min())
    if arr.size >= 2:
        diffs = [abs(float(a - b)) for i, a in enumerate(arr) for b in arr[i + 1 :]]
        mad = float(np.mean(diffs))
    else:
        mad = 0.0
    m = float(arr.mean())
    cv = float(std / m) if abs(m) > 1e-9 else None
    return {
        "n_models": int(arr.size),
        "mean": m,
        "std": std,
        "range": rng,
        "pairwise_mad": mad,
        "cv": cv,
    }


def load_model_scores(pack: Path, models: List[str], tau: float) -> Dict[str, dict]:
    root = pack / "judgments" / "mesh_v3"
    out: Dict[str, dict] = {}
    for mid in models:
        d = root / mid
        if not d.is_dir():
            continue
        buckets = {
            "anat": {k: [] for k in ("penalty", "add_cont", "add_gate", "add_binary")},
            "inter": {k: [] for k in ("penalty", "add_cont", "add_gate", "add_binary")},
        }
        n_ok = 0
        for p in d.glob("*.json"):
            if p.name.startswith("_"):
                continue
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if j.get("ok") is False:
                continue
            n_ok += 1
            a = score_anat_variants(j, tau)
            i = score_inter_variants(j, tau)
            for k, v in a.items():
                buckets["anat"][k].append(v)
            for k, v in i.items():
                buckets["inter"][k].append(v)

        def _mean(xs: List[float]) -> Optional[float]:
            return float(np.mean(xs)) if xs else None

        def _std(xs: List[float]) -> Optional[float]:
            return float(np.std(xs, ddof=0)) if xs else None

        out[mid] = {
            "n_ok": n_ok,
            "anat_mean": {k: _mean(v) for k, v in buckets["anat"].items()},
            "inter_mean": {k: _mean(v) for k, v in buckets["inter"].items()},
            "anat_std": {k: _std(v) for k, v in buckets["anat"].items()},
            "inter_std": {k: _std(v) for k, v in buckets["inter"].items()},
        }
    return out


def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def bar_chart(title: str, pairs: List[Tuple[str, Optional[float]]], color: str) -> str:
    vals = [(n, v) for n, v in pairs if v is not None]
    if not vals:
        return ""
    scale = 1.0
    rows = []
    for name, v in vals:
        pct = max(0.0, min(100.0, 100.0 * float(v) / scale))
        rows.append(
            f'<div class="bar-row"><span class="bn">{name}</span>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;background:{color}"></div></div>'
            f'<span class="bv">{fmt(v)}</span></div>'
        )
    return f'<div class="chart"><h3>{title}</h3>{"".join(rows)}</div>'


def render_html(summary: dict) -> str:
    models = summary["ranking"]
    variants = ["penalty", "add_cont", "add_gate", "add_binary"]
    vlabels = {
        "penalty": "Point deduction system (current)1−ΣwP",
        "add_cont": "Continuous bonus points Σw(1−P) ≈ equivalence",
        "add_gate": f"Threshold bonus points (P&lt;{summary['tau']:.2f} Just score)",
        "add_binary": "Equally weighted binary bonus points (calculated if the threshold is exceeded) 1/k）",
    }

    # main comparison table: model × variant for Anat and Inter
    def table_axis(axis: str) -> str:
        key = f"{axis}_mean"
        head = "".join(f"<th>{vlabels[v]}</th>" for v in variants)
        body = []
        for mid in models:
            m = summary["by_model"][mid]
            cells = "".join(f'<td class="num">{fmt(m[key].get(v))}</td>' for v in variants)
            body.append(f'<tr><td class="model">{mid}</td>{cells}</tr>')
        return f"""<h2>{axis.upper()} · model mean</h2>
<table><thead><tr><th>Model</th>{head}</tr></thead>
<tbody>{''.join(body)}</tbody></table>"""

    # discrimination table
    disc_rows = []
    for axis in ("anat", "inter"):
        for v in variants:
            st = summary["discrimination"][axis][v]
            disc_rows.append(
                f"<tr><td>{axis}</td><td>{vlabels[v]}</td>"
                f'<td class="num">{fmt(st.get("mean"))}</td>'
                f'<td class="num">{fmt(st.get("std"))}</td>'
                f'<td class="num">{fmt(st.get("range"))}</td>'
                f'<td class="num">{fmt(st.get("pairwise_mad"))}</td>'
                f'<td class="num">{fmt(st.get("cv"))}</td></tr>'
            )

    charts = []
    for axis, color in (("anat", "#ef4444"), ("inter", "#f59e0b")):
        for v in variants:
            pairs = [(mid, summary["by_model"][mid][f"{axis}_mean"].get(v)) for mid in models]
            charts.append(bar_chart(f"{axis.upper()} · {vlabels[v]}", pairs, color))

    # verdict
    verdict_bits = []
    for axis in ("anat", "inter"):
        best = max(
            variants,
            key=lambda v: (summary["discrimination"][axis][v].get("std") or 0.0),
        )
        pen_std = summary["discrimination"][axis]["penalty"].get("std") or 0.0
        best_std = summary["discrimination"][axis][best].get("std") or 0.0
        ratio = (best_std / pen_std) if pen_std > 1e-9 else None
        verdict_bits.append(
            f"<li><b>{axis.upper()}</b>:Model room std The biggest is <b>{vlabels[best]}</b>"
            f"（std={fmt(best_std)}, relative penalty points system ×{fmt(ratio) if ratio is not None else '—'}）。"
            f"The continuous points plus and minus points system should be almost the same.</li>"
        )

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MPIE Extra points system vs Point deduction system · Anat/Inter</title>
<style>
:root {{ --bg:#f7f8fa; --card:#fff; --line:#e8eaee; --text:#1f2430; --muted:#6b7280; }}
* {{ box-sizing:border-box }}
body {{ margin:0; padding:28px 24px 48px; font-family:-apple-system,"PingFang SC",Helvetica,sans-serif;
  background:var(--bg); color:var(--text); }}
h1 {{ font-size:22px; margin:0 0 6px; }}
h2 {{ font-size:15px; margin:28px 0 10px; }}
.sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; max-width:920px; line-height:1.55; }}
.box {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px;
  max-width:960px; margin-bottom:18px; font-size:13px; line-height:1.55; color:#374151; }}
.box code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12px; }}
table {{ width:100%; max-width:1100px; border-collapse:collapse; background:var(--card);
  border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:13px; margin-bottom:8px; }}
th, td {{ padding:9px 10px; border-bottom:1px solid #f0f2f5; }}
th {{ background:#fafbfc; color:var(--muted); font-size:11px; text-align:right; }}
th:first-child, td.model {{ text-align:left; font-weight:600; }}
td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
.charts {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:14px; max-width:1100px; }}
.chart {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.chart h3 {{ margin:0 0 10px; font-size:12px; }}
.bar-row {{ display:grid; grid-template-columns:140px 1fr 48px; gap:8px; align-items:center; margin:5px 0; }}
.bn {{ font-size:11px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.track {{ height:10px; background:#eef0f3; border-radius:5px; overflow:hidden; }}
.fill {{ height:100%; border-radius:5px; }}
.bv {{ font-size:11px; color:var(--muted); text-align:right; }}
.note {{ color:var(--muted); font-size:12px; margin-top:14px; max-width:960px; }}
ul.verdict {{ margin:8px 0 0; padding-left:1.2em; }}
</style></head><body>
<h1>Extra points system vs Point deduction system · Anat / Inter(Full quantity already available mesh data)</h1>
<p class="sub">
  N Samples by model <code>ok</code> bar; threshold τ={summary['tau']:.2f} · {summary['generated_at']} ·
  <a href="/">← Review home page</a> ·
  <a href="/eval_outputs/full2500_v3/">Compare: current plan③</a>
</p>

<div class="box">
  <p><b>Let’s make the core conclusion clear first</b>: If you just put "from 1 "withhold punishment" was rewritten as "from 0 add (1−P)」，
  official <code>Σ w·(1−P) = 1 − Σ w·P</code>, and the current deduction system<strong>The values ​​are completely equivalent</strong>, the distinction will not change.
  What really makes a difference is "points will be awarded only if you do one item correctly" - the following<strong>Threshold bonus points / Equally weighted binary</strong>。</p>
  <ul class="verdict">{''.join(verdict_bits)}</ul>
  <p style="margin-top:10px;margin-bottom:0">Discrimination index look at<strong>model room</strong> std / range / The average absolute difference of each pair (the larger, the more separated they are).</p>
</div>

<div class="box">
  <b>four recipes</b>
  <ul>
    <li><b>Point deduction system</b>：placement <code>S_*_mesh</code>（Anat=1−0.40P_extra−…；Inter according to intent buckle P_fuse/P_miss/P_unwanted）</li>
    <li><b>Continuous bonus points</b>：Σ w(1−P), should be almost the same as the deduction points (for verification)</li>
    <li><b>Threshold bonus points</b>:from 0 from; only if the item P&lt;τ Only then add the weight w</li>
    <li><b>Equally weighted binary</b>: The proportion of applicable items that pass the threshold (Inter according to intent Determine the number of items)</li>
  </ul>
</div>

<h2>Summary table of discrimination (larger is usually better)</h2>
<table>
<thead><tr><th>axis</th><th>formula</th><th>mean of model mean</th><th>model room std</th><th>range</th><th>Pairs|Δ|mean</th><th>CV</th></tr></thead>
<tbody>
{''.join(disc_rows)}
</tbody>
</table>

{table_axis("anat")}
{table_axis("inter")}

<h2>column contrast</h2>
<div class="charts">
{''.join(charts)}
</div>

<p class="note">
  Data source:<code>{summary['pack']}/judgments/mesh_v3/</code>, offline recalculation, no rerun Multi-HMR。
  Model:{', '.join(models)}. Extra points if threshold std It is significantly higher than the deduction system, indicating that "points are given only if you do it right" is more distinguishable;
  If they are almost the same, then changing the bonus point system itself will not help, and the signal still needs to be changed (leftover / intent wait).
</p>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--out", default="")
    ap.add_argument("--tau", type=float, default=0.35, help="gate: credit only if P < tau")
    ap.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="default: auto from mesh_v3 dirs",
    )
    args = ap.parse_args()
    pack = Path(args.pack).expanduser().resolve()
    out = (
        Path(args.out).expanduser()
        if args.out
        else Path("data") / "eval_outputs" / "full2500_additive_v1"
    )
    out.mkdir(parents=True, exist_ok=True)

    root = pack / "judgments" / "mesh_v3"
    if args.models:
        models = list(args.models)
    else:
        models = sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
        )
    models = order_models_closed_then_open(models)

    by_model = load_model_scores(pack, models, args.tau)
    models = [m for m in models if m in by_model]
    ranking = order_models_closed_then_open(models)

    discrimination: Dict[str, Dict[str, dict]] = {"anat": {}, "inter": {}}
    for axis in ("anat", "inter"):
        for v in ("penalty", "add_cont", "add_gate", "add_binary"):
            means = [by_model[m][f"{axis}_mean"].get(v) for m in ranking]
            discrimination[axis][v] = disc_stats(means)

    summary = {
        "protocol": "additive_vs_penalty_v1",
        "pack": str(pack),
        "tau": args.tau,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": models,
        "ranking": ranking,
        "by_model": by_model,
        "discrimination": discrimination,
        "note": (
            "add_cont ≡ penalty algebraically; add_gate/add_binary are thresholded credit."
        ),
    }

    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "index.html").write_text(render_html(summary), encoding="utf-8")

    # console verdict
    print(json.dumps({
        "out": str(out),
        "tau": args.tau,
        "ranking": ranking,
        "disc_anat_std": {v: discrimination["anat"][v].get("std") for v in discrimination["anat"]},
        "disc_inter_std": {v: discrimination["inter"][v].get("std") for v in discrimination["inter"]},
        "means_anat_gate": {m: by_model[m]["anat_mean"]["add_gate"] for m in ranking},
        "means_inter_gate": {m: by_model[m]["inter_mean"]["add_gate"] for m in ranking},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
