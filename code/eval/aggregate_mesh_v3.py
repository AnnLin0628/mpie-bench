#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate protocol-v3 mesh scores into a leaderboard HTML/JSON report.

Front = Count/ID/Anat/Inter/Instr/Qual (no Overall).
Anat/Inter → Multi-HMR mesh when available (else keep VLM and mark source).
Instr → weighted instr_v2 total; sub-scores in supplement.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_eval_v2 import (  # noqa: E402
    aggregate_v2,
    fmt,
)
from aggregate_vlm_judge_v1 import (  # noqa: E402
    MODEL_DISPLAY_ORDER,
    order_models_closed_then_open,
)

# Level 4 contact density: original C3(Near-fitting load-bearing) and C4(fighting and confrontation) merged into C3 high contact gear
CONTACT_LEVELS = ("C0", "C1", "C2", "C3")
CONTACT_LABELS = {
    "C0": "C0 No contact",
    "C1": "C1 light / instantaneous",
    "C2": "C2 point / line",
    "C3": "C3 High contact (close fit / fight)",
}
# Five levels of history CSV If it still contains C4, merged when reading C3
_CONTACT_LEGACY_MERGE = {"C4": "C3"}
DEFAULT_TAGGED_CSV = (
    Path(".")
    / "data"
    / "manifests"
    / "prompt_distribution"
    / "targets_tagged.csv"
)


# Embedded fallback mesh summaries — used only when pack mesh is missing / n mismatch
FALLBACK_MESH: Dict[str, dict] = {
    "_gt": {
        "S_inter_mesh": 0.7995809495939196,
        "S_anat_mesh": 0.8040800776274287,
        "P_anat_extra_mean": 0.25108707576920675,
        "anat_orphan_frac_mean": 0.01566651473367043,
        "anat_leftover_frac_mean": 0.6524722844053122,
        "n": 100,
    },
    "flux1-kontext-dev": {
        "S_inter_mesh": 0.855543158411161,
        "S_anat_mesh": 0.8651805637342339,
        "P_anat_extra_mean": 0.25066436100897455,
        "anat_orphan_frac_mean": 0.020434572096221765,
        "anat_leftover_frac_mean": 0.6448092450558058,
        "n": 100,
    },
    "gemini-3-pro-image": {
        "S_inter_mesh": 0.8380388022505721,
        "S_anat_mesh": 0.7880588634060837,
        "P_anat_extra_mean": 0.2404983648245687,
        "anat_orphan_frac_mean": 0.018647980559718443,
        "anat_leftover_frac_mean": 0.5933340335943691,
        "n": 100,
    },
    "gpt-image-2": {
        "S_inter_mesh": 0.7640773508542268,
        "S_anat_mesh": 0.744491009051597,
        "P_anat_extra_mean": 0.2699342457792116,
        "anat_orphan_frac_mean": 0.021874244560465505,
        "anat_leftover_frac_mean": 0.658826497589755,
        "n": 100,
    },
    "seedream-5-pro": {
        "S_inter_mesh": 0.8433466254300976,
        "S_anat_mesh": 0.7810046345102089,
        "P_anat_extra_mean": 0.30182416710139165,
        "anat_orphan_frac_mean": 0.018324737994357312,
        "anat_leftover_frac_mean": 0.6797388677378199,
        "n": 100,
    },
}


def _mean_floats(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _normalize_contact_c(cc: str) -> Optional[str]:
    c = (cc or "").strip().upper()
    c = _CONTACT_LEGACY_MERGE.get(c, c)
    return c if c in CONTACT_LEVELS else None


def load_contact_c_map(tagged_csv: Optional[Path] = None) -> Dict[str, str]:
    """sample_id → C0..C3 from prompt_distribution/targets_tagged.csv."""
    path = Path(tagged_csv) if tagged_csv else DEFAULT_TAGGED_CSV
    out: Dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = f"{row['board_cat']}__{row['anchor']}__{row['target_id']}"
            cc = _normalize_contact_c(row.get("contact_c") or "")
            if cc:
                out[sid] = cc
    return out


def aggregate_mesh_by_density(
    pack: Path,
    model_ids: List[str],
    *,
    tagged_csv: Optional[Path] = None,
) -> Dict[str, dict]:
    """Per model × C0–C3 mean Anat/Inter from mesh_v3 judgments."""
    sid2c = load_contact_c_map(tagged_csv)
    root = pack / "judgments" / "mesh_v3"
    out: Dict[str, dict] = {}
    for mid in model_ids:
        d = root / mid
        buckets = {c: {"anat": [], "inter": []} for c in CONTACT_LEVELS}
        n_ok = n_miss_c = 0
        if d.is_dir():
            for p in d.glob("*.json"):
                if p.name.startswith("_"):
                    continue
                try:
                    j = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if j.get("ok") is False:
                    continue
                sid = j.get("sample_id") or p.stem
                cc = sid2c.get(sid)
                if cc is None:
                    n_miss_c += 1
                    continue
                n_ok += 1
                if j.get("S_anat_mesh") is not None:
                    buckets[cc]["anat"].append(float(j["S_anat_mesh"]))
                if j.get("S_inter_mesh") is not None:
                    buckets[cc]["inter"].append(float(j["S_inter_mesh"]))
        by_c = {}
        for c in CONTACT_LEVELS:
            by_c[c] = {
                "Anat": _mean_floats(buckets[c]["anat"]),
                "Inter": _mean_floats(buckets[c]["inter"]),
                "n_anat": len(buckets[c]["anat"]),
                "n_inter": len(buckets[c]["inter"]),
            }
        # slope: C3 − C0 (negative = degrades with density)
        def _delta(key: str) -> Optional[float]:
            a0, a3 = by_c["C0"].get(key), by_c["C3"].get(key)
            if a0 is None or a3 is None:
                return None
            return float(a3) - float(a0)

        dlt_a, dlt_i = _delta("Anat"), _delta("Inter")
        out[mid] = {
            "by_c": by_c,
            "n_ok_tagged": n_ok,
            "n_miss_tag": n_miss_c,
            "delta_anat_c3_c0": dlt_a,
            "delta_inter_c3_c0": dlt_i,
            "delta_anat_c4_c0": dlt_a,  # legacy alias
            "delta_inter_c4_c0": dlt_i,
        }
    return out


def summarize_mesh_dir(model_dir: Path) -> Optional[dict]:
    """Aggregate per-sample mesh_v3 judgments. Returns None if no usable scores."""
    if not model_dir.is_dir():
        return None
    anat, inter, p_extra, orphan, leftover = [], [], [], [], []
    for p in model_dir.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j.get("ok") is False:
            continue
        if j.get("S_anat_mesh") is not None:
            anat.append(float(j["S_anat_mesh"]))
        if j.get("S_inter_mesh") is not None:
            inter.append(float(j["S_inter_mesh"]))
        if j.get("P_anat_extra") is not None:
            p_extra.append(float(j["P_anat_extra"]))
        if j.get("anat_orphan_frac") is not None:
            orphan.append(float(j["anat_orphan_frac"]))
        if j.get("anat_leftover_frac") is not None:
            leftover.append(float(j["anat_leftover_frac"]))
    n = max(len(anat), len(inter))
    if n == 0:
        return None
    return {
        "n": n,
        "S_anat_mesh": _mean_floats(anat),
        "S_inter_mesh": _mean_floats(inter),
        "P_anat_extra_mean": _mean_floats(p_extra),
        "anat_orphan_frac_mean": _mean_floats(orphan),
        "anat_leftover_frac_mean": _mean_floats(leftover),
        "source": "recomputed_from_per_sample",
    }


def _render_density_section(summary: dict) -> str:
    dens = summary.get("by_density") or {}
    if not dens:
        return ""
    # pack-level n per C (from first model that has counts, or recompute max)
    n_c = {c: 0 for c in CONTACT_LEVELS}
    for mid, blk in dens.items():
        for c in CONTACT_LEVELS:
            n_c[c] = max(n_c[c], int((blk.get("by_c") or {}).get(c, {}).get("n_anat") or 0))

    # two-row header
    top = "".join(
        f'<th colspan="2" class="cband">{CONTACT_LABELS[c]}'
        f'<span class="th-hint">n≈{n_c[c]}</span></th>'
        for c in CONTACT_LEVELS
    )
    sub = "".join(
        '<th class="subh">Anat</th><th class="subh">Inter</th>' for _ in CONTACT_LEVELS
    )
    body = []
    for mid in summary["ranking"]:
        blk = dens.get(mid) or {}
        by_c = blk.get("by_c") or {}
        # skip empty models (no mesh at all)
        if not any((by_c.get(c) or {}).get("Anat") is not None for c in CONTACT_LEVELS):
            if not any((by_c.get(c) or {}).get("Inter") is not None for c in CONTACT_LEVELS):
                continue
        cells = []
        for c in CONTACT_LEVELS:
            cell = by_c.get(c) or {}
            na, ni = cell.get("n_anat") or 0, cell.get("n_inter") or 0
            tip = f'title="n_anat={na} n_inter={ni}"'
            cells.append(f'<td class="num" {tip}>{fmt(cell.get("Anat"))}</td>')
            cells.append(f'<td class="num" {tip}>{fmt(cell.get("Inter"))}</td>')
        d_a = blk.get("delta_anat_c3_c0", blk.get("delta_anat_c4_c0"))
        d_i = blk.get("delta_inter_c3_c0", blk.get("delta_inter_c4_c0"))
        body.append(
            f'<tr><td class="model">{mid}</td>{"".join(cells)}'
            f'<td class="num delta">{fmt(d_a)}</td>'
            f'<td class="num delta">{fmt(d_i)}</td></tr>'
        )
    if not body:
        return ""

    # multi-model curves: one SVG per axis (Anat / Inter), legend by model
    palette = [
        "#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed",
        "#0891b2", "#be185d", "#4d7c0f", "#c2410c", "#475569",
    ]
    n_lvl = max(1, len(CONTACT_LEVELS) - 1)
    series_mids: List[str] = []
    for mid in summary["ranking"]:
        blk = dens.get(mid) or {}
        by_c = blk.get("by_c") or {}
        if any((by_c.get(c) or {}).get("Anat") is not None for c in CONTACT_LEVELS) or any(
            (by_c.get(c) or {}).get("Inter") is not None for c in CONTACT_LEVELS
        ):
            series_mids.append(mid)

    def _multi_curve(axis_key: str, title: str) -> str:
        series = []
        all_ys: List[float] = []
        for mid in series_mids:
            by_c = (dens.get(mid) or {}).get("by_c") or {}
            vals = [(by_c.get(c) or {}).get(axis_key) for c in CONTACT_LEVELS]
            if all(v is None for v in vals):
                continue
            series.append((mid, vals))
            all_ys.extend(float(v) for v in vals if v is not None)
        if len(series) < 1 or len(all_ys) < 2:
            return ""
        w, h = 560, 280
        pad_l, pad_r, pad_t, pad_b = 48, 16, 18, 36
        ymin, ymax = min(all_ys), max(all_ys)
        pad_y = max(0.02, (ymax - ymin) * 0.12)
        ymin, ymax = max(0.0, ymin - pad_y), min(1.0, ymax + pad_y)
        if abs(ymax - ymin) < 1e-6:
            ymin, ymax = ymin - 0.05, ymax + 0.05

        def xy(i: int, v: float) -> Tuple[float, float]:
            x = pad_l + (w - pad_l - pad_r) * (i / float(n_lvl))
            y = pad_t + (h - pad_t - pad_b) * (1.0 - (v - ymin) / (ymax - ymin))
            return x, y

        # grid + axes
        grid = []
        for t in range(5):
            gv = ymin + (ymax - ymin) * (t / 4.0)
            _, gy = xy(0, gv)
            grid.append(
                f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w-pad_r}" y2="{gy:.1f}" '
                f'stroke="#eef0f3" stroke-width="1"/>'
                f'<text x="{pad_l-6}" y="{gy+3:.1f}" text-anchor="end" '
                f'font-size="10" fill="#9ca3af">{gv:.2f}</text>'
            )
        xlabels = "".join(
            f'<text x="{xy(i, ymin)[0]:.1f}" y="{h-12}" text-anchor="middle" '
            f'font-size="11" fill="#6b7280">{c}</text>'
            for i, c in enumerate(CONTACT_LEVELS)
        )
        paths = []
        legend = []
        for si, (mid, vals) in enumerate(series):
            color = palette[si % len(palette)]
            pts = [(i, float(v)) for i, v in enumerate(vals) if v is not None]
            if len(pts) < 2:
                continue
            poly = " ".join(f"{xy(i, v)[0]:.1f},{xy(i, v)[1]:.1f}" for i, v in pts)
            dots = "".join(
                f'<circle cx="{xy(i, v)[0]:.1f}" cy="{xy(i, v)[1]:.1f}" '
                f'r="3" fill="{color}" stroke="#fff" stroke-width="1"/>'
                for i, v in pts
            )
            paths.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
                f'points="{poly}"/>{dots}'
            )
            legend.append(
                f'<span class="leg"><i style="background:{color}"></i>{mid}</span>'
            )
        svg = (
            f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'role="img" aria-label="{title}">'
            f"{''.join(grid)}{xlabels}{''.join(paths)}</svg>"
        )
        return (
            f'<div class="chart dens-multi">'
            f"<h3>{title}</h3>"
            f'<div class="dens-legend">{"".join(legend)}</div>'
            f"{svg}</div>"
        )

    charts = [
        _multi_curve("Anat", "Anat · C0→C3(Multiple models)"),
        _multi_curve("Inter", "Inter · C0→C3(Multiple models)"),
    ]
    charts = [c for c in charts if c]

    analysis = """
<div class="howto dens-analysis">
  <h4>Contact density binning instructions (C0–C3）</h4>
  <p>
    Currently <b>Level 4</b>：C0 contactless → C1 light/Instantaneous → C2 point/line of continuous contact →
    <b>C3 high contact</b>(formerly "large area/"Load-bearing" and "winding"/Confrontation" merger).
    Reason for merger: close-fitting load-bearing (hug/piggyback) versus fighting (fight/wrestle) is not strictly sorted in terms of contact area,
    But both are significantly higher than C0–C2; Forcibly splitting it into two levels can easily create a false "difficulty ladder".
  </p>
  <ul>
    <li>C3 It still contains multiple interaction types; for fine-grained failure modes, please click board_cat（hug / fight ...) Separately.</li>
    <li>main curve view <b>C2→C3</b> and <b>Δ(C3−C0)</b>: high contact relatively low/Whether the contact is degraded.</li>
  </ul>
</div>
"""

    return f"""
<hr class="sep">
<h2>Anat / Inter × Contact density C0–C3(Core Observation Sheet)</h2>
<p class="note" style="margin-top:0;margin-bottom:12px">
  OK=model, column=Contact Density File (from <code>targets_tagged.csv</code> of <code>contact_c</code>,No board_cat）。
  <b>C3 = high contact</b>(Original C3∪C4: Close-fitting and load-bearing + Fighting and confrontation).
  Every report mesh <b>Anat</b> / <b>Inter</b> mean (only <code>ok</code> sample).
  right side <b>Δ(C3−C0)</b>: Negative value=Decreases as density increases (expected degradation).
  Hover the grid to see the number of samples in that file.
</p>
<table class="dens">
<thead>
<tr><th rowspan="2">Model</th>{top}<th colspan="2">Δ C3−C0</th></tr>
<tr>{sub}<th class="subh">Anat</th><th class="subh">Inter</th></tr>
</thead>
<tbody>
{"".join(body)}
</tbody>
</table>
<div class="charts dens-charts dens-multi-wrap">
{"".join(charts)}
</div>
{analysis}
"""


def load_mesh_summaries(pack: Path) -> Dict[str, dict]:
    root = pack / "judgments" / "mesh_v3"
    out: Dict[str, dict] = {}
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        sp = sub / "_summary.json"
        if not sp.is_file():
            continue
        try:
            out[sub.name] = json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            continue
    return out


def _n_ok(summary: dict, n_samples: int) -> bool:
    if n_samples <= 0:
        return True
    mn = summary.get("n")
    if mn is None:
        return True
    return abs(int(mn) - int(n_samples)) <= max(30, int(0.3 * n_samples))


def get_mesh_by_model(
    pack: Path,
    *,
    force_fallback: bool = False,
    n_samples: int = 0,
    model_ids: Optional[List[str]] = None,
) -> Tuple[Dict[str, dict], str]:
    """Load mesh. Reject n-mismatched summaries; recompute from per-sample files.

    FALLBACK_MESH only for smoke-scale packs (n<=200) when that model has no files.
    Never invent mesh numbers for full packs.
    """
    if force_fallback:
        if n_samples > 200:
            return {}, "fallback_blocked_for_full_pack"
        return {k: dict(v) for k, v in FALLBACK_MESH.items()}, "fallback_embedded"

    root = pack / "judgments" / "mesh_v3"
    loaded = load_mesh_summaries(pack)
    out: Dict[str, dict] = {}
    notes: List[str] = []
    dirs = {sub.name for sub in root.iterdir() if sub.is_dir()} if root.is_dir() else set()
    want = set(model_ids or []) | set(loaded) | dirs
    want.discard("")

    for mid in sorted(want):
        m = loaded.get(mid)
        if m is not None and not _n_ok(m, n_samples):
            notes.append(f"{mid}:reject_n={m.get('n')}≠{n_samples}")
            m = None
        if m is not None and m.get("S_anat_mesh") is not None:
            out[mid] = m
            continue

        # Prefer recomputing from per-sample files (fixes corrupt/mismatched summary).
        recomputed = summarize_mesh_dir(root / mid) if root.is_dir() else None
        if recomputed is not None and _n_ok(recomputed, n_samples):
            out[mid] = recomputed
            notes.append(f"{mid}:recomputed_n={recomputed['n']}")
            continue
        if recomputed is not None:
            notes.append(f"{mid}:recomputed_reject_n={recomputed.get('n')}≠{n_samples}")

        # smoke-scale only: embedded FALLBACK when that model truly has no mesh files
        if n_samples > 0 and n_samples <= 200 and mid in FALLBACK_MESH:
            out[mid] = dict(FALLBACK_MESH[mid])
            notes.append(f"{mid}:fallback")

    src = "pack/judgments/mesh_v3"
    if notes:
        src = "hybrid:" + ",".join(notes[:16])
    return out, src


def _coverage_ok(n_meas: int, *, n_out: int, n_pack: int, floor: float = 0.90) -> bool:
    """Only publish a mean when the axis is ~complete (not a partial mid-run mean)."""
    if n_meas <= 0:
        return False
    # Full pack: require ~pack coverage, OR finished scoring this model's available gens
    # (closed APIs may stop short of pack, e.g. gpt-image-2 at 2078/2500).
    if n_pack > 200:
        if n_meas >= floor * n_pack:
            return True
        if n_out > 0 and n_meas >= floor * n_out and n_out >= 0.80 * n_pack:
            return True
        return False
    # Smoke: relative to available outputs / pack
    target = max(n_out, n_pack, 1)
    return n_meas >= floor * target


def _empty_model_stub(mid: str) -> dict:
    return {
        "model_id": mid,
        "n_outputs": 0,
        "n_vlm": 0,
        "n_main_vlm": 0,
        "n_arcface": 0,
        "n_hpsv2": 0,
        "n_instr_v2": 0,
        "S_count": None,
        "S_anat": None,
        "S_inter": None,
        "S_instr": None,
        "S_instr_source": "missing",
        "S_instr_asymm": None,
        "S_instr_role_duty": None,
        "S_instr_prop_object": None,
        "p_perfect": None,
        "S_id": None,
        "HPSv2": None,
        "status": {
            "vlm_four_axis": False,
            "arcface": False,
            "hpsv2": False,
            "instr_v2": False,
        },
    }


def _display_row_v3(m: dict, m_mesh: dict, *, n_pack: int) -> dict:
    """Protocol v3 display: canonical sources only; incomplete runs stay blank."""
    n_out = int(m.get("n_outputs") or 0)
    n_count = int(m.get("n_vlm") or m.get("n_main_vlm") or 0)
    n_id = int(m.get("n_arcface") or 0)
    n_instr = int(m.get("n_instr_v2") or 0)
    n_qual = int(m.get("n_hpsv2") or 0)

    mesh_anat = m_mesh.get("S_anat_mesh")
    mesh_inter = m_mesh.get("S_inter_mesh")
    # mesh summaries are pack-level means; trust when present + n matches (handled upstream)
    has_mesh_anat = mesh_anat is not None
    has_mesh_inter = mesh_inter is not None

    has_instr_v2 = (m.get("S_instr_source") == "instr_v2") and m.get("S_instr") is not None
    has_arc = m.get("S_id") is not None and (m.get("status") or {}).get("arcface")
    has_hps = m.get("HPSv2") is not None and (m.get("status") or {}).get("hpsv2")

    count_ok = m.get("S_count") is not None and _coverage_ok(
        n_count, n_out=n_out, n_pack=n_pack
    )
    id_ok = bool(has_arc) and _coverage_ok(n_id, n_out=n_out, n_pack=n_pack)
    # Instr bank may be slightly < pack (e.g. 2486/2500); allow that ceiling
    instr_target_pack = n_pack
    instr_ok = bool(has_instr_v2) and (
        _coverage_ok(n_instr, n_out=n_out, n_pack=instr_target_pack)
        or (n_pack > 200 and n_instr >= 0.90 * min(n_out or n_pack, n_pack) and n_instr >= 2400)
    )
    qual_ok = bool(has_hps) and _coverage_ok(n_qual, n_out=n_out, n_pack=n_pack)

    return {
        "Count": float(m["S_count"]) if count_ok else None,
        "ID": float(m["S_id"]) if id_ok else None,
        "Anat": float(mesh_anat) if has_mesh_anat else None,
        "Inter": float(mesh_inter) if has_mesh_inter else None,
        "Anat_source": "mesh" if has_mesh_anat else "missing",
        "Inter_source": "mesh" if has_mesh_inter else "missing",
        "Instr": float(m["S_instr"]) if instr_ok else None,
        "Instr_source": "instr_v2" if instr_ok else "missing",
        "Instr_asymm": m.get("S_instr_asymm") if instr_ok else None,
        "Instr_role_duty": m.get("S_instr_role_duty") if instr_ok else None,
        "Instr_prop_object": m.get("S_instr_prop_object") if instr_ok else None,
        "p_perfect": m.get("p_perfect") if instr_ok else None,
        "Qual": float(m["HPSv2"]) if qual_ok else None,
        "partial_count": (m.get("S_count") is not None) and not count_ok,
        "partial_instr": has_instr_v2 and not instr_ok,
    }


def build_page_summary(pack: Path, force_fallback: bool = False) -> dict:
    v2 = aggregate_v2(pack)
    n_pack = int(v2.get("n_samples") or 0)
    # Always show the fixed zoo (3 closed + 7 open), even if no outputs yet.
    want = list(MODEL_DISPLAY_ORDER)
    mesh, mesh_src = get_mesh_by_model(
        pack,
        force_fallback=force_fallback,
        n_samples=n_pack,
        model_ids=want,
    )

    by_model = {}
    for mid in want:
        m = dict(v2["by_model"].get(mid) or _empty_model_stub(mid))
        m_mesh = mesh.get(mid) or {}
        d = _display_row_v3(m, m_mesh, n_pack=n_pack)
        by_model[mid] = {
            **m,
            "display": d,
            "S_inter_mesh": m_mesh.get("S_inter_mesh"),
            "S_anat_mesh": m_mesh.get("S_anat_mesh"),
            "P_anat_extra_mean": m_mesh.get("P_anat_extra_mean"),
            "anat_orphan_frac_mean": m_mesh.get("anat_orphan_frac_mean"),
            "anat_leftover_frac_mean": m_mesh.get("anat_leftover_frac_mean"),
        }

    ranking = order_models_closed_then_open(list(by_model.keys()))
    by_density = aggregate_mesh_by_density(pack, ranking)
    return {
        "protocol": "eval_protocol_v3_mesh",
        "pack": str(pack),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "n_samples": n_pack,
        "models": list(by_model.keys()),
        "ranking": ranking,
        "by_model": by_model,
        "mesh_extra": mesh,
        "mesh_data_source": mesh_src,
        "by_density": by_density,
        "note": (
            "Main table fixed 10 Model(3 Closed source + 7 open source); table sequence closed source→Open source; none Overall。"
            "The main table writes the quality score near the end of the run:≥about 90% pack, or the model has been scored for all available drawings (drawings≥about 80% pack,like gpt-image-2 of 2078）。"
            "Anat/Inter only mesh；ID only ArcFace；Qual only HPSv2；Instr only instr_v2;prohibit VLM replace. "
            "In addition C0–C3 Contact Density Breakdown (targets_tagged.contact_c;Original C3∪C4→C3）。"
        ),
    }


def _chart_block(title: str, vals: List[tuple], color: str) -> str:
    if not vals:
        return ""
    vmax = max(abs(float(v)) for _, v in vals) or 1.0
    scale = 1.0 if vmax <= 1.05 else vmax
    bars = []
    for name, v in vals:
        pct_w = max(0.0, min(100.0, 100.0 * float(v) / scale))
        bars.append(
            f'<div class="bar-row"><span class="bn">{name}</span>'
            f'<div class="track"><div class="fill" style="width:{pct_w:.1f}%;background:{color}"></div></div>'
            f'<span class="bv">{fmt(v)}</span></div>'
        )
    return f'<div class="chart"><h3>{title}</h3>{"".join(bars)}</div>'


def _render_instr_detail_section(summary: dict) -> str:
    keys = [
        ("Instr", "Instr", "#10b981"),
        ("asymm×0.50", "Instr_asymm", "#059669"),
        ("role_duty×0.35", "Instr_role_duty", "#0d9488"),
        ("prop_object×0.15", "Instr_prop_object", "#14b8a6"),
        ("p✓ (Instr≥.999)", "p_perfect", "#64748b"),
    ]
    rows = []
    for mid in summary["ranking"]:
        d = summary["by_model"][mid]["display"]
        rows.append({"model": mid, **d})
    if not rows:
        return ""
    head = "".join(f"<th>{title}</th>" for title, _, _ in keys)
    body = []
    for r in rows:
        cells = "".join(f'<td class="num">{fmt(r.get(k))}</td>' for _, k, _ in keys)
        body.append(f'<tr><td class="model">{r["model"]}</td>{cells}</tr>')
    charts = []
    for title, key, color in keys[1:]:
        vals = [(r["model"], r[key]) for r in rows if r.get(key) is not None]
        block = _chart_block(title, vals, color)
        if block:
            charts.append(block)
    return f"""
<hr class="sep">
<h2>Instr Supplementary (weighted sub-scores)</h2>
<p class="note" style="margin-top:0;margin-bottom:12px">
  main table <b>Instr</b> = Weighted normalization of available subtype means:
  <code>asymm 0.50</code> + <code>role_duty 0.35</code> + <code>prop_object 0.15</code>
  (If a bucket is missing, it will be reset to one). Below is the diagnostic column.
</p>
<table>
<thead><tr><th>Model</th>{head}</tr></thead>
<tbody>
{"".join(body)}
</tbody>
</table>
<div class="charts">
{"".join(charts)}
</div>
"""


def render_html(summary: dict) -> str:
    metrics = ["Count", "ID", "Anat", "Inter", "Instr", "Qual"]
    metric_headers = {
        "Count": "Count",
        "ID": "ID",
        "Anat": 'Anat<br><span class="th-hint">mesh</span>',
        "Inter": 'Inter<br><span class="th-hint">mesh</span>',
        "Instr": 'Instr<br><span class="th-hint">weighted total score</span>',
        "Qual": 'Qual<br><span class="th-hint">HPS≈0.20–0.35</span>',
    }
    colors = {
        "Count": "#3b82f6",
        "ID": "#8b5cf6",
        "Anat": "#ef4444",
        "Inter": "#f59e0b",
        "Instr": "#10b981",
        "Qual": "#06b6d4",
    }

    rows_data = []
    for mid in summary["ranking"]:
        d = summary["by_model"][mid]["display"]
        rows_data.append({"model": mid, **d})

    trs = []
    for r in rows_data:
        cells = "".join(f'<td class="num">{fmt(r[k])}</td>' for k in metrics)
        src_a = r.get("Anat_source") or "?"
        src_i = r.get("Inter_source") or "?"
        tip = f' title="Anat={src_a} Inter={src_i}"'
        trs.append(f'<tr{tip}><td class="model">{r["model"]}</td>{cells}</tr>')
    table_body = "\n".join(trs)

    chart_blocks = []
    for k in metrics:
        vals = [(r["model"], r[k]) for r in rows_data if r[k] is not None]
        block = _chart_block(k, vals, colors[k])
        if block:
            chart_blocks.append(block)

    mesh = summary.get("mesh_extra") or {}
    mesh_order = [m for m in summary["ranking"] if m in mesh]
    if "_gt" in mesh:
        mesh_order = mesh_order + ["_gt"]
    mesh_metrics = [
        ("Inter_mesh", "S_inter_mesh", "#f59e0b"),
        ("Anat_mesh", "S_anat_mesh", "#ef4444"),
        ("P_extra", "P_anat_extra_mean", "#8b5cf6"),
        ("orphan", "anat_orphan_frac_mean", "#06b6d4"),
        ("leftover", "anat_leftover_frac_mean", "#64748b"),
    ]
    mesh_rows = []
    for mid in mesh_order:
        m = mesh[mid]
        name = "GT(reference)" if mid == "_gt" else mid
        mesh_rows.append({"model": name, "mid": mid, **m})

    m_trs = []
    for r in mesh_rows:
        cells = "".join(
            f'<td class="num">{fmt(r.get(k))}</td>' for _, k, _ in mesh_metrics
        )
        m_trs.append(f'<tr><td class="model">{r["model"]}</td>{cells}</tr>')
    mesh_table = "\n".join(m_trs)

    mesh_charts = []
    for title, key, color in mesh_metrics:
        vals = [(r["model"], r[key]) for r in mesh_rows if r.get(key) is not None]
        block = _chart_block(title, vals, color)
        if block:
            mesh_charts.append(block)

    id_src = (
        "ArcFace"
        if any(summary["by_model"][m]["status"].get("arcface") for m in summary["models"])
        else "VLM(temporary)"
    )
    qual_src = (
        "HPSv2"
        if any(summary["by_model"][m]["status"].get("hpsv2") for m in summary["models"])
        else "VLM(temporary)"
    )
    instr_detail = _render_instr_detail_section(summary)
    density_detail = _render_density_section(summary)
    title_n = summary.get("n_samples")
    page_title = (
        f"MPIE-Bench · N={title_n}"
        if title_n
        else "MPIE-Bench evaluation"
    )

    return f"""<!DOCTYPE html>
<html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>{page_title}</title>
<style>
:root {{ --bg:#f7f8fa; --card:#fff; --line:#e8eaee; --text:#1f2430; --muted:#6b7280; }}
* {{ box-sizing:border-box }}
body {{ margin:0; padding:28px 24px 48px; font-family:-apple-system,BlinkMacSystemFont,Helvetica,Arial,sans-serif;
  background:var(--bg); color:var(--text); }}
h1 {{ font-size:22px; margin:0 0 6px; font-weight:700; }}
.sub {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
h2 {{ font-size:15px; margin:28px 0 10px; font-weight:600; }}
.metrics {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:8px 16px;
  background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin-bottom:22px; }}
.metrics dt {{ font-weight:600; font-size:13px; }}
.metrics dd {{ margin:0 0 6px; color:var(--muted); font-size:12.5px; line-height:1.45; }}
table {{ width:100%; max-width:1100px; border-collapse:collapse; background:var(--card);
  border:1px solid var(--line); border-radius:10px; overflow:hidden; font-size:13.5px; }}
th, td {{ padding:10px 12px; border-bottom:1px solid #f0f2f5; }}
th {{ background:#fafbfc; color:var(--muted); font-size:12px; font-weight:600; text-align:right; }}
th:first-child, td.model {{ text-align:left; font-weight:600; }}
th .th-hint {{ display:block; font-weight:500; font-size:10px; color:#9ca3af; margin-top:2px; }}
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
hr.sep {{ border:0; border-top:1px solid var(--line); margin:36px 0 8px; max-width:1100px; }}
.howto {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:14px 18px;
  max-width:960px; margin-bottom:22px; font-size:13px; line-height:1.55; color:#374151; }}
.howto p {{ margin:0 0 12px; }}
.howto h4 {{ margin:16px 0 8px; font-size:13px; font-weight:700; color:#1f2430; }}
.howto h4:first-child {{ margin-top:0; }}
.howto ul {{ margin:0 0 12px; padding-left:1.2em; }}
.howto code {{ display:inline-block; margin-top:4px; padding:2px 6px; background:#f3f4f6; border-radius:4px;
  font-size:12px; color:#1f2430; }}
.tag-up {{ color:#059669; font-weight:600; }}
.tag-dn {{ color:#dc2626; font-weight:600; }}
.warn {{ background:#fff7ed; border:1px solid #fdba74; color:#9a3412; padding:10px 12px; border-radius:8px;
  font-size:12.5px; max-width:960px; margin-bottom:16px; }}
table.dens {{ max-width:1200px; font-size:12.5px; }}
table.dens th.cband {{ text-align:center; border-left:1px solid #eef0f3; }}
table.dens th.subh {{ font-size:11px; font-weight:500; }}
table.dens td.delta {{ color:#b45309; font-weight:600; }}
.dens-charts {{ margin-top:14px; }}
.dens-multi-wrap {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; max-width:1200px; }}
@media (max-width: 980px) {{ .dens-multi-wrap {{ grid-template-columns:1fr; }} }}
.dens-multi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
.dens-multi h3 {{ margin:0 0 8px; font-size:14px; }}
.dens-legend {{ display:flex; flex-wrap:wrap; gap:8px 12px; margin:0 0 8px; font-size:11px; color:#4b5563; }}
.dens-legend .leg {{ display:inline-flex; align-items:center; gap:5px; }}
.dens-legend .leg i {{ width:10px; height:10px; border-radius:2px; display:inline-block; }}
.dens-analysis {{ margin-top:16px; max-width:1200px; }}
</style></head><body>
<h1>{page_title}</h1>
<p class="sub">N = {summary['n_samples']} · {summary['generated_at'][:19]} ·
  mesh source: <code>{summary.get('mesh_data_source','')}</code>
</p>
<div class="warn">Main table: six axes for models with near-complete coverage.
Missing cells are shown as <b>—</b>. Anat/Inter use Multi-HMR mesh;
ID=ArcFace; Qual=HPSv2; Instr=frozen Instr QA; Count=VLM.</div>

<h2>Metric glossary</h2>
<dl class="metrics">
  <div><dt>Count</dt><dd>Person-count correctness (VLM judge).</dd></div>
  <div><dt>ID</dt><dd>Identity preservation vs. references (source: {id_src}).</dd></div>
  <div><dt>Anat</dt><dd>Anatomy plausibility from Multi-HMR mesh (leftover/orphan + residual/structure/detect terms).</dd></div>
  <div><dt>Inter</dt><dd>Interaction geometry from Multi-HMR mesh (penetration / proximity vs. prompt contact intent).</dd></div>
  <div><dt>Instr</dt><dd>Instruction fidelity from frozen atomic QA (asymm 0.50 / role_duty 0.35 / prop_object 0.15).</dd></div>
  <div><dt>Qual</dt><dd>HPSv2 raw score (source: {qual_src}); typical range 0.20–0.35.</dd></div>
</dl>

<h2>Main results</h2>
<p class="note" style="margin-top:0">Cells are <b>0–1 quality scores</b> when coverage is near-complete; otherwise —.</p>
<table>
<thead><tr><th>Model</th>{''.join(f'<th>{metric_headers[k]}</th>' for k in metrics)}</tr></thead>
<tbody>
{table_body}
</tbody>
</table>

<h2>Comparison of items</h2>
<div class="charts">
{''.join(chart_blocks)}
</div>

<p class="note">The higher the score, the better (Qual Except, its scale is different). Sort by Anat/Inter/Count。</p>

{density_detail}

{instr_detail}

<hr class="sep">
<h2>Mesh Anat / Inter detail</h2>
<p class="note" style="margin-top:0;margin-bottom:14px">
  main table <b>Anat</b> = <b>Anat_mesh</b>，<b>Inter</b> = <b>Inter_mesh</b>。
  The last three columns (P_extra / orphan / leftover) is a diagnostic quantity. Data source:{summary.get('mesh_data_source','')}。
</p>

<h2>How to calculate it (popular version)</h2>
<div class="howto">
  <h4>common prefix</h4>
  <p>Multi-HMR After checking out the person, press prompt of R# number of people to do top-k（keep）。
  Anat and Inter <b>share the same group of people</b>。</p>
  <h4>Anat_mesh（↑the better)</h4>
  <p>ask"N How much of the human body can a frame cover?"; how much of the body can't be covered? → leftover / orphan → s_extra, and then with the residual/structure/Number of people fidelity bonus points synthesis.</p>
  <code>Anat ≈ 0.40 s_extra + 0.30 s_resid + 0.15 s_struct + 0.15 s_detect</code>（s=1−P）
  <h4>Inter_mesh（↑the better)</h4>
  <p>look at two people 3D: Whether the mold penetration and contact are consistent prompt Intention (to hold) / Don't touch / not specified).</p>
  <code>Inter ≈ 0.55 s_pen + 0.45 s_prox</code>（required；forbidden use s_clear；unspecified only s_pen）
</div>

<table>
<thead><tr><th>Model</th>{''.join(f'<th>{t}</th>' for t,_,_ in mesh_metrics)}</tr></thead>
<tbody>
{mesh_table}
</tbody>
</table>
<div class="charts">
{''.join(mesh_charts)}
</div>
<p class="note">GT Rows are for reference only.P_extra / orphan / leftover Lower is usually better;Anat_mesh / Inter_mesh The higher the better.</p>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    default_pack = repo_root / "data" / "testset"
    ap.add_argument(
        "--pack",
        default=str(default_pack if (default_pack / "manifest.jsonl").exists() else Path.home() / "mpie_testset_pack"),
    )
    ap.add_argument("--out", default="")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument(
        "--force-fallback",
        action="store_true",
        help="use embedded fallback mesh numbers even if pack has summaries",
    )
    args = ap.parse_args()
    pack = Path(args.pack).expanduser().resolve()
    default_out = repo_root / "data" / "eval_outputs" / "latest"
    out = Path(args.out).expanduser() if args.out else default_out
    out.mkdir(parents=True, exist_ok=True)

    summary = build_page_summary(pack, force_fallback=args.force_fallback)
    if args.models:
        keep = set(args.models)
        summary["by_model"] = {k: v for k, v in summary["by_model"].items() if k in keep}
        summary["models"] = [m for m in summary["models"] if m in keep]
        summary["ranking"] = [m for m in summary["ranking"] if m in keep]
        if summary.get("by_density"):
            summary["by_density"] = {
                k: v for k, v in summary["by_density"].items() if k in keep
            }

    dump = {k: v for k, v in summary.items() if k != "by_model"}
    dump["by_model"] = {
        mid: {
            "display": m["display"],
            "S_instr": m.get("S_instr"),
            "S_instr_asymm": m.get("S_instr_asymm"),
            "S_instr_role_duty": m.get("S_instr_role_duty"),
            "S_instr_prop_object": m.get("S_instr_prop_object"),
            "p_perfect": m.get("p_perfect"),
            "S_inter_mesh": m.get("S_inter_mesh"),
            "S_anat_mesh": m.get("S_anat_mesh"),
            "P_anat_extra_mean": m.get("P_anat_extra_mean"),
            "anat_orphan_frac_mean": m.get("anat_orphan_frac_mean"),
            "anat_leftover_frac_mean": m.get("anat_leftover_frac_mean"),
            "n_outputs": m.get("n_outputs"),
            "n_vlm": m.get("n_vlm") or m.get("n_main_vlm"),
            "n_main_vlm": m.get("n_main_vlm"),
            "n_arcface": m.get("n_arcface"),
            "n_hpsv2": m.get("n_hpsv2"),
            "n_instr_v2": m.get("n_instr_v2"),
            "status": m.get("status"),
        }
        for mid, m in summary["by_model"].items()
    }
    (out / "summary.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "index.html").write_text(render_html(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "ranking": summary["ranking"],
                "mesh_data_source": summary["mesh_data_source"],
                "anat_sources": {
                    mid: summary["by_model"][mid]["display"].get("Anat_source")
                    for mid in summary["ranking"]
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
