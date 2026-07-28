#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spearman rank correlation of Anat/Inter across mesh frontends.

Example:
  python corr_mesh_frontends.py \\
    --pack "$MPIE_TEST_PACK" \\
    --model-ids flux1-kontext-dev,ace,omnigen2 \\
    --frontends multi_hmr,hmr2,smpler_x
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    # average ranks for ties
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
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    deny = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if denx < 1e-12 or deny < 1e-12:
        return None
    return float(num / (denx * deny))


JUDGE = {
    "multi_hmr": "mesh_v3",
    "multihmr": "mesh_v3",
    "hmr2": "mesh_hmr2",
    "smpler_x": "mesh_smplerx",
    "smplerx": "mesh_smplerx",
}


def load_scores(pack: Path, judge: str, mid: str) -> Dict[str, Dict[str, float]]:
    d = pack / "judgments" / judge / mid
    out: Dict[str, Dict[str, float]] = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not rec.get("ok"):
            continue
        sid = rec.get("sample_id") or p.stem
        a, i = rec.get("S_anat_mesh"), rec.get("S_inter_mesh")
        try:
            if a is None or i is None:
                continue
            out[sid] = {"S_anat_mesh": float(a), "S_inter_mesh": float(i)}
        except (TypeError, ValueError):
            continue
    return out


def fmt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--model-ids", required=True, help="comma-separated")
    ap.add_argument("--frontends", default="multi_hmr,hmr2,smpler_x")
    ap.add_argument("--out", default="", help="json path; default $PACK/judgments/_frontend_corr.json")
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    mids = [x.strip() for x in args.model_ids.split(",") if x.strip()]
    fronts = [x.strip() for x in args.frontends.split(",") if x.strip()]
    judges = {f: JUDGE.get(f.replace("-", "_"), f"mesh_{f}") for f in fronts}

    pairs: List[Tuple[str, str]] = []
    for i in range(len(fronts)):
        for j in range(i + 1, len(fronts)):
            pairs.append((fronts[i], fronts[j]))

    report: Dict[str, Any] = {
        "pack": str(pack),
        "model_ids": mids,
        "frontends": fronts,
        "judges": judges,
        "by_model": {},
        "pooled": {},
    }

    # pooled across models
    pooled: Dict[str, Dict[str, Dict[str, float]]] = {f: {} for f in fronts}
    # key = f"{mid}::{sid}"

    md = [
        "# Mesh frontend rank correlation",
        "",
        f"Pack: `{pack}`",
        "",
        "| model | pair | n | ρ Anat | ρ Inter |",
        "|---|---|---:|---:|---:|",
    ]

    for mid in mids:
        scores = {f: load_scores(pack, judges[f], mid) for f in fronts}
        report["by_model"][mid] = {}
        for f in fronts:
            for sid, v in scores[f].items():
                pooled[f][f"{mid}::{sid}"] = v
        for a, b in pairs:
            common = sorted(set(scores[a]) & set(scores[b]))
            xa = [scores[a][s]["S_anat_mesh"] for s in common]
            ya = [scores[b][s]["S_anat_mesh"] for s in common]
            xi = [scores[a][s]["S_inter_mesh"] for s in common]
            yi = [scores[b][s]["S_inter_mesh"] for s in common]
            ra, ri = spearman(xa, ya), spearman(xi, yi)
            report["by_model"][mid][f"{a}__vs__{b}"] = {
                "n": len(common),
                "rho_anat": ra,
                "rho_inter": ri,
            }
            md.append(
                f"| {mid} | {a} vs {b} | {len(common)} | {fmt(ra)} | {fmt(ri)} |"
            )
            print(
                f"{mid:24s} {a:10s} vs {b:10s}  n={len(common):4d}  "
                f"Anat ρ={fmt(ra)}  Inter ρ={fmt(ri)}",
                flush=True,
            )

    md += ["", "## Pooled (all models)", ""]
    md += ["| pair | n | ρ Anat | ρ Inter |", "|---|---:|---:|---:|"]
    for a, b in pairs:
        common = sorted(set(pooled[a]) & set(pooled[b]))
        xa = [pooled[a][k]["S_anat_mesh"] for k in common]
        ya = [pooled[b][k]["S_anat_mesh"] for k in common]
        xi = [pooled[a][k]["S_inter_mesh"] for k in common]
        yi = [pooled[b][k]["S_inter_mesh"] for k in common]
        ra, ri = spearman(xa, ya), spearman(xi, yi)
        report["pooled"][f"{a}__vs__{b}"] = {
            "n": len(common),
            "rho_anat": ra,
            "rho_inter": ri,
        }
        md.append(f"| {a} vs {b} | {len(common)} | {fmt(ra)} | {fmt(ri)} |")
        print(
            f"{'POOLED':24s} {a:10s} vs {b:10s}  n={len(common):4d}  "
            f"Anat ρ={fmt(ra)}  Inter ρ={fmt(ri)}",
            flush=True,
        )

    out = Path(args.out).expanduser() if args.out else (
        pack / "judgments" / "_frontend_corr.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}\nwrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
