#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare Instr v1 vs v2 discrimination on a pack.

Usage:
  python report_instr_v2.py --pack "$MPIE_TEST_PACK"
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from aggregate_vlm_judge_v1 import order_models_closed_then_open, score_axes
from instr_qa_common import MAIN_BUCKETS, MAIN_SUBTYPES, REVISION
from pack_io import pack_root


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load_instr_v2_means(pack: Path) -> dict[str, dict]:
    root = pack / "judgments" / "instr_v2"
    out = {}
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        scores, role, asymm, prop, scene = [], [], [], [], []
        role_duty, prop_object = [], []
        n = 0
        for p in sub.glob("*.json"):
            if p.name.startswith("_"):
                continue
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            meta = j.get("_meta") if isinstance(j.get("_meta"), dict) else {}
            rev = j.get("revision") or meta.get("revision")
            if rev and rev != REVISION:
                continue
            if j.get("confidence") == "low":
                continue
            if j.get("S_instr") is None:
                continue
            n += 1
            scores.append(float(j["S_instr"]))
            if j.get("S_instr_role") is not None:
                role.append(float(j["S_instr_role"]))
            if j.get("S_instr_asymm") is not None:
                asymm.append(float(j["S_instr_asymm"]))
            if j.get("S_instr_prop") is not None:
                prop.append(float(j["S_instr_prop"]))
            if j.get("S_instr_scene") is not None:
                scene.append(float(j["S_instr_scene"]))
            if j.get("S_instr_role_duty") is not None:
                role_duty.append(float(j["S_instr_role_duty"]))
            if j.get("S_instr_prop_object") is not None:
                prop_object.append(float(j["S_instr_prop_object"]))
        if n:
            out[sub.name] = {
                "n": n,
                "S_instr": _mean(scores),
                "std": st.pstdev(scores) if len(scores) > 1 else 0.0,
                "p_perfect": sum(1 for s in scores if s >= 0.999) / len(scores),
                "S_instr_role": _mean(role),
                "S_instr_asymm": _mean(asymm),
                "S_instr_prop": _mean(prop),
                "S_instr_scene": _mean(scene),
                "S_instr_role_duty": _mean(role_duty),
                "S_instr_prop_object": _mean(prop_object),
            }
    return out


def load_instr_v1_means(pack: Path) -> dict[str, dict]:
    root = pack / "judgments" / "vlm_judge_v1"
    out = {}
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        scores = []
        for p in sub.glob("*.json"):
            if p.name.startswith("_"):
                continue
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if j.get("confidence") == "low":
                continue
            ax = score_axes(j)
            if ax.get("instr") is not None:
                scores.append(float(ax["instr"]))
        if scores:
            out[sub.name] = {
                "n": len(scores),
                "S_instr": _mean(scores),
                "std": st.pstdev(scores) if len(scores) > 1 else 0.0,
                "p_perfect": sum(1 for s in scores if s >= 0.999) / len(scores),
            }
    return out


def qa_bank_stats(pack: Path) -> dict:
    d = pack / "instr_qa_v2"
    if not d.is_dir():
        return {"n": 0}
    n = 0
    buckets = {b: 0 for b in ("role", "asymm", "prop", "scene")}
    subtypes = {s: 0 for s in (*MAIN_SUBTYPES, "scene", "role_spatial", "prop_clothing")}
    warn = 0
    asymm_missing = 0
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = j.get("_meta") if isinstance(j.get("_meta"), dict) else {}
        rev = j.get("revision") or meta.get("revision")
        if rev and rev != REVISION:
            continue
        n += 1
        has_asymm = False
        for q in j.get("questions") or []:
            b = q.get("bucket")
            if b in buckets:
                buckets[b] += 1
            stype = q.get("subtype") or ""
            if stype in subtypes:
                subtypes[stype] += 1
            if stype == "asymm" or b == "asymm":
                has_asymm = True
        if not has_asymm:
            asymm_missing += 1
        if j.get("warnings"):
            warn += 1
    return {
        "n": n,
        "revision": REVISION,
        "bucket_question_counts": buckets,
        "subtype_question_counts": subtypes,
        "n_with_warnings": warn,
        "n_asymm_missing": asymm_missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="")
    ap.add_argument("--out", default="", help="optional JSON report path")
    args = ap.parse_args()
    pack = pack_root(args.pack or None)
    v1 = load_instr_v1_means(pack)
    v2 = load_instr_v2_means(pack)
    models = order_models_closed_then_open(sorted(set(v1) | set(v2)))
    rows = []
    for mid in models:
        rows.append(
            {
                "model_id": mid,
                "v1": v1.get(mid),
                "v2": v2.get(mid),
            }
        )
    report = {
        "pack": str(pack),
        "revision": REVISION,
        "qa_bank": qa_bank_stats(pack),
        "main_buckets": list(MAIN_BUCKETS),
        "main_subtypes": list(MAIN_SUBTYPES),
        "models": rows,
        "note": (
            f"{REVISION}: S_instr = weighted asymm/role_duty/prop_object; "
            "higher std / lower p_perfect = better separation among strong models."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
