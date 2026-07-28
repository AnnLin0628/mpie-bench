#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""people–people IAA + H↔M / H↔V / M↔V Consistency report.

usage:
  python compute_agreement.py --pack "$MPIE_TEST_PACK" --split pilot
  python compute_agreement.py --pack ... --split pilot,holdout --judge-model gpt-5.5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import (  # noqa: E402
    ANAT_ITEMS,
    ANAT_PASS_ITEMS,
    INTER_ITEMS,
    atomic_write_json,
    pair_key,
)
from pack_io import pack_root  # noqa: E402


def cohen_kappa(y1: Sequence[int], y2: Sequence[int]) -> Optional[float]:
    pairs = [(int(a), int(b)) for a, b in zip(y1, y2) if a in (0, 1) and b in (0, 1)]
    n = len(pairs)
    if n == 0:
        return None
    tp = sum(1 for a, b in pairs if a == 1 and b == 1)
    tn = sum(1 for a, b in pairs if a == 0 and b == 0)
    fp = sum(1 for a, b in pairs if a == 0 and b == 1)
    fn = sum(1 for a, b in pairs if a == 1 and b == 0)
    po = (tp + tn) / n
    pe = (((tp + fp) / n) * ((tp + fn) / n)) + (((tn + fn) / n) * ((tn + fp) / n))
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def binary_stats(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    pairs = [(int(a), int(b)) for a, b in zip(y_true, y_pred) if a in (0, 1) and b in (0, 1)]
    n = len(pairs)
    if n == 0:
        return {"n": 0}
    tp = sum(1 for a, b in pairs if a == 1 and b == 1)
    tn = sum(1 for a, b in pairs if a == 0 and b == 0)
    fp = sum(1 for a, b in pairs if a == 0 and b == 1)
    fn = sum(1 for a, b in pairs if a == 1 and b == 0)
    acc = (tp + tn) / n
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r)) else None
    return {
        "n": n,
        "acc": acc,
        "P": p,
        "R": r,
        "F1": f1,
        "kappa": cohen_kappa([a for a, _ in pairs], [b for _, b in pairs]),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "pos_rate_true": (tp + fn) / n,
        "pos_rate_pred": (tp + fp) / n,
    }


def fleiss_kappa(ratings: List[List[int]]) -> Optional[float]:
    """ratings: N samples × k raters, values in {0,1}."""
    if not ratings:
        return None
    n = len(ratings)
    k = len(ratings[0])
    if k < 2:
        return None
    cats = (0, 1)
    P_i = []
    cat_counts = {c: 0 for c in cats}
    for row in ratings:
        cnt = {c: row.count(c) for c in cats}
        for c in cats:
            cat_counts[c] += cnt[c]
        P_i.append(
            (sum(v * v for v in cnt.values()) - k) / (k * (k - 1)) if k > 1 else 0.0
        )
    P_bar = sum(P_i) / n
    total = n * k
    p = {c: cat_counts[c] / total for c in cats}
    P_e = sum(v * v for v in p.values())
    if abs(1.0 - P_e) < 1e-12:
        return 1.0 if abs(P_bar - 1.0) < 1e-12 else 0.0
    return (P_bar - P_e) / (1.0 - P_e)


def get_item(obj: dict, item: str) -> Any:
    if item in ("Inter_pass", "Anat_pass"):
        return obj.get(item)
    if item.startswith("I"):
        return (obj.get("inter") or {}).get(item)
    return (obj.get("anat") or {}).get(item)


def load_json_map(dir_path: Path, *, key_from_name: bool = True) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not dir_path.is_dir():
        return out
    for p in dir_path.rglob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = obj.get("key")
        if not key and obj.get("sample_id") and obj.get("model_id"):
            key = pair_key(obj["sample_id"], obj["model_id"])
        if not key and key_from_name:
            key = p.stem
        if key:
            out[key] = obj
    return out


def load_human_raters(hc: Path) -> Dict[str, Dict[str, dict]]:
    """key -> {ann_id: record}"""
    root = hc / "human"
    by: Dict[str, Dict[str, dict]] = defaultdict(dict)
    if not root.is_dir():
        return by
    for p in root.glob("*/*.json"):
        if p.parent.name.startswith("_"):
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        key = obj.get("key") or pair_key(obj["sample_id"], obj["model_id"])
        by[key][p.parent.name] = obj
    return by


def split_keyset(split_path: Path, names: Sequence[str]) -> set:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    want = set(names)
    keys = set()
    for name, rows in (data.get("splits") or {}).items():
        if name not in want:
            continue
        for r in rows:
            keys.add(r.get("key") or pair_key(r["sample_id"], r["model_id"]))
    return keys


def pairwise_report(a: Dict[str, dict], b: Dict[str, dict], items: Sequence[str]) -> dict:
    out = {}
    for item in items:
        yt, yp = [], []
        for k in sorted(set(a) & set(b)):
            va, vb = get_item(a[k], item), get_item(b[k], item)
            if va in (0, 1) and vb in (0, 1):
                yt.append(int(va))
                yp.append(int(vb))
        out[item] = binary_stats(yt, yp)
    return out


def iaa_report(raters: Dict[str, Dict[str, dict]], items: Sequence[str]) -> dict:
    out = {}
    for item in items:
        matrix: List[List[int]] = []
        for key, ann_map in raters.items():
            vals = []
            for ann_id in sorted(ann_map):
                v = get_item(ann_map[ann_id], item)
                if v in (0, 1):
                    vals.append(int(v))
            if len(vals) >= 2 and len(vals) == len(ann_map):
                # Only hard tags for all members
                matrix.append(vals)
            elif len(vals) >= 2:
                # allow part U: Take at least 2 Sample of human hard label, right pad Not used for fleiss
                # Simplify: Require exactly 3 Everyone is 0/1
                pass
        # unified rater Number: keep only the exact 3 indivual 0/1 of
        mat3 = [row for row in matrix if len(row) == 3]
        out[item] = {
            "n": len(mat3),
            "fleiss_kappa": fleiss_kappa(mat3) if mat3 else None,
        }
    return out


def md_table(axis_blocks: dict) -> str:
    lines = [
        "| axis | H↔M κ | H↔V κ | M↔V κ | people–people Fleiss κ |",
        "|----|-------|-------|-------|----------------|",
    ]
    for item in ("Inter_pass", "Anat_pass", "I1", "A1"):
        hm = (axis_blocks.get("H_M") or {}).get(item, {}).get("kappa")
        hv = (axis_blocks.get("H_V") or {}).get(item, {}).get("kappa")
        mv = (axis_blocks.get("M_V") or {}).get(item, {}).get("kappa")
        iaa = (axis_blocks.get("IAA") or {}).get(item, {}).get("fleiss_kappa")

        def fmt(x):
            return "—" if x is None else f"{x:.3f}"

        lines.append(f"| {item} | {fmt(hm)} | {fmt(hv)} | {fmt(mv)} | {fmt(iaa)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="")
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--judge-model", default="gpt-5.5")
    args = ap.parse_args()

    root = pack_root(args.pack or None)
    hc = root / "judgments" / "human_consistency"
    names = [x.strip() for x in args.split.split(",") if x.strip()]
    keys = split_keyset(hc / "_split.json", names)

    H = {
        k: v
        for k, v in load_json_map(hc / "human" / "_consensus").items()
        if k in keys
    }
    M = {k: v for k, v in load_json_map(hc / "mesh_bin").items() if k in keys}
    Vroot = hc / "checklist_vlm" / args.judge_model.replace("/", "_")
    V = {k: v for k, v in load_json_map(Vroot).items() if k in keys}

    raters_all = load_human_raters(hc)
    raters = {k: v for k, v in raters_all.items() if k in keys}

    items = (
        list(INTER_ITEMS)
        + list(ANAT_ITEMS)
        + ["Inter_pass", "Anat_pass"]
    )
    # v4: The main text should also be reported S_* Related + intent Layering (subsequent enhancement); follow here first INTER/ANAT_ITEMS
    core = ["Inter_pass", "Anat_pass", "I1", "A1", "A5"]

    report = {
        "protocol": "checklist_anat_inter_v4",
        "split": names,
        "judge_model": args.judge_model,
        "n_keys_split": len(keys),
        "n_H": len(H),
        "n_M": len(M),
        "n_V": len(V),
        "n_human_multi": sum(1 for v in raters.values() if len(v) >= 2),
        "IAA": iaa_report(raters, core + ["I0", "Ic", "Ir", "A2", "A3", "A4"]),
        "H_M": pairwise_report(H, M, items) if H and M else {},
        "H_V": pairwise_report(H, V, items) if H and V else {},
        "M_V": pairwise_report(M, V, items) if M and V else {},
        "note": "v4: prefer construct-score correlation + intent-stratified item κ in paper tables",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    reports = hc / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "iaa_human.json", report["IAA"])
    atomic_write_json(reports / "agree_H_M.json", report["H_M"])
    atomic_write_json(reports / "agree_H_V.json", report["H_V"])
    atomic_write_json(reports / "agree_M_V.json", report["M_V"])
    atomic_write_json(reports / "agreement_full.json", report)

    paper = md_table(
        {
            "H_M": report["H_M"],
            "H_V": report["H_V"],
            "M_V": report["M_V"],
            "IAA": report["IAA"],
        }
    )
    note = (
        f"# Human consistency tables\n\n"
        f"split={names} · H={len(H)} M={len(M)} V={len(V)} · judge={args.judge_model}\n\n"
        f"{paper}\n"
        f"_Generated {report['written_at']}_\n"
    )
    (reports / "tables_for_paper.md").write_text(note, encoding="utf-8")

    print(
        json.dumps(
            {
                "wrote": str(reports),
                "n_H": len(H),
                "n_M": len(M),
                "n_V": len(V),
                "core_table": paper,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
