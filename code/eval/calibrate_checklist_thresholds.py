#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan on pilot set τ,maximize H*↔M of Cohen κ,freeze _thresholds.json。

need:
  - human/_consensus/*.json(Pilot majority vote)
  - judgments/mesh_v3/<model>/<sample>.json

usage:
  python calibrate_checklist_thresholds.py --pack "$MPIE_TEST_PACK" --split pilot
  python calibrate_checklist_thresholds.py --pack ... --split pilot --freeze
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    atomic_write_json,
    load_thresholds,
    map_mesh_to_checklist,
    mesh_rec_path,
    pair_key,
)
from pack_io import pack_root  # noqa: E402


def cohen_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> Optional[float]:
    pairs = [(int(a), int(b)) for a, b in zip(y_true, y_pred) if a in (0, 1) and b in (0, 1)]
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
    return float((po - pe) / (1.0 - pe))


def load_split_keys(split_path: Path, names: Sequence[str]) -> List[dict]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    want = set(names)
    out = []
    for name, rows in (data.get("splits") or {}).items():
        if name in want:
            out.extend(rows)
    return out


def load_consensus(hc: Path) -> Dict[str, dict]:
    d = hc / "human" / "_consensus"
    out: Dict[str, dict] = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        out[obj["key"]] = obj
    return out


def item_labels(h: dict, m: dict, item: str) -> Tuple[Optional[int], Optional[int]]:
    if item.startswith("I"):
        hv = (h.get("inter") or {}).get(item)
        mv = (m.get("inter") or {}).get(item)
    elif item in ("Inter_pass", "Anat_pass"):
        hv, mv = h.get(item), m.get(item)
    else:
        hv = (h.get("anat") or {}).get(item)
        mv = (m.get("anat") or {}).get(item)
    if hv in (0, 1) and mv in (0, 1):
        return int(hv), int(mv)
    return None, None


SCAN_GRID = {
    "tau_fuse": [0.30, 0.40, 0.50, 0.60, 0.70],
    "tau_miss": [0.30, 0.40, 0.50, 0.60, 0.70],
    "tau_unw": [0.30, 0.40, 0.50, 0.60, 0.70],
    "tau_extra": [0.20, 0.30, 0.35, 0.45, 0.55],
    "tau_own": [0.50, 0.60, 0.70, 0.80],
    "tau_scale": [0.50, 0.60, 0.70, 0.80],
    "tau_S_inter": [0.40, 0.50, 0.60, 0.70],
    "tau_S_anat": [0.40, 0.50, 0.60, 0.70],
}

ITEM_FOR_TAU = {
    "tau_fuse": "I1",
    "tau_miss": "I2",
    "tau_unw": "I3",
    "tau_extra": "A1",
    "tau_own": "A3",
    "tau_scale": "A4",
    "tau_S_inter": "Inter_pass",  # compared via M-score field separately below
    "tau_S_anat": "Anat_pass",
}


def eval_kappa_for_item(
    pairs: List[Tuple[dict, dict]],
    thr: dict,
    calib: dict,
    item: str,
    *,
    use_mscore: bool = False,
) -> Optional[float]:
    yt, yp = [], []
    for h, rec in pairs:
        m = map_mesh_to_checklist(rec, thresholds=thr, calib=calib)
        if use_mscore:
            if item == "Inter_pass":
                hv, mv = h.get("Inter_pass"), m.get("Inter_pass_Mscore")
            else:
                hv, mv = h.get("Anat_pass"), m.get("Anat_pass_Mscore")
            if hv in (0, 1) and mv in (0, 1):
                yt.append(int(hv))
                yp.append(int(mv))
            continue
        a, b = item_labels(h, m, item)
        if a is None:
            continue
        yt.append(a)
        yp.append(b)
    return cohen_kappa(yt, yp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="")
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--freeze", action="store_true", help="Write status=frozen")
    ap.add_argument("--force", action="store_true", help="Allow overwriting frozen of thresholds")
    args = ap.parse_args()

    root = pack_root(args.pack or None)
    hc = root / "judgments" / "human_consistency"
    split_path = hc / "_split.json"
    thr_path = hc / "_thresholds.json"
    if thr_path.is_file():
        cur = load_thresholds(thr_path)
        if cur.get("status") == "frozen" and not args.force:
            raise SystemExit(f"{thr_path} already frozen; pass --force to recalibrate")

    keys = load_split_keys(split_path, [x.strip() for x in args.split.split(",")])
    cons = load_consensus(hc)
    calib_path = root / "judgments" / "mesh_v3" / "_calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.is_file() else {}

    pairs: List[Tuple[dict, dict]] = []
    miss = 0
    for it in keys:
        key = it.get("key") or pair_key(it["sample_id"], it["model_id"])
        h = cons.get(key)
        if not h:
            continue
        mp = mesh_rec_path(root, it["model_id"], it["sample_id"])
        if not mp.is_file():
            miss += 1
            continue
        rec = json.loads(mp.read_text(encoding="utf-8"))
        rec.setdefault("sample_id", it["sample_id"])
        rec.setdefault("model_id", it["model_id"])
        pairs.append((h, rec))

    if len(pairs) < 10:
        raise SystemExit(
            f"need ≥10 consensus+mesh pairs on split={args.split}; "
            f"have {len(pairs)} (miss_mesh_among_cons_or_split≈{miss}, "
            f"consensus={len(cons)}). Finish human pilot + sync mesh_v3 first."
        )

    base = dict(DEFAULT_THRESHOLDS)
    if thr_path.is_file():
        base.update({k: v for k, v in load_thresholds(thr_path).items() if not str(k).startswith("_")})

    best = deepcopy(base)
    scans: Dict[str, Any] = {}
    for tau_name, grid in SCAN_GRID.items():
        item = ITEM_FOR_TAU[tau_name]
        use_ms = tau_name in ("tau_S_inter", "tau_S_anat")
        rows = []
        best_k, best_v = -999.0, best.get(tau_name)
        for v in grid:
            thr = deepcopy(best)
            thr[tau_name] = v
            k = eval_kappa_for_item(pairs, thr, calib, item, use_mscore=use_ms)
            rows.append({"value": v, "kappa": k, "n_pairs": len(pairs)})
            if k is not None and k > best_k:
                best_k, best_v = k, v
        best[tau_name] = best_v
        scans[tau_name] = {"item": item, "best": best_v, "best_kappa": best_k, "grid": rows}

    # Shaft level M-item κ(Use final best）
    axis = {}
    for item in ("Inter_pass", "Anat_pass", "I1", "A1"):
        axis[item] = eval_kappa_for_item(pairs, best, calib, item, use_mscore=False)

    best.update(
        {
            "version": "map_v1",
            "status": "frozen" if args.freeze else "calibrated_unfrozen",
            "calibrated_on_split": args.split,
            "n_pairs": len(pairs),
            "axis_kappa_Mitem": axis,
            "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": "τ Maximizing in pilot H↔M κ；holdout/main Only rate once",
        }
    )
    atomic_write_json(thr_path, best)
    atomic_write_json(hc / "reports" / "calibrate_scan.json", {"thresholds": best, "scans": scans})
    print(
        json.dumps(
            {
                "wrote": str(thr_path),
                "status": best["status"],
                "n_pairs": len(pairs),
                "axis_kappa_Mitem": axis,
                "chosen": {k: best[k] for k in SCAN_GRID},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
