#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit according to the question mesh→checklist map(ann_01 calibration), write _item_map_calib.json。

For each question: Select features → Standardized ridge regression predicts continuous scores → Quantile cut 0/1 or 0/1/2。
Do it at the same time 5-fold CV,and V Compare ρ, to avoid the illusion of only looking at the training set.

usage:
  python calibrate_mesh_item_map.py --pack "$MPIE_TEST_PACK"
  python export_mesh_checklist.py --pack ... --force
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import atomic_write_json, pair_key  # noqa: E402
from pack_io import pack_root  # noqa: E402

# Priority features for each question (data diagnosis + The meaning of the question; to join the puzzle through the pattern/Belonging signal, because P_fuse with people I1 Almost irrelevant)
ITEM_FEATS: Dict[str, List[str]] = {
    "I0": ["count_match", "S_count_mesh", "P_anat_detect", "n_humans", "n_expected"],
    "I1": [
        "S_pen",
        "P_fuse",
        "pen_vert_ratio",
        "pen_inside_ratio",
        "S_inter_mesh",
        "anat_leftover_frac",
        "ownership_confused",
        "ownership_score",
        "residual_iou",
        "S_anat_shape",
    ],
    "Ic": ["S_prox", "P_miss", "min_surf_dist", "S_inter_mesh", "S_anat_contact_region"],
    "I3": ["P_unwanted", "P_miss", "S_prox", "min_surf_dist", "S_inter_mesh"],
    "Ir": ["S_anat_contact_region", "contact_region_score", "S_prox", "P_miss"],
    "A1": [
        "ownership_score",
        "ownership_confused",
        "P_anat_extra",
        "anat_leftover_frac",
        "overcount_score",
        "residual_iou",
        "S_anat_overcount",
    ],
    "A2": [
        "S_anat_shape",
        "P_anat_resid",
        "S_anat_residual",
        "anat_leftover_frac",
        "residual_score",
        "S_anat_mesh",
    ],
    "A3": [
        "ownership_score",
        "ownership_confused",
        "residual_iou",
        "S_anat_ownership",
        "P_anat_extra",
    ],
    "A4": ["S_anat_scale", "cross_scale_score", "S_anat_shape", "S_anat_mesh"],
    "A5": [
        "S_anat_hand",
        "residual_iou",
        "S_anat_shape",
        "ownership_confused",
        "residual_score",
    ],
}

ORD3 = frozenset({"I1", "Ic", "A1", "A2", "A3"})


def _dig(d: dict, path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def _f(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def extract_features(rec: dict) -> Dict[str, float]:
    """from mesh_v3 Record extracted flat features. """
    f: Dict[str, float] = {}
    flat = [
        "P_fuse",
        "P_miss",
        "P_unwanted",
        "P_anat_extra",
        "P_anat_resid",
        "P_anat_struct",
        "P_anat_detect",
        "S_inter_mesh",
        "S_anat_mesh",
        "S_pen",
        "S_prox",
        "S_anat_ownership",
        "S_anat_scale",
        "S_anat_hand",
        "S_anat_overcount",
        "S_anat_residual",
        "S_anat_shape",
        "S_anat_self",
        "S_anat_bone",
        "S_anat_joint",
        "S_anat_contact_region",
        "S_anat_person",
        "S_count_mesh",
        "pen_volume_m3",
        "pen_depth_mean",
        "pen_inside_ratio",
        "pen_vert_ratio",
        "min_surf_dist",
        "anat_leftover_frac",
        "anat_n_leftover_blobs",
        "fuse_value",
        "n_humans",
        "n_expected",
    ]
    for k in flat:
        x = _f(rec.get(k))
        if x is not None:
            f[k] = x
    nested = {
        "ownership_score": "anat_scene.ownership.score",
        "ownership_confused": "anat_scene.ownership.confused_frac",
        "overcount_score": "anat_scene.structure_overcount.score",
        "residual_score": "anat_scene.explain_residual.score",
        "residual_iou": "anat_scene.explain_residual.iou",
        "contact_region_score": "anat_scene.contact_region_anat.score",
        "cross_scale_score": "anat_scene.cross_person_scale.score",
    }
    for name, path in nested.items():
        x = _f(_dig(rec, path))
        if x is not None:
            f[name] = x
    nh, ne = rec.get("n_humans"), rec.get("n_expected")
    if nh is not None and ne is not None:
        f["count_match"] = (
            1.0
            if int(nh) == int(ne) and not bool(rec.get("under_detect"))
            else 0.0
        )
    # Non-linear expansion (for v6.2 Calibration model used; when missing predict use median）
    base_sq = [
        "S_prox",
        "P_miss",
        "min_surf_dist",
        "S_inter_mesh",
        "S_pen",
        "P_fuse",
        "ownership_score",
        "ownership_confused",
        "residual_iou",
        "anat_leftover_frac",
        "S_anat_shape",
        "pen_vert_ratio",
        "overcount_score",
        "S_anat_contact_region",
        "contact_region_score",
    ]
    for k in base_sq:
        if k in f:
            f[f"{k}_sq"] = f[k] * f[k]
    if "S_prox" in f and "P_miss" in f:
        f["S_prox*(1-P_miss)"] = f["S_prox"] * (1.0 - f["P_miss"])
    if "S_prox" in f and "min_surf_dist" in f:
        f["S_prox-dist"] = f["S_prox"] - f["min_surf_dist"]
    if "ownership_confused" in f and "S_pen" in f:
        f["conf*(1-S_pen)"] = f["ownership_confused"] * (1.0 - f["S_pen"])
    return f


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    pairs = [(float(a), float(b)) for a, b in zip(xs, ys)]
    n = len(pairs)
    if n < 5:
        return None

    def ranks(vals: List[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    rx, ry = ranks(xv), ranks(yv)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx < 1e-12 or deny < 1e-12:
        return None
    return num / (denx * deny)


def _standardize(
    X: List[List[float]],
) -> Tuple[List[List[float]], List[float], List[float]]:
    n, m = len(X), len(X[0])
    means, stds = [], []
    for j in range(m):
        col = [X[i][j] for i in range(n)]
        mu = sum(col) / n
        var = sum((x - mu) ** 2 for x in col) / max(n - 1, 1)
        sd = math.sqrt(var) if var > 1e-12 else 1.0
        means.append(mu)
        stds.append(sd)
    Xs = [[(X[i][j] - means[j]) / stds[j] for j in range(m)] for i in range(n)]
    return Xs, means, stds


def _ridge_fit(X: List[List[float]], y: List[float], lam: float) -> List[float]:
    n = len(X)
    m = len(X[0]) + 1
    A = [[0.0] * m for _ in range(m)]
    b = [0.0] * m
    for i in range(n):
        xi = X[i] + [1.0]
        for a in range(m):
            b[a] += xi[a] * y[i]
            for c in range(m):
                A[a][c] += xi[a] * xi[c]
    for j in range(m - 1):
        A[j][j] += lam
    M = [A[i][:] + [b[i]] for i in range(m)]
    for col in range(m):
        piv = max(range(col, m), key=lambda r: abs(M[r][col]))
        M[col], M[piv] = M[piv], M[col]
        if abs(M[col][col]) < 1e-12:
            continue
        div = M[col][col]
        for j in range(col, m + 1):
            M[col][j] /= div
        for r in range(m):
            if r == col:
                continue
            fac = M[r][col]
            for j in range(col, m + 1):
                M[r][j] -= fac * M[col][j]
    return [M[i][m] for i in range(m)]


def _ridge_pred(w: List[float], x: List[float]) -> float:
    s = w[-1]
    for j, xj in enumerate(x):
        s += w[j] * xj
    return s


def _fit_cuts(
    pred: List[float], y: List[float], levels: int
) -> Tuple[Dict[str, Any], float]:
    """Search for the boundary on the predicted score, maximizing the y of Spearman。"""
    order = sorted(pred)
    best: Dict[str, Any] = {"levels": levels, "invert": False}
    best_rho = -9.0
    qs = [i / 24 for i in range(1, 24)]
    if levels == 2:
        for q in qs:
            thr = order[min(len(order) - 1, int(q * (len(order) - 1)))]
            for invert in (False, True):
                yhat = []
                for p in pred:
                    hi = p >= thr
                    yhat.append(int((not hi) if invert else hi))
                rho = spearman(yhat, y)
                if rho is not None and rho > best_rho:
                    best_rho = rho
                    best = {"levels": 2, "thr": thr, "invert": invert}
    else:
        for q1 in qs:
            for q2 in qs:
                if q2 <= q1:
                    continue
                t1 = order[min(len(order) - 1, int(q1 * (len(order) - 1)))]
                t2 = order[min(len(order) - 1, int(q2 * (len(order) - 1)))]
                if t2 <= t1:
                    continue
                for invert in (False, True):
                    yhat = []
                    for p in pred:
                        if not invert:
                            yhat.append(0 if p < t1 else (1 if p < t2 else 2))
                        else:
                            yhat.append(0 if p > t2 else (1 if p > t1 else 2))
                    rho = spearman(yhat, y)
                    if rho is not None and rho > best_rho:
                        best_rho = rho
                        best = {
                            "levels": 3,
                            "t1": t1,
                            "t2": t2,
                            "invert": invert,
                        }
    return best, best_rho


def apply_cuts(pred: float, cuts: dict) -> int:
    if cuts["levels"] == 2:
        hi = pred >= float(cuts["thr"])
        return int((not hi) if cuts.get("invert") else hi)
    t1, t2 = float(cuts["t1"]), float(cuts["t2"])
    if not cuts.get("invert"):
        return 0 if pred < t1 else (1 if pred < t2 else 2)
    return 0 if pred > t2 else (1 if pred > t1 else 2)


def _get_item(o: dict, item: str) -> Any:
    g = "inter" if item.startswith("I") else "anat"
    return (o.get(g) or {}).get(item)


def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2]


def fit_item(
    samples: List[Tuple[Dict[str, float], float, Optional[float]]],
    feat_names: List[str],
    levels: int,
    *,
    lam: float = 2.0,
) -> Optional[dict]:
    """samples: (feats, yH, yV|None)"""
    # drop feats with <50% coverage
    keep = []
    for f in feat_names:
        cov = sum(1 for fe, _, _ in samples if f in fe) / max(1, len(samples))
        if cov >= 0.4:
            keep.append(f)
    if not keep:
        return None
    med = {
        f: _median([fe[f] for fe, _, _ in samples if f in fe]) or 0.0 for f in keep
    }
    X, y = [], []
    for fe, yh, _ in samples:
        X.append([fe.get(f, med[f]) for f in keep])
        y.append(yh)
    Xs, means, stds = _standardize(X)
    w = _ridge_fit(Xs, y, lam=lam)
    pred = [_ridge_pred(w, x) for x in Xs]
    cuts, rho_tr = _fit_cuts(pred, y, levels)
    yhat = [apply_cuts(p, cuts) for p in pred]
    return {
        "features": keep,
        "medians": med,
        "means": means,
        "stds": stds,
        "weights": w,  # len = n_feat+1 bias last
        "cuts": cuts,
        "lam": lam,
        "rho_train": rho_tr,
        "agree_train": sum(a == b for a, b in zip(yhat, y)) / len(y),
        "n": len(y),
        "dist_H": dict(Counter(int(v) for v in y)),
        "dist_M": dict(Counter(yhat)),
    }


def predict_item(feats: Dict[str, float], model: dict) -> Optional[int]:
    names = model["features"]
    med = model["medians"]
    means = model["means"]
    stds = model["stds"]
    w = model["weights"]
    x = []
    for j, f in enumerate(names):
        raw = feats.get(f, med[f])
        sd = stds[j] if stds[j] > 1e-12 else 1.0
        x.append((raw - means[j]) / sd)
    pred = _ridge_pred(w, x)
    return apply_cuts(pred, model["cuts"])


def cv_item(
    samples: List[Tuple[Dict[str, float], float, Optional[float]]],
    feat_names: List[str],
    levels: int,
    *,
    folds: int = 5,
    seed: int = 0,
) -> dict:
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    fs = max(1, len(samples) // folds)
    rho_m, rho_v, agree_m = [], [], []
    for fi in range(folds):
        te = set(idx[fi * fs : (fi + 1) * fs] if fi < folds - 1 else idx[fi * fs :])
        tr = [samples[i] for i in range(len(samples)) if i not in te]
        te_s = [samples[i] for i in range(len(samples)) if i in te]
        if len(tr) < 20 or len(te_s) < 5:
            continue
        model = fit_item(tr, feat_names, levels)
        if not model:
            continue
        yh, ym, yv_h, yv = [], [], [], []
        for fe, yH, yV in te_s:
            pred = predict_item(fe, model)
            if pred is None:
                continue
            yh.append(yH)
            ym.append(pred)
            if yV is not None:
                yv_h.append(yH)
                yv.append(yV)
        rho_m.append(spearman(ym, yh))
        agree_m.append(sum(a == b for a, b in zip(ym, yh)) / len(yh))
        if len(yv) >= 5:
            rho_v.append(spearman(yv, yv_h))
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    return {
        "rho_M_cv": avg(rho_m),
        "rho_V_cv": avg(rho_v),
        "agree_M_cv": avg(agree_m),
        "winner_cv": (
            "M"
            if (avg(rho_m) or -9) > (avg(rho_v) or -9)
            else ("V" if avg(rho_v) is not None else "—")
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pack", default="")
    ap.add_argument("--ann", default="ann_01")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    pack = pack_root(args.pack) if args.pack else pack_root()
    hc = pack / "judgments" / "human_consistency"
    split = json.loads((hc / "_split.json").read_text(encoding="utf-8"))
    rows = []
    for rs in (split.get("splits") or {}).values():
        for r in rs:
            d = dict(r)
            d["key"] = d.get("key") or pair_key(d["sample_id"], d["model_id"])
            rows.append(d)

    H: Dict[str, dict] = {}
    for p in (hc / "human" / args.ann).glob("*.json"):
        o = json.loads(p.read_text(encoding="utf-8"))
        H[o.get("key") or pair_key(o["sample_id"], o["model_id"])] = o
    V: Dict[str, dict] = {}
    vroot = hc / "checklist_vlm" / "gpt-5.5"
    for p in vroot.rglob("*.json"):
        if p.name.startswith("_"):
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        V[o.get("key") or pair_key(o["sample_id"], o["model_id"])] = o

    # load mesh feats
    keyed: Dict[str, Dict[str, float]] = {}
    for r in rows:
        mp = pack / "judgments" / "mesh_v3" / r["model_id"] / f"{r['sample_id']}.json"
        if not mp.is_file():
            continue
        keyed[r["key"]] = extract_features(json.loads(mp.read_text(encoding="utf-8")))

    items = list(ITEM_FEATS.keys())
    calib: Dict[str, Any] = {
        "version": "item_map_v6",
        "ann": args.ann,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "note": "Every title ridge returns+fractional cutting;Ir/A5 No more blind spots (if the feature is available)",
        "items": {},
        "cv": {},
    }

    print("| item | rho_M_cv | rho_V | agree_M_cv | win | rho_M_train |")
    print("|------|----------|-------|------------|-----|-------------|")
    wins = 0
    for item in items:
        levels = 3 if item in ORD3 else 2
        samples: List[Tuple[Dict[str, float], float, Optional[float]]] = []
        for r in rows:
            k = r["key"]
            if k not in keyed or k not in H:
                continue
            yH = _get_item(H[k], item)
            if yH is None or yH == "U":
                continue
            yV = _get_item(V.get(k) or {}, item)
            yV_f = None if yV is None or yV == "U" else float(yV)
            samples.append((keyed[k], float(yH), yV_f))
        if len(samples) < 30:
            print(f"| {item} | skip n={len(samples)} |")
            continue
        # select CV superior ρ(H,M) highest lam
        best_lam, best_cv = 2.0, None
        for lam in (0.5, 1.0, 2.0, 4.0, 8.0):
            # temporarily patch fit via wrapping: re-run cv with different lam by
            # fitting inside a local loop
            pass
        # manual CV sweep
        rng = random.Random(args.seed)
        idx = list(range(len(samples)))
        rng.shuffle(idx)
        folds = 5
        fs = max(1, len(samples) // folds)

        def cv_for_lam(lam: float) -> dict:
            rho_m, rho_v, agree_m = [], [], []
            for fi in range(folds):
                te = set(
                    idx[fi * fs : (fi + 1) * fs]
                    if fi < folds - 1
                    else idx[fi * fs :]
                )
                tr = [samples[i] for i in range(len(samples)) if i not in te]
                te_s = [samples[i] for i in range(len(samples)) if i in te]
                if len(tr) < 20 or len(te_s) < 5:
                    continue
                model_f = fit_item(tr, ITEM_FEATS[item], levels, lam=lam)
                if not model_f:
                    continue
                yh, ym, yvh, yv = [], [], [], []
                for fe, yH, yV in te_s:
                    p = predict_item(fe, model_f)
                    if p is None:
                        continue
                    yh.append(yH)
                    ym.append(p)
                    if yV is not None:
                        yvh.append(yH)
                        yv.append(yV)
                rho_m.append(spearman(ym, yh))
                agree_m.append(sum(a == b for a, b in zip(ym, yh)) / len(yh))
                if len(yv) >= 5:
                    rho_v.append(spearman(yv, yvh))

            def avg(xs):
                xs = [x for x in xs if x is not None]
                return sum(xs) / len(xs) if xs else None

            return {
                "rho_M_cv": avg(rho_m),
                "rho_V_cv": avg(rho_v),
                "agree_M_cv": avg(agree_m),
                "winner_cv": (
                    "M"
                    if (avg(rho_m) or -9) > (avg(rho_v) or -9)
                    else ("V" if avg(rho_v) is not None else "—")
                ),
                "lam": lam,
            }

        best_cv = None
        for lam in (0.3, 0.7, 1.5, 3.0, 6.0, 12.0):
            cur = cv_for_lam(lam)
            if best_cv is None or (cur["rho_M_cv"] or -9) > (best_cv["rho_M_cv"] or -9):
                best_cv = cur
                best_lam = lam
        cv = best_cv or cv_item(samples, ITEM_FEATS[item], levels, seed=args.seed)
        model = fit_item(samples, ITEM_FEATS[item], levels, lam=best_lam)
        if not model:
            continue
        # also train rho vs V on full for reference
        yhat, yh, yv, yvh = [], [], [], []
        for fe, yH, yV in samples:
            p = predict_item(fe, model)
            if p is None:
                continue
            yhat.append(p)
            yh.append(yH)
            if yV is not None:
                yv.append(yV)
                yvh.append(yH)
        rho_v_full = spearman(yv, yvh)
        win = cv.get("winner_cv")
        if win == "M":
            wins += 1
        calib["items"][item] = model
        calib["cv"][item] = {**cv, "rho_V_full": rho_v_full}
        print(
            f"| {item} | {cv.get('rho_M_cv'):.3f} | {cv.get('rho_V_cv') or float('nan'):.3f} | "
            f"{cv.get('agree_M_cv'):.3f} | {win} | {model['rho_train']:.3f} |"
        )

    calib["n_items_M_wins_cv"] = wins
    calib["n_items"] = len(calib["items"])
    out = hc / "_item_map_calib.json"
    atomic_write_json(out, calib)
    print(f"\nwrote {out}")
    print(f"CV wins: {wins}/{len(calib['items'])}")
    if wins < len(calib["items"]):
        print(
            "NOTE: Some questions (especially I1) The geometric signal is weakly correlated with the human mark,CV may not fully exceed V；"
            "Fit with the strongest available features. Deployment uses fully calibrated model."
        )


if __name__ == "__main__":
    main()
