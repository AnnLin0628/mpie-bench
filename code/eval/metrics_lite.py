#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPIE-Bench Lightweight objective indicators (division in six dimensions) mesh interspersed with VLM judge outer part).

  count_gate      Number of people gate: Number of people tested == Expected number of people (hard door)
  id_match        Hungarian-ArcFace Multiple identity matching (MultiHuman-Testbench formula)
                  + M_CP copy paste penalty (WithAnyone Ideas: The generated face is too close to the reference face at the pixel level=cheeky)
  joint_limits    Joint angle legality: RTMPose Key points → elbow/knees/Is the neck angle out of bounds?

Depends on pipeline same environment(ultralytics/rtmlib/insightface). Scripts evaluated as libraries import,
Also possible CLI Single picture debugging: python metrics_lite.py --img gen.png --refs a.jpg b.jpg --n 2
"""
import argparse
import itertools

import os
from pathlib import Path as _P
import cv2
import numpy as np


def _w(name):
    d = os.environ.get("MPIE_WEIGHTS_DIR", "")
    if d and (_P(d) / name).exists():
        return str(_P(d) / name)
    return name


# ---------- lazy singleton ----------
_DET = _WB = _FACE = None


def _det():
    global _DET
    if _DET is None:
        from ultralytics import YOLO
        _DET = YOLO(_w("yolov8m.pt"))
    return _DET


def _wb():
    global _WB
    if _WB is None:
        from rtmlib import Wholebody
        _WB = Wholebody(backend="onnxruntime", device="cuda")
    return _WB


def _face():
    global _FACE
    if _FACE is None:
        from insightface.app import FaceAnalysis
        _FACE = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider"])
        _FACE.prepare(ctx_id=0, det_size=(640, 640))
    return _FACE


# ---------- 1. Number of people gate ----------
def count_gate(img: np.ndarray, expected: int) -> dict:
    r = _det()(img, classes=[0], conf=0.5, verbose=False)[0]
    n = len(r.boxes) if r.boxes is not None else 0
    return {"n_detected": n, "n_expected": expected, "pass": n == expected}


# ---------- 2. Hungarian-ArcFace + M_CP ----------
def id_match(img: np.ndarray, ref_imgs: list) -> dict:
    """ref_imgs: One reference picture for each character(BGR). Returns player-by-player match scores M_CP。"""
    from scipy.optimize import linear_sum_assignment
    gen_faces = _face().get(img)
    ref_embs = []
    for r in ref_imgs:
        fs = _face().get(r)
        ref_embs.append(fs[0].normed_embedding if fs else None)
    if not gen_faces or all(e is None for e in ref_embs):
        return {"per_ref_sim": [0.0] * len(ref_imgs), "mean_sim": 0.0, "m_cp": 0.0,
                "note": "no face detected"}
    G = np.stack([f.normed_embedding for f in gen_faces])
    C = np.zeros((len(ref_embs), len(G)))
    for i, e in enumerate(ref_embs):
        C[i] = -(G @ e) if e is not None else 0.0
    ri, gi = linear_sum_assignment(C)
    sims = [0.0] * len(ref_embs)
    for a, b in zip(ri, gi):
        sims[a] = float(-C[a, b])
    # M_CP: Generate face crop with reference face crop pixel-level similarity of(too high=copy paste)。
    # Use low frequency DCT Hash approximation; Threshold Behavior Vs. WithAnyone The original implementation reported consecutive values ​​before alignment.
    m_cp = 0.0
    for a, b in zip(ri, gi):
        if ref_embs[a] is None:
            continue
        gf = gen_faces[b].bbox.astype(int)
        crop = img[max(0, gf[1]):gf[3], max(0, gf[0]):gf[2]]
        rfs = _face().get(ref_imgs[a])
        if crop.size == 0 or not rfs:
            continue
        rb = rfs[0].bbox.astype(int)
        rcrop = ref_imgs[a][max(0, rb[1]):rb[3], max(0, rb[0]):rb[2]]
        if rcrop.size == 0:
            continue
        h1 = cv2.dct(np.float32(cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32))))[:8, :8]
        h2 = cv2.dct(np.float32(cv2.resize(cv2.cvtColor(rcrop, cv2.COLOR_BGR2GRAY), (32, 32))))[:8, :8]
        m_cp = max(m_cp, float(np.corrcoef(h1.ravel(), h2.ravel())[0, 1]))
    return {"per_ref_sim": [round(s, 4) for s in sims],
            "mean_sim": round(float(np.mean(sims)), 4), "m_cp": round(m_cp, 4)}


# ---------- 3. joint limit ----------
# (a,b,c): by b is the angle between the vertices; The limit is based on the loose bounds of common anatomical knowledge, Only grasp the obvious anti-joints
JOINT_RULES = [
    (5, 7, 9, 5, 180),    # left elbow
    (6, 8, 10, 5, 180),   # right elbow
    (11, 13, 15, 15, 180),  # left knee
    (12, 14, 16, 15, 180),  # right knee
]


def joint_limits(img: np.ndarray) -> dict:
    kpts, scores = _wb()(img)
    viol, checked = 0, 0
    for p, s in zip(kpts, scores):
        for a, b, c, lo, hi in JOINT_RULES:
            if min(s[a], s[b], s[c]) < 0.35:
                continue
            v1, v2 = p[a] - p[b], p[c] - p[b]
            ang = np.degrees(np.arccos(np.clip(
                v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8), -1, 1)))
            checked += 1
            if not (lo <= ang <= hi):
                viol += 1
    return {"n_person": len(kpts), "checked": checked, "violations": viol,
            "score": round(1 - viol / checked, 4) if checked else None}


def evaluate(img_path: str, ref_paths: list, expected_n: int) -> dict:
    img = cv2.imread(img_path)
    refs = [cv2.imread(p) for p in ref_paths]
    return {"count": count_gate(img, expected_n),
            "id": id_match(img, refs),
            "joints": joint_limits(img)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=2)
    args = ap.parse_args()
    import json
    print(json.dumps(evaluate(args.img, args.refs, args.n), ensure_ascii=False, indent=2))
