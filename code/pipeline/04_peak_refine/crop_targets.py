#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4b: Target image cropping. selected peak frame(selected=1)Crop to "Interactive Subject Enclosed Area", Erase non-interactive passersby.

with reference picture(Single)different: The target map retains [those people interacting](2-3people), Only passers-by who do not participate in the interaction are included in the screen
use inpaint erase, And clip to the enclosing rectangle of the interaction group + Leave blank, Enlarge interaction and remove unnecessary background.
interactive group = Principal connected components of overlapping graphs of character frames(the tightest group of people)。<2Human frames are judged as invalid targets(selected=0)。

usage: python crop_targets.py --db <db> --out <targets_dir> [--pad 0.12]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.weights import local_or_name  # noqa: E402
from common.manifest import connect, rows  # noqa: E402


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def seg_persons(det, img, min_h=100):
    r = det(img, classes=[0], conf=0.5, verbose=False)[0]
    if r.masks is None or r.boxes is None:
        return []
    H, W = img.shape[:2]
    out = []
    for m in r.masks.data.cpu().numpy():
        mm = cv2.resize(m.astype("float32"), (W, H)) > 0.5
        ys, xs = np.where(mm)
        if len(xs) < 50:
            continue
        box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        if box[3] - box[1] >= min_h:
            out.append((box, mm))
    return out


def interaction_cluster(persons, iou_thr=0.04):
    """Character frame overlay image → Connected components containing "maximally overlapping pairs"(=interactive group). Returns the index collection. """
    n = len(persons)
    boxes = [p[0] for p in persons]
    best, bi, bj = 0.0, 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            v = _iou(boxes[i], boxes[j])
            if v > best:
                best, bi, bj = v, i, j
    cluster = {bi, bj}
    changed = True
    while changed:                                  # Pass extension: Those that overlap with anyone in the group are also merged into
        changed = False
        for k in range(n):
            if k in cluster:
                continue
            if any(_iou(boxes[k], boxes[c]) > iou_thr for c in cluster):
                cluster.add(k); changed = True
    return cluster, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pad", type=float, default=0.12, help="Extend the white space outside the interactive group enclosing frame")
    args = ap.parse_args()
    conn = connect(args.db)
    det = YOLO(local_or_name("yolov8m-seg.pt"))
    tgt_dir = Path(args.out); tgt_dir.mkdir(parents=True, exist_ok=True)

    kfs = rows(conn, "SELECT kf_id, frame_path FROM keyframes WHERE selected=1")
    n_ok = n_drop = 0
    for kf in kfs:
        img = cv2.imread(kf["frame_path"])
        persons = seg_persons(det, img) if img is not None else []
        if len(persons) < 2:                        # interact at least 2 people, Otherwise it is not a valid target
            conn.execute("UPDATE keyframes SET selected=0 WHERE kf_id=?", (kf["kf_id"],))
            n_drop += 1
            continue
        cluster, _ = interaction_cluster(persons)
        H, W = img.shape[:2]
        # bounding box(interactive group)
        cb = [persons[c][0] for c in cluster]
        x1 = min(b[0] for b in cb); y1 = min(b[1] for b in cb)
        x2 = max(b[2] for b in cb); y2 = max(b[3] for b in cb)
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw*args.pad)); y1 = max(0, int(y1 - bh*args.pad))
        x2 = min(W, int(x2 + bw*args.pad)); y2 = min(H, int(y2 + bh*args.pad))
        crop = img[y1:y2, x1:x2].copy()
        # Erase passers-by in non-interactive groups(exist crop inner part)
        erase = np.zeros(crop.shape[:2], bool)
        for k, (_, mk) in enumerate(persons):
            if k not in cluster:
                erase |= mk[y1:y2, x1:x2]
        if erase.any():
            em = cv2.dilate((erase*255).astype("uint8"), np.ones((5, 5), np.uint8), iterations=2)
            crop = cv2.inpaint(crop, em, 3, cv2.INPAINT_TELEA)
        tp = tgt_dir / f"{kf['kf_id']}.jpg"
        cv2.imwrite(str(tp), crop)
        conn.execute("UPDATE keyframes SET frame_path=?, n_person=? WHERE kf_id=?",
                     (str(tp), len(cluster), kf["kf_id"]))
        n_ok += 1
    conn.commit()
    print(f"target crop: {n_ok} Crop, {n_drop} throw away(insufficient2people)")


if __name__ == "__main__":
    main()
