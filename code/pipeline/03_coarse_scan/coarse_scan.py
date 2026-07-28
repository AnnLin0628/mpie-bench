#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 3 (Stage A Rough sweep): YOLO Human detection + bbox IoU + Frame difference motion energy.

right Stage 2 The extracted low frame rate frames are calculated frame by frame as three cheap signals., sieve out"≥2person and there are signs of contact"
candidate time window, Write keyframes surface(Candidate state, selected=0). Actuarial in Stage 4。

usage: python coarse_scan.py --frames ~/mpie_data/frames --db ~/mpie_data/manifests/mpie.db \
        [--shard 0 --n-shards 4] [--min-person 2] [--iou-gate 0.01]
"""
import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.weights import local_or_name  # noqa: E402
from common.manifest import connect, upsert, rows  # noqa: E402


def pair_iou(boxes: np.ndarray) -> float:
    """Everyone in pairs bbox IoU the maximum value of(0=No overlap at all)。"""
    best = 0.0
    for a, b in itertools.combinations(boxes, 2):
        x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:4], b[2:4])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        if inter <= 0:
            continue
        ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
        best = max(best, inter / ua)
    return best


def motion_energy(prev_gray, gray) -> float:
    if prev_gray is None:
        return 0.0
    d = cv2.absdiff(prev_gray, gray)
    return float(d.mean()) / 255.0


def scan_video(model: YOLO, vid: str, frame_dir: Path, conn, min_person: int, iou_gate: float):
    frames = sorted(frame_dir.glob("*.jpg"))
    prev_gray, n_cand = None, 0
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        gray = cv2.cvtColor(cv2.resize(img, (320, 180)), cv2.COLOR_BGR2GRAY)
        me = motion_energy(prev_gray, gray)
        prev_gray = gray

        r = model(img, classes=[0], conf=0.4, verbose=False)[0]  # class 0 = person
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.empty((0, 4))
        n = len(boxes)
        if n < min_person:
            continue
        iou = pair_iou(boxes)
        if iou < iou_gate and me < 0.02:   # No overlap and almost stationary → jump over
            continue
        frame_idx = int(fp.stem)
        # Rough scan score = IoU Simple combination with sports energy, Only used to rank candidates; Actuarial points in Stage 4 cover
        upsert(conn, "keyframes", {
            "kf_id": f"{vid}_f{frame_idx:07d}", "video_id": vid, "shot_id": None,
            "frame_idx": frame_idx, "n_person": n,
            "density_score": float(round(0.7 * iou + 0.3 * me, 4)), "density_level": None,
            "sharpness": None, "frame_path": str(fp), "selected": 0})
        n_cand += 1
    conn.commit()
    print(f"{vid}: {len(frames)} frames -> {n_cand} candidates", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--min-person", type=int, default=2)
    ap.add_argument("--iou-gate", type=float, default=0.01)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    conn = connect(args.db)
    model = YOLO(local_or_name("yolov8m.pt"))
    vids = sorted(p.name for p in Path(args.frames).iterdir() if p.is_dir())
    vids = [v for i, v in enumerate(vids) if i % args.n_shards == args.shard]
    done = {r["video_id"] for r in rows(conn, "SELECT DISTINCT video_id FROM keyframes")}
    for vid in vids:
        if vid in done:
            print(f"{vid}: skip (scanned)", flush=True)
            continue
        scan_video(model, vid, Path(args.frames) / vid, conn, args.min_person, args.iou_gate)


if __name__ == "__main__":
    main()
