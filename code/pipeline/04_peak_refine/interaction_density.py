#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 4 (Stage B settlement): Fine scoring of interaction density within the candidate window + Peak frame selection.

InteractionDensity(t,i,j) = w1*bbox_IoU + w2*PoseProximity(Wrist weighted)
                          + w3*FlowMag_norm + w4*OcclusionRatio
The four items are weighted after normalization by within-video percentile. Weight default [0.35,0.30,0.20,0.15]，
treat Hi4D Use the vertex contact annotation once you have it. calibrate_weights.py The logistic regression results are covered.
clarity(Laplacian variance)Just do tie-break With the quality gate, density points are not entered.

Peak detection: Fractional sequence sliding smoothing + scipy find_peaks, For each peak, take the clearest frame within the window;
same interaction event(Interval between adjacent peaks < min_gap Second)Only keep the highest peak → selected=1。

usage: python interaction_density.py --db ~/mpie_data/manifests/mpie.db \
        [--weights 0.35,0.30,0.20,0.15] [--top-per-video 40] [--shard 0 --n-shards 4]
"""
import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from scipy.signal import find_peaks
from rtmlib import Wholebody

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, rows  # noqa: E402

# COCO-wholebody wrist/elbow/shoulder index(The trunk and upper limbs are the main contact carriers, The wrist has the highest weight)
WRIST_W = {9: 3.0, 10: 3.0, 7: 1.5, 8: 1.5, 5: 1.0, 6: 1.0}


def pose_proximity(kpts_a: np.ndarray, kpts_b: np.ndarray, scale: float) -> float:
    """Minimum weighted distance between two people's key points → normalized proximity(The closer, the higher)。scale=two people bbox Diagonal mean. """
    best = 1e9
    for i, wi in WRIST_W.items():
        for j, wj in WRIST_W.items():
            if kpts_a[i, 2] < 0.3 or kpts_b[j, 2] < 0.3:
                continue
            d = np.linalg.norm(kpts_a[i, :2] - kpts_b[j, :2]) / (wi * wj) ** 0.5
            best = min(best, d)
    if best > 1e8:
        return 0.0
    return float(np.exp(-best / (scale * 0.15 + 1e-6)))


def occlusion_ratio(boxes: np.ndarray) -> float:
    """figure bbox The maximum proportion of the overlapping area of ​​two pairs to the area of ​​the smaller frame(Occlusion agent)。"""
    best = 0.0
    for a, b in itertools.combinations(boxes, 2):
        x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:4], b[2:4])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
        best = max(best, inter / (smaller + 1e-6))
    return best


def sharpness(img) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def pct_norm(x: np.ndarray) -> np.ndarray:
    if len(x) < 2 or x.max() == x.min():
        return np.zeros_like(x)
    order = x.argsort().argsort()
    return order / (len(x) - 1)


class FlowMag:
    def __init__(self, device="cuda"):
        self.model = torchvision.models.optical_flow.raft_large(weights="DEFAULT").eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, img1, img2) -> float:
        def prep(im):
            im = cv2.resize(im, (480, 272))  # RAFT The required width and height are 8 multiples of
            t = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1).float() / 127.5 - 1
            return t.unsqueeze(0).to(self.device)
        flow = self.model(prep(img1), prep(img2))[-1][0]
        return float(flow.norm(dim=0).mean())


def refine_video(vid: str, cands: list, wb: Wholebody, flow: FlowMag, conn,
                 weights, top_per_video: int, min_gap_frames: int):
    feats = []
    prev_img = None
    for c in sorted(cands, key=lambda r: r["frame_idx"]):
        img = cv2.imread(c["frame_path"])
        if img is None:
            continue
        kpts, scores = wb(img)
        kp = np.concatenate([kpts, scores[..., None]], axis=-1) if len(kpts) else np.empty((0, 133, 3))
        # Get the maximum proximity between two people
        prox, boxes = 0.0, []
        for p in kp:
            xy = p[p[:, 2] > 0.3][:, :2]
            if len(xy) < 4:
                continue
            boxes.append([*xy.min(0), *xy.max(0)])
        boxes = np.array(boxes) if boxes else np.empty((0, 4))
        if len(kp) >= 2 and len(boxes) >= 2:
            diag = np.mean([np.linalg.norm(b[2:] - b[:2]) for b in boxes])
            for a, b in itertools.combinations(range(len(kp)), 2):
                prox = max(prox, pose_proximity(kp[a], kp[b], diag))
        fm = flow(prev_img, img) if prev_img is not None else 0.0
        prev_img = img
        feats.append({"kf_id": c["kf_id"], "frame_idx": c["frame_idx"],
                      "iou": c["density_score"],  # rough sweep approximation bbox_iou item
                      "prox": prox, "flow": fm,
                      "occ": occlusion_ratio(boxes) if len(boxes) >= 2 else 0.0,
                      "sharp": sharpness(img)})
    if not feats:
        return 0
    F = {k: pct_norm(np.array([f[k] for f in feats])) for k in ("iou", "prox", "flow", "occ")}
    score = weights[0]*F["iou"] + weights[1]*F["prox"] + weights[2]*F["flow"] + weights[3]*F["occ"]

    peaks, _ = find_peaks(np.convolve(score, np.ones(3)/3, "same"),
                          distance=max(1, min_gap_frames), height=np.percentile(score, 60))
    chosen = sorted(peaks, key=lambda i: -score[i])[:top_per_video]
    for rank, i in enumerate(chosen):
        f = feats[i]
        conn.execute("UPDATE keyframes SET density_score=?, sharpness=?, selected=1 WHERE kf_id=?",
                     (round(float(score[i]), 4), round(f["sharp"], 1), f["kf_id"]))
    conn.commit()
    return len(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--weights", default="0.35,0.30,0.20,0.15")
    ap.add_argument("--top-per-video", type=int, default=40)
    ap.add_argument("--min-gap-sec", type=float, default=2.0)
    ap.add_argument("--scan-fps", type=float, default=4.0, help="Stage2 frame rate, Convert peak interval")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    weights = [float(x) for x in args.weights.split(",")]

    conn = connect(args.db)
    wb = Wholebody(backend="onnxruntime", device="cuda")
    flow = FlowMag()
    vids = [r["video_id"] for r in rows(conn, "SELECT DISTINCT video_id FROM keyframes WHERE selected=0")]
    vids = [v for i, v in enumerate(sorted(vids)) if i % args.n_shards == args.shard]
    for vid in vids:
        # can only be Stage3 candidate(selected=0)Select peak value; Never touch the reference pool(selected=2,
        # Trough frame, Stage5b output)——Otherwise, selected clean single-player trough frames will be mistakenly promoted to target frames.,
        # Also flush it out of the reference pool(2 is overridden into 1)。
        cands = rows(conn, "SELECT * FROM keyframes WHERE video_id=? AND selected=0", (vid,))
        n = refine_video(vid, cands, wb, flow, conn, weights,
                         args.top_per_video, int(args.min_gap_sec * args.scan_fps))
        print(f"{vid}: {len(cands)} cands -> {n} selected", flush=True)


if __name__ == "__main__":
    main()
