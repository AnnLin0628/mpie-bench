#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5b: Reference pool extraction —— Sample frames evenly from the full timeline of each video, Layoff people will be put into the database as reference candidates.

motivation: Stage 4 The selected keyframes are all "interaction peaks", In the short video, they are crowded together in time and have similar postures,
Selecting reference images only from these frames will result in reference≈Target(copy-paste Hidden danger). The reference picture should be from**off-peak
various moments**(Especially at the beginning of the video/At the end, when the two separate and the single person is clear). This script cuts out the people in these frames,
Register as keyframes(selected=2, reference pool tag) + persons, Share the same identity cluster as the peak frame,
Stage 6 You can use them as**Various reference candidates**。

usage(and Stage5 extract Same environment, Each shard is independent):
  python ref_pool_extract.py --db <db> --out <crops> --frames <frames_dir> \
        --n-ref 10 [--shard i --n-shards 4]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.weights import local_or_name, load_dinov2  # noqa: E402
from common.manifest import connect, upsert, rows  # noqa: E402


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def frame_sep(boxes):
    """Frame separation: Single 0.95; multiple people = 1 - Maximum two IoU(The less overlap, the higher=more like interaction trough)。"""
    if len(boxes) == 1:
        return 0.95
    m = max(_iou(boxes[i], boxes[j]) for i in range(len(boxes))
            for j in range(i + 1, len(boxes)))
    return 1.0 - m


def seg_persons(det, img, min_h=128, pad=0.04):
    """YOLO-seg → [(tight frame, full image maskbool), ...]. tight frame=Mask bounding rectangle(Go to white space, Still a rectangle)。"""
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
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, int(x1 - bw * pad)); y1 = max(0, int(y1 - bh * pad))
        x2 = min(W, int(x2 + bw * pad)); y2 = min(H, int(y2 + bh * pad))
        if (y2 - y1) >= min_h:
            out.append(([x1, y1, x2, y2], mm))
    return out


def crop_clean(img, persons, idx, reject_thr=0.02):
    """judgment persons[idx] Single person picture: Prioritize [zoom out] to avoid others(Half body→The avatar is gradually tightened, Go to others with fewer pixels
    One side), No longer dependent on inpaint Erase——inpaint Distorted smear artifacts will be compensated for large areas of other people's areas., Image quality ratio
    The blurred background is not that good. Tighten it until the avatar is still unavoidable and give up this candidate directly.(scan/keep enough candidates, Not missing this one),
    Only a very small amount of edge pixels remain after tightening(<2%)Only used when inpaint Do the finishing touches. """
    (x1, y1, x2, y2), tgt = persons[idx]
    w, h = x2 - x1, y2 - y1

    def region_overlap(bx1, by1, bx2, by2):
        oth = np.zeros((by2 - by1, bx2 - bx1), bool)
        for j, (_, mj) in enumerate(persons):
            if j != idx:
                oth |= mj[by1:by2, bx1:bx2]
        oth &= ~tgt[by1:by2, bx1:bx2]
        return oth, (oth.mean() if oth.size else 0.0)

    cands = [(x1, y1, x2, y2)]
    oth0, r0 = region_overlap(x1, y1, x2, y2)
    if r0 > reject_thr:
        left_mass = oth0[:, :w // 2].sum()
        right_mass = oth0[:, w // 2:].sum()
        nx1, nx2 = (x1, x1 + int(w * 0.65)) if left_mass <= right_mass else (x1 + int(w * 0.35), x2)
        cands.append((nx1, y1, nx2, y1 + max(1, int(h * 0.55))))   # Half body, Toward the clean side
        cands.append((nx1, y1, nx2, y1 + max(1, int(h * 0.32))))   # avatar

    for bx1, by1, bx2, by2 in cands:                # from wide to narrow, Take the first one that is clean enough(Prioritize retaining more information)
        oth, r = region_overlap(bx1, by1, bx2, by2)
        if r <= reject_thr:
            crop = img[by1:by2, bx1:bx2].copy()
            if oth.any():                            # Very little edge residue inpaint ending
                em = cv2.dilate((oth * 255).astype("uint8"), np.ones((5, 5), np.uint8), iterations=2)
                crop = cv2.inpaint(crop, em, 3, cv2.INPAINT_TELEA)
            return crop
    return None                                       # Even if you shrink to your avatar, you still can’t avoid others → give up, Give it to other candidate frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", required=True, help="Stage2 Frame directory(Contains <vid>/ subdirectory)")
    ap.add_argument("--scan", type=int, default=40, help="Number of candidate frames per video scan(Calculate separation)")
    ap.add_argument("--keep", type=int, default=12, help="Retain the highest degree of separation(interaction trough)Several frames serve as the reference pool. "
                    "The coverage should be wider: The two may not show their faces at the same time, The more frames there are, the better chance each person has of getting a clear shot individually.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    conn = connect(args.db)

    det = YOLO(local_or_name("yolov8m-seg.pt"))          # segmentation model: Mask counts as tight frame
    face = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider"])
    face.prepare(ctx_id=0, det_size=(640, 640))
    dino = load_dinov2().eval().cuda()
    out = Path(args.out); (out / "persons").mkdir(parents=True, exist_ok=True)
    emb_dir = out / "embeddings"; emb_dir.mkdir(exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

    vids = [r["video_id"] for r in rows(conn,
            "SELECT DISTINCT video_id FROM keyframes WHERE selected=1")]
    vids = [v for i, v in enumerate(sorted(vids)) if i % args.n_shards == args.shard]
    n_frames_done = 0
    for vid in vids:
        fdir = Path(args.frames) / vid
        frames = sorted(fdir.glob("*.jpg"))
        if len(frames) < 2:
            continue
        # 1) Scan a batch of candidate frames, Detect people + Calculate separation(Caching detection results to avoid duplication YOLO)
        scan_idx = np.linspace(0, len(frames) - 1, min(args.scan, len(frames))).round().astype(int)
        scored = []
        for fi in sorted(set(scan_idx.tolist())):
            fpath = frames[fi]
            img = cv2.imread(str(fpath))
            if img is None:
                continue
            persons = seg_persons(det, img)                  # [(tight frame, mask), ...]
            if not persons:
                continue
            scored.append((frame_sep([p[0] for p in persons]), fpath, img, persons))
        # 2) Take the highest degree of separation(interaction trough=The two of them are most separated)of keep frame
        scored.sort(key=lambda x: -x[0])
        for sep, fpath, img, persons in scored[:args.keep]:
            frame_idx = int(fpath.stem)
            kf_id = f"{vid}_r{frame_idx:07d}"          # r = ref-pool mark
            upsert(conn, "keyframes", {"kf_id": kf_id, "video_id": vid, "shot_id": None,
                                       "frame_idx": frame_idx, "n_person": len(persons),
                                       "density_score": float(sep), "density_level": None,
                                       "sharpness": 0.0, "frame_path": str(fpath),
                                       "selected": 2})    # selected=2: reference pool, non-target
            for pi in range(len(persons)):
                box = persons[pi][0]
                x1, y1, x2, y2 = [int(v) for v in box]
                crop = crop_clean(img, persons, pi)          # Tight frame(Half body→The avatar tightens up step by step to avoid others)
                if crop is None or crop.size == 0 or crop.shape[0] < 64:
                    continue                                  # Even if you shrink to the avatar, you still can’t avoid it → abandon this candidate
                track_id = f"{kf_id}_p{pi}"
                cv2.imwrite(str(out / "persons" / f"{track_id}.jpg"), crop)
                faces = face.get(crop)
                femb = faces[0].normed_embedding if faces else None
                t = cv2.resize(crop, (224, 224))[:, :, ::-1].copy()
                t = torch.from_numpy(t).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255
                t = (t - mean) / std
                with torch.no_grad():
                    bemb = dino(t)[0].cpu().numpy()
                fe = emb_dir / f"{track_id}.npz"
                np.savez(fe, face=femb if femb is not None else np.array([]), body=bemb)
                upsert(conn, "persons", {"track_id": track_id, "video_id": vid,
                                         "identity_id": None, "bbox_json": [x1, y1, x2, y2],
                                         "n_frames": 1, "face_emb_path": str(fe),
                                         "body_emb_path": str(fe)})
            n_frames_done += 1
        conn.commit()
    print(f"ref-pool done: {len(vids)} videos, {n_frames_done} ref frames")


if __name__ == "__main__":
    main()
