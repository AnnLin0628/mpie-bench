#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harmony4D full extract: peak target frames + per-identity cross-camera clean refs (all GT; no clustering/detection).

Harmony4D provides exo multi-camera frames + GT bbox (dict{aria:[x,y,w,h]}) + GT 3D poses (poses3d) +
stable GT identities (aria01/02 across frames/cameras). Therefore:
  targets = contact-density peak frames from poses3d, taking N exo cameras (selected=1)
  refs = per identity, crop with GT bbox on max pairwise-distance frames, preferring non-target cameras
         → natural cross-view diversity; source of pilot group B (anti copy-paste)
Identity is GT — no ArcFace/DINOv2/clustering. Prefer (frame, camera) where the person's 2D box
least overlaps others so crops stay single-person clean; keep sharp raw crops (no blur).

Usage: python harmony4d_extract.py --root ~/mpie_data/raw_video/harmony4d/extracted \
        --db ~/mpie_data/manifests/mpie_h4d.db --out ~/mpie_data/h4d_crops \
        --cams 4 --peaks-per-seq 6 --refs-per-identity 1 [--packages 01_hugging] [--no-blur]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, upsert  # noqa: E402
from harmony4d_ingest import (load_pose, contact_density, _pair_min_dist,  # noqa: E402
                              density_band, ACTION_MAP, action_of, find_sequences)

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None


def read_bbox(seq, cam, fid):
    """Read GT bbox → {aria: [x1,y1,x2,y2]}. Stored as [x,y,w,h]."""
    p = seq / "processed_data" / "bbox" / cam / f"{fid}.npy"
    if not p.exists():
        return {}
    d = np.load(p, allow_pickle=True).item()
    out = {}
    for k, v in d.items():
        x, y, w, h = [float(t) for t in np.asarray(v).ravel()[:4]]
        out[k] = [int(x), int(y), int(x + w), int(y + h)]
    return out


def read_poses2d(seq, cam, fid):
    """Read GT 2D joints → {aria: (45,2) pixel coords}. Used for tight body rectangles."""
    p = seq / "processed_data" / "poses2d" / cam / f"{fid}.npy"
    if not p.exists():
        return {}
    try:
        d = np.load(p, allow_pickle=True).item()
    except Exception:
        return {}
    return {k: np.asarray(v, float) for k, v in d.items()}


def kps_rect(kps_list, img_w, img_h, pad=0.10):
    """Joint set (possibly multi-person) → tight rectangle [x1,y1,x2,y2] (axis-aligned, not a contour)."""
    pts = []
    for kps in kps_list:
        k = np.asarray(kps, float)
        if k.ndim != 2 or k.shape[1] < 2:
            continue
        k = k[(k[:, 0] > 0) & (k[:, 1] > 0) & (k[:, 0] < img_w) & (k[:, 1] < img_h)]
        if len(k):
            pts.append(k[:, :2])
    if not pts:
        return None
    P = np.concatenate(pts, 0)
    x1, y1 = P.min(0); x2, y2 = P.max(0)
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - bw * pad); y1 = max(0, y1 - bh * pad)
    x2 = min(img_w, x2 + bw * pad); y2 = min(img_h, y2 + bh * pad)
    return [int(x1), int(y1), int(x2), int(y2)]


def pick_cameras(seq, n):
    exo = seq / "exo"
    cams = sorted([d.name for d in exo.iterdir() if d.is_dir()])
    if len(cams) <= n:
        return cams
    idx = np.linspace(0, len(cams) - 1, n).round().astype(int)
    return [cams[i] for i in idx]


def img_path(seq, cam, fid):
    p = seq / "exo" / cam / "images" / f"{fid}.jpg"
    return p if p.exists() else None


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def crop_bbox(img, box, pad=0.08):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * pad)); y1 = max(0, int(y1 - bh * pad))
    x2 = min(w, int(x2 + bw * pad)); y2 = min(h, int(y2 + bh * pad))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return img[y1:y2, x1:x2]


def union_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def has_frontal_face(face, crop):
    """Treat InsightFace face detection as usable (backs-to-camera fail naturally). None=skip check."""
    if face is None:
        return True
    try:
        return len(face.get(crop)) >= 1
    except Exception:
        return False


def process_seq(seq, args, conn, face):
    action = action_of(seq.name)
    itype, _ = ACTION_MAP.get(action, ("other", "C2"))
    poses = sorted((seq / "processed_data" / "poses3d").glob("*.npy"))
    if not poses:
        return 0, 0
    fids, scores, mindists, keyss = [], [], [], []
    for pp in poses:
        pose, valid, keys = load_pose(pp)
        P = pose.shape[0]
        md = min((_pair_min_dist(pose, valid, i, j)[0] for i in range(P)
                  for j in range(i + 1, P)), default=9.9)
        if not np.isfinite(md):
            md = 9.9
        sc, _ = contact_density(pose, valid)
        fids.append(pp.stem); scores.append(sc); mindists.append(md); keyss.append(keys)
    scores = np.array(scores); mindists = np.array(mindists)
    exo_cams = [c.name for c in sorted((seq / "exo").iterdir()) if c.is_dir()]
    vid = f"h4d_{seq.name}"
    upsert(conn, "videos", {"video_id": vid, "dataset": "harmony4d", "path": str(seq),
                            "license_tier": "restricted", "fps": 0, "n_frames": len(poses),
                            "meta_json": {"action": action, "interaction_type": itype}})
    tgt_dir = Path(args.out) / "targets"; tgt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Targets: contact-peak frames; prefer nearest cam (largest subjects); crop joint hull + pad ----
    if find_peaks is not None and len(scores) > 5:
        pk, _ = find_peaks(scores, distance=max(1, len(scores) // (args.peaks_per_seq * 2)))
        if len(pk) == 0:
            pk = np.argsort(scores)[::-1][:args.peaks_per_seq]
    else:
        pk = np.argsort(scores)[::-1][:args.peaks_per_seq]
    pk = sorted(pk, key=lambda i: scores[i], reverse=True)[:args.peaks_per_seq]
    n_tgt = 0
    used_target_cams = set()
    for i in pk:
        fid = fids[i]; band = density_band(mindists[i]); ids_here = keyss[i]
        # Score cameras by min subject height; larger = nearer = higher subject fraction
        cam_rank = []
        for cam in exo_cams:
            boxes = read_bbox(seq, cam, fid)
            present = [boxes[k] for k in ids_here if k in boxes]
            if len(present) < len(ids_here):
                continue                                     # require all subjects visible
            cam_rank.append((min(b[3] - b[1] for b in present), cam, present))
        cam_rank.sort(reverse=True)
        for minh, cam, present in cam_rank[:args.cams]:
            if minh < args.min_h:
                continue                                     # drop if subjects too small/far
            img = cv2.imread(str(img_path(seq, cam, fid) or ""))
            if img is None:
                continue
            H, W = img.shape[:2]
            p2 = read_poses2d(seq, cam, fid)
            rect = kps_rect([p2[k] for k in ids_here if k in p2], W, H, pad=0.08)  # tight two-person joint rect
            crop = img[rect[1]:rect[3], rect[0]:rect[2]] if rect else crop_bbox(img, union_box(present), pad=0.05)
            if crop is None or crop.size == 0:
                continue
            kf_id = f"{vid}_{cam}_{fid}"
            tp = tgt_dir / f"{kf_id}.jpg"
            cv2.imwrite(str(tp), crop)
            used_target_cams.add(cam)
            upsert(conn, "keyframes", {"kf_id": kf_id, "video_id": vid, "shot_id": f"{vid}_{cam}",
                                       "frame_idx": int(fid), "n_person": len(ids_here),
                                       "density_score": float(scores[i]), "density_level": band,
                                       "sharpness": 0.0, "frame_path": str(tp), "selected": 1})
            for pidx, ak in enumerate(ids_here):
                upsert(conn, "persons", {"track_id": f"{kf_id}_p{pidx}", "video_id": vid,
                                         "identity_id": f"{vid}_{ak}", "bbox_json": None,
                                         "n_frames": 1, "face_emb_path": None, "body_emb_path": None})
            n_tgt += 1

    # ---- Ref pool: N candidates/actor/scene (2D non-overlap + large enough); prefer faces but soft,
    #      humans can drop unclean crops; one pool serves all targets in the scene ----
    ref_cams = [c for c in exo_cams if c not in used_target_cams] or exo_cams  # prefer non-target cams
    all_ids = sorted({k for ks in keyss for k in ks})
    sep_order = np.argsort(mindists)[::-1]                    # 3D separation large→small
    n_ref = 0
    for ak in all_ids:
        cands = []
        for i in sep_order[:args.ref_scan_frames]:
            if ak not in keyss[i]:
                continue
            fid = fids[i]
            for cam in ref_cams:
                boxes = read_bbox(seq, cam, fid)
                if ak not in boxes:
                    continue
                others = [boxes[o] for o in boxes if o != ak]
                overlap = max((_iou(boxes[ak], ob) for ob in others), default=0.0)
                bh = boxes[ak][3] - boxes[ak][1]
                if bh < args.min_h or overlap > args.max_overlap:
                    continue                                 # too small or 2D overlap with others
                cands.append((bh, i, fid, cam, boxes[ak]))
        cands.sort(reverse=True)                             # prefer largest box (nearest cam)
        # Take largest boxes, face-check, prefer frontal; at most 1 per camera (avoid view dupes)
        scored, seen_cam = [], set()
        for bh, i, fid, cam, box in cands[:args.refs_per_identity * 5]:
            if cam in seen_cam:
                continue
            img = cv2.imread(str(img_path(seq, cam, fid) or ""))
            if img is None:
                continue
            H, W = img.shape[:2]
            p2 = read_poses2d(seq, cam, fid)
            rect = kps_rect([p2[ak]], W, H, pad=0.10) if ak in p2 else None  # single-person joint rect
            crop = img[rect[1]:rect[3], rect[0]:rect[2]] if rect else crop_bbox(img, box, pad=0.04)
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < args.min_h:
                continue
            seen_cam.add(cam)
            hasf = has_frontal_face(face, crop)
            scored.append((1 if hasf else 0, bh, fid, cam, crop))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)   # prefer face, then larger box
        for hasf, bh, fid, cam, crop in scored[:args.refs_per_identity]:
            ref_id = f"{vid}_{ak}_{cam}_{fid}"
            clean_p = Path(args.out) / "refs_clean" / f"{ref_id}.jpg"
            cv2.imwrite(str(clean_p), crop)
            upsert(conn, "refs", {"ref_id": ref_id, "identity_id": f"{vid}_{ak}",
                                  "kf_id": f"{vid}_{cam}_{fid}", "tier": 3,
                                  "diversity_score": float(bh) + (100.0 if hasf else 0.0),
                                  "clean_path": str(clean_p), "raw_path": str(clean_p)})
            n_ref += 1
    conn.commit()
    return n_tgt, n_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cams", type=int, default=4, help="number of exo cameras for each target")
    ap.add_argument("--peaks-per-seq", type=int, default=6)
    ap.add_argument("--refs-per-identity", type=int, default=6, help="candidate refs per actor (for human pick)")
    ap.add_argument("--min-h", type=int, default=220, help="min subject height (px); smaller = too far, drop")
    ap.add_argument("--max-overlap", type=float, default=0.35, help="refs: max allowed IoU with other 2D boxes")
    ap.add_argument("--ref-scan-frames", type=int, default=40, help="frames to scan for refs (largest 3D separation)")
    ap.add_argument("--no-face-check", action="store_true", help="do not require a detected face (default: require)")
    ap.add_argument("--packages", default="")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "refs_clean").mkdir(parents=True, exist_ok=True)
    face = None
    if not args.no_face_check:
        from insightface.app import FaceAnalysis
        face = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider"])
        face.prepare(ctx_id=0, det_size=(640, 640))

    conn = connect(args.db)
    seqs = find_sequences(Path(args.root).expanduser())
    if args.packages:
        keep = set(args.packages.split(","))
        seqs = [s for s in seqs if s.parents[1].name in keep]
    seqs = [s for i, s in enumerate(seqs) if i % args.n_shards == args.shard]
    print(f"extract {len(seqs)} sequences (shard {args.shard}/{args.n_shards})")
    T = R = 0
    for k, s in enumerate(seqs):
        t, r = process_seq(s, args, conn, face)
        T += t; R += r
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(seqs)} seqs, {T} targets, {R} refs", flush=True)
    print(f"done: {T} target keyframes, {R} identity refs from {len(seqs)} sequences")


if __name__ == "__main__":
    main()
