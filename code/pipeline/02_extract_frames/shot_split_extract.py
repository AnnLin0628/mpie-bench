#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 2: shot cut + Low frame rate frame extraction.

right raw_video Each video in the directory: PySceneDetect cut shot → according to --fps Save frame →
shots/keyframes Basic information writing manifest。
Hi4D/CHI3D/Harmony4D Mostly single-lens continuous collection, shot Segmentation mainly serves the future CC0 stock footage;
Degenerate a single-shot video into a whole segment shot, It will not affect the follow-up.

usage: python shot_split_extract.py --videos ~/mpie_data/raw_video/harmony4d \
        --dataset harmony4d --license restricted \
        --out ~/mpie_data --db ~/mpie_data/manifests/mpie.db --fps 4
"""
import argparse
import hashlib
import sys
from pathlib import Path

import cv2
from scenedetect import detect, ContentDetector

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, upsert  # noqa: E402

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def vid_id(path: Path) -> str:
    return hashlib.md5(str(path).encode()).hexdigest()[:12]


def process(video: Path, dataset: str, license_tier: str, out: Path, conn, fps: float):
    vid = vid_id(video)
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 1. shot cut(fail/single lens → One whole paragraph shot)
    try:
        scenes = detect(str(video), ContentDetector(threshold=27.0))
        shots = [(s[0].get_frames(), s[1].get_frames()) for s in scenes] or [(0, n_frames)]
    except Exception:
        shots = [(0, n_frames)]
    for i, (s, e) in enumerate(shots):
        upsert(conn, "shots", {"shot_id": f"{vid}_s{i:03d}", "video_id": vid,
                               "start_frame": s, "end_frame": e})

    # 2. Low frame rate frame extraction
    frame_dir = out / "frames" / vid
    frame_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, round(src_fps / fps))
    idx, saved = 0, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                fp = frame_dir / f"{idx:07d}.jpg"
                if not fp.exists():
                    cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
        idx += 1
    cap.release()

    upsert(conn, "videos", {"video_id": vid, "dataset": dataset, "path": str(video),
                            "license_tier": license_tier, "fps": src_fps,
                            "n_frames": n_frames, "meta_json": {"extracted": saved, "shots": len(shots)}})
    conn.commit()
    print(f"{video.name}: {len(shots)} shots, {saved} frames @ {fps}fps", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--license", default="restricted", choices=["cc0", "restricted"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--shard", type=int, default=0, help="Multi-card sharding: This process number")
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    conn = connect(args.db)
    vids = sorted(p for p in Path(args.videos).rglob("*") if p.suffix.lower() in VIDEO_EXT)
    vids = [v for i, v in enumerate(vids) if i % args.n_shards == args.shard]
    print(f"shard {args.shard}/{args.n_shards}: {len(vids)} videos")
    for v in vids:
        try:
            process(v, args.dataset, args.license, Path(args.out), conn, args.fps)
        except Exception as e:
            print(f"FAIL {v}: {e}", flush=True)


if __name__ == "__main__":
    main()
