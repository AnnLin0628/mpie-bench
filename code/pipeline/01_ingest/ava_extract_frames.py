#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AVA Directional frame extraction(SG, CPU): original movie -> Target graph candidate + Reference image candidates jpg。

target map: core contact8kind(fight/grab/handshake/hug/kick/kiss/lift/push)multiplayer keyframes,
  pump every frame ts-0.5 / ts / ts+0.5 three(Tag span±1s, encrypted sampling);
Reference image candidates: Annotation timestamp of "no core contact tag" in the same movie, Every 4s take one(Auditions for facial formulas)。
name: <movie>_f<millisecond>.jpg (target map) / <movie>_r<millisecond>.jpg (reference candidate), and CC0 Naming isomorphism.
incremental security: already exists jpg jump over, Can kill at any time/rerun; only handle trainval/ Movies that have been downloaded.
usage: python ava_extract_frames.py [Number of concurrencies=4]
"""
import csv
import os
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
ROOT = Path(".") / "data" / "raw" / "ava"
OUT = ROOT / "frames"
CONTACT = {64, 66, 68, 70, 71, 72, 73, 76}
REF_STRIDE = 4          # Reference candidate sampling interval(Second)
TGT_OFFS = (-0.5, 0.0, 0.5)

frame_people = defaultdict(set)
frame_acts = defaultdict(set)
for f in ("ava_train_v2.2.csv", "ava_val_v2.2.csv"):
    for vid, ts, *_, act, pid in csv.reader(open(ROOT / "annotations" / f)):
        frame_people[(vid, ts)].add(pid)
        frame_acts[(vid, ts)].add(int(act))

movies = {p.stem: p for p in (ROOT / "trainval").iterdir()}
# Only process downloads of complete(Reconciliation with manifest byte count by fetch script guarantee, The skip size here is still changing)
jobs = []
for vid, path in movies.items():
    tgt_ts, ref_ts = [], []
    for (v, ts), acts in frame_acts.items():
        if v != vid:
            continue
        if acts & CONTACT and len(frame_people[(v, ts)]) >= 2:
            tgt_ts.append(int(ts))
        elif int(ts) % REF_STRIDE == 0:
            ref_ts.append(int(ts))
    if tgt_ts:
        jobs.append((vid, str(path), sorted(tgt_ts), sorted(ref_ts)))

def grab(path, sec, dst):
    if os.path.exists(dst):
        return True
    r = subprocess.run([FF, "-ss", f"{sec:.2f}", "-i", path, "-frames:v", "1",
                        "-q:v", "3", "-y", dst],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return r.returncode == 0 and os.path.exists(dst)

def work(job):
    vid, path, tgts, refs = job
    d = OUT / vid
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    for ts in tgts:
        for off in TGT_OFFS:
            sec = ts + off
            n += grab(path, sec, str(d / f"{vid}_f{int(sec*1000):07d}.jpg"))
    for ts in refs:
        n += grab(path, ts, str(d / f"{vid}_r{int(ts)*1000:07d}.jpg"))
    print(f"{vid}: Targetts {len(tgts)} referencets {len(refs)} -> {n} jpg", flush=True)
    return n

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"Movie {len(jobs)} Part to be drawn, concurrent {workers}")
    with Pool(workers) as p:
        total = sum(p.map(work, jobs))
    print(f"ava extract done: {total} jpg -> {OUT}")
