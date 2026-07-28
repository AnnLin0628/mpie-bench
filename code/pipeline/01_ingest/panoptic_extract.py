#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CMU Panoptic extract: GTcontact peak frame → target map(full frame×aircraft seat), Face formulav3 → Reference picture.

and chi3d_extract isomorphic, difference:
  - projection convention X_cam = R@X + t (CHI3D yes (X-T)@Rᵀ), OpenCV 5Parameter distortion(k1,k2,p1,p2,k3)
  - GT Comes with tracking id Cluster-free, But it must be done first**Ghosting dummy removal**(legacy15 oldGTDuplicate tagging by fans,
    160226_ultimatum1 because 77% Ghosting is obsolete; remaining sequence 0% But just to be on the safe side, go through them all)
  - Multiplayer scene(3-8people): The reference frame is "minimum distance between the actor and others"(Isolation)"Actor-by-actor selection, Not global separation
  - Unit cm; joints19/15 No head joint → overhead margin 0.32(formulav3)

usage, mpie env, needGPUruninsightface):
  python panoptic_extract.py                      # Default final version8sequence
  python panoptic_extract.py --sessions 170221_haggling_b1 --max-tgt 6   # smoke
product: ~/mpie_data/panoptic_review/panoptic/{flat/,ref_clusters.json,meta.json,debug/}
  + ~/panoptic_review.tar(.md5) → passSG: bash ingest_review_pkg.sh panoptic ~/panoptic_review.tar
"""
import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

DATA = Path.home() / "mpie_data/datasets/panoptic"
OUT_ROOT = Path.home() / "mpie_data/panoptic_review/panoptic"

FINAL_SEQS = ["160422_ultimatum1", "160224_ultimatum2", "160226_mafia2", "160422_mafia2",
              "160226_haggling1", "170407_haggling_a2", "160906_pizza1", "160906_band2"]

STEP = 15            # GTSampling step size(0.5s)
GHOST_MEAN = 20.0    # Ghost judgment: Average distance of joints with the same name<20cm
PRESENCE_SAMPLES = 40   # Actor threshold: Appear≥40sample frames(about20s)——hagglingWaiting for the rotation lineup to be calculated according to the length of stay., Don’t use full length ratio
TGT_SPACING = 90     # Target frame minimum interval(3s)
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 0.25, 0.32, 0.06   # formulav3
MIN_REF_H = 200
TOPK, MAX_PER_FRAME = 12, 2
DET_TH, YAW_TH, MIN_FACE_H = 0.62, 38, 26


def dedup(bodies):
    """bodies: [(id, joints Nx4)] → After ghosting [(id, joints Nx4)]"""
    n = len(bodies)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for k in range(i + 1, n):
            a, b = bodies[i][1], bodies[k][1]
            ok = (a[:, 3] > 0.1) & (b[:, 3] > 0.1)
            if ok.sum() >= 5 and float(np.linalg.norm(a[ok, :3] - b[ok, :3], axis=1).mean()) < GHOST_MEAN:
                parent[find(i)] = find(k)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [bodies[max(idx, key=lambda i: (bodies[i][1][:, 3] > 0.1).sum())] for idx in groups.values()]


def load_cams(root, seq):
    calib = json.loads((root / f"calibration_{seq}.json").read_text())
    cams = {}
    for c in calib["cameras"]:
        if not str(c["name"]).startswith("00_"):
            continue
        node = int(c["node"])
        vp = root / "hdVideos" / f"hd_00_{node:02d}.mp4"
        if not vp.exists():
            continue
        cams[node] = {"K": np.array(c["K"]), "R": np.array(c["R"]),
                      "t": np.array(c["t"]).reshape(3), "d": np.array(c["distCoef"]).reshape(-1), "vp": vp}
    return cams


def project(pts, cam):
    Xc = pts @ cam["R"].T + cam["t"]
    Xc = np.where(Xc[:, 2:3] < 1e-4, np.nan, Xc)
    x = Xc[:, :2] / Xc[:, 2:3]
    r2 = (x ** 2).sum(1)
    k1, k2, p1, p2, k3 = cam["d"][:5]
    radial = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xx = x * radial[:, None]
    xx[:, 0] += 2 * p1 * x[:, 0] * x[:, 1] + p2 * (r2 + 2 * x[:, 0] ** 2)
    xx[:, 1] += p1 * (r2 + 2 * x[:, 1] ** 2) + 2 * p2 * x[:, 0] * x[:, 1]
    K = cam["K"]
    return np.stack([K[0][0] * xx[:, 0] + K[0][2], K[1][1] * xx[:, 1] + K[1][2]], 1)


def bbox_of(pts2d, w, h):
    ok = pts2d[~np.isnan(pts2d).any(1)]
    ok = ok[(ok[:, 0] > -50) & (ok[:, 0] < w + 50) & (ok[:, 1] > -50) & (ok[:, 1] < h + 50)]
    if len(ok) < 8:
        return None
    x0, y0 = ok.min(0); x1, y1 = ok.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - bw * MARGIN_X, x1 + bw * MARGIN_X
    y0, y1 = y0 - bh * MARGIN_TOP, y1 + bh * MARGIN_BOT
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 40 or y1 - y0 < MIN_REF_H:
        return None
    return x0, y0, x1, y1


class FrameReader:
    """Hold the handle according to the camera position, Frame numbers out of order seek read. """
    def __init__(self):
        self.caps = {}

    def read(self, vp, fidx):
        cap = self.caps.get(vp)
        if cap is None:
            cap = self.caps[vp] = cv2.VideoCapture(str(vp))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, img = cap.read()
        return img if ret else None

    def close(self):
        for c in self.caps.values():
            c.release()
        self.caps = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=FINAL_SEQS)
    ap.add_argument("--max-tgt", type=int, default=40, help="Maximum number of target frames per sequence")
    ap.add_argument("--tag", default="", help="Multi-card slice name(like shard1): jsonWith suffix and no typingtar, run last panoptic_merge_shards.py merge")
    args = ap.parse_args()

    fa = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))

    flat = OUT_ROOT / "flat"; dbg = OUT_ROOT / "debug"
    flat.mkdir(parents=True, exist_ok=True); dbg.mkdir(exist_ok=True)
    all_groups, meta = [], {}

    for seq in args.sessions:
        root = DATA / seq
        gtdir = next((root / n for n in ("hdPose3d_stage1_coco19", "hdPose3d_stage1") if (root / n).is_dir()), None)
        cams = load_cams(root, seq)
        if gtdir is None or not cams:
            print(f"jump over {seq}: GTorHDVideo missing"); continue
        vid = f"panoptic_{seq}"
        stype = "".join(ch for ch in seq.split("_", 1)[1] if not ch.isdigit()).strip("_")

        # ---- pass1: samplingGT, Go ghosting, Frame-by-frame personnel list ----
        info = {}                          # fidx -> [(id, joints)]
        presence = Counter()
        for f in sorted(gtdir.glob("body3DScene_*.json"))[::STEP]:
            fidx = int(f.stem.split("_")[-1])
            raw = []
            for b in json.loads(f.read_text()).get("bodies", []):
                j = np.array(b.get("joints19") or b.get("joints15"), dtype=float).reshape(-1, 4)
                if (j[:, 3] > 0.1).sum() >= 5:
                    raw.append((int(b["id"]), j))
            ppl = dedup(raw)
            info[fidx] = ppl
            for pid, _ in ppl:
                presence[pid] += 1
        n_frames = len(info)
        thr = max(PRESENCE_SAMPLES, int(0.05 * n_frames))
        actors = sorted(pid for pid, c in presence.items() if c >= thr)
        top = ", ".join(f"id{pid}:{100 * c // max(n_frames, 1)}%" for pid, c in presence.most_common(12))
        print(f"{seq}: sample frame{n_frames} onlyid{len(presence)} threshold{thr} → actor{len(actors)} | {top}")
        rp = {pid: k for k, pid in enumerate(actors)}
        if len(actors) < 2:
            print(f"jump over {seq}: Not enough stable actors({len(actors)})"); continue

        def joints3(j):
            return j[j[:, 3] > 0.1][:, :3]

        def pairdist(a, b):
            return float(np.linalg.norm(joints3(a)[:, None] - joints3(b)[None], axis=-1).min())

        # ---- target frame: Global closest pair distance ascending order + time interval ----
        cand = []
        for fidx, ppl in info.items():
            known = [(pid, j) for pid, j in ppl if pid in rp]
            if len(known) < 2:
                continue
            mind = min(pairdist(a, b) for i, (_, a) in enumerate(known) for _, b in known[i + 1:])
            cand.append((mind, fidx, len(known)))
        cand.sort()
        tgt_frames = []
        for mind, fidx, np_ in cand:
            if all(abs(fidx - t) >= TGT_SPACING for t in tgt_frames):
                tgt_frames.append(fidx)
            if len(tgt_frames) >= args.max_tgt:
                break
        tgt_frames.sort()

        rd = FrameReader()
        n_tgt = 0
        for node, cam in sorted(cams.items()):
            for fidx in tgt_frames:
                img = rd.read(cam["vp"], fidx)
                if img is None:
                    continue
                fn = f"{vid}_f{fidx:06d}_hd{node:02d}.jpg"
                cv2.imwrite(str(flat / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                meta[fn] = {"sess": seq, "action": stype, "cam": f"hd{node:02d}", "frame": fidx, "kind": "target"}
                n_tgt += 1

        # Alignment self-test: first target frameGTprojection → debug/
        if tgt_frames:
            node = sorted(cams)[0]
            img = rd.read(cams[node]["vp"], tgt_frames[0])
            if img is not None:
                for pid, j in info[tgt_frames[0]]:
                    for u, v in project(joints3(j), cams[node]):
                        if not (np.isnan(u) or np.isnan(v)):
                            cv2.circle(img, (int(u), int(v)), 4, (0, 255, 0) if pid in rp else (0, 0, 255), -1)
                cv2.imwrite(str(dbg / f"{vid}_align_hd{node:02d}_f{tgt_frames[0]:06d}.jpg"), img)

        # ---- Reference picture: Get frames by actor by isolation, formulav3 ----
        iso = {pid: [] for pid in actors}   # pid -> [(Isolation, fidx)]
        for fidx, ppl in info.items():
            known = [(pid, j) for pid, j in ppl if pid in rp]
            for i, (pid, a) in enumerate(known):
                others = [b for k, (_, b) in enumerate(known) if k != i]
                if others:
                    iso[pid].append((min(pairdist(a, b) for b in others), fidx))
        for pid in actors:
            kept, per_frame, fallback = [], Counter(), []
            for d_iso, fidx in sorted(iso[pid], reverse=True)[:60]:
                if len(kept) >= TOPK * 3:
                    break
                jmap = dict(info[fidx])
                if pid not in jmap:
                    continue
                pts3 = joints3(jmap[pid])
                for node, cam in sorted(cams.items()):
                    if per_frame[fidx] >= MAX_PER_FRAME:
                        break
                    pts = project(pts3, cam)
                    img = rd.read(cam["vp"], fidx)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    bb = bbox_of(pts, w, h)
                    if not bb:
                        continue
                    x0, y0, x1, y1 = bb
                    crop = img[y0:y1, x0:x1]
                    faces = fa.get(crop)
                    if not faces:
                        continue
                    f_ = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    fx, fy = (f_.bbox[0] + f_.bbox[2]) / 2, (f_.bbox[1] + f_.bbox[3]) / 2
                    face_h = float(f_.bbox[3] - f_.bbox[1])
                    top = pts[~np.isnan(pts).any(1)]
                    hx, hy = top[top[:, 1].argmin()] - np.array([x0, y0])
                    if abs(fy - hy) > 2.0 * face_h or abs(fx - hx) > 2.0 * face_h:
                        continue          # Prevent cross-face: Calculate distance according to face scale(In crowded scenes, neighbors’ faces will be included according to the proportion of the cropping frame.)
                    fn = f"{vid}_rp{rp[pid]}_f{fidx:06d}_hd{node:02d}.jpg"
                    rec = (face_h, float(f_.det_score), fn, crop, fidx)
                    if face_h < MIN_FACE_H or f_.det_score < 0.5:
                        continue
                    fallback.append(rec)
                    yaw = abs(float(f_.pose[1])) if getattr(f_, "pose", None) is not None else 0.0
                    if f_.det_score < DET_TH or yaw > YAW_TH:
                        continue
                    kept.append(rec)
                    per_frame[fidx] += 1
            pool = sorted(kept, reverse=True)[:TOPK] or sorted(fallback, reverse=True)[:4]
            members = []
            for face_h, det, fn, crop, fidx in pool:
                cv2.imwrite(str(flat / fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                meta[fn] = {"sess": seq, "action": stype, "cam": fn.rsplit("_", 1)[1][:-4],
                            "frame": fidx, "kind": "ref"}
                members.append({"video_id": vid, "thumb": fn})
            if members:
                all_groups.append({"avg_sim": 0.99, "members": members})
            tag = "Front face" if kept else "⚠reveal all the details(No face)"
            print(f"{seq} p{rp[pid]}(id{pid}): candidate{len(kept)} → Pick{len(members)} {tag}")
        rd.close()
        print(f"{seq}: actor{len(actors)} target frame{len(tgt_frames)} target map{n_tgt}")

    sfx = f"_{args.tag}" if args.tag else ""
    (OUT_ROOT / f"ref_clusters{sfx}.json").write_text(json.dumps({"groups": all_groups, "no_face": []}, ensure_ascii=False))
    (OUT_ROOT / f"meta{sfx}.json").write_text(json.dumps(meta, ensure_ascii=False))
    if args.tag:
        print(f"Sharding {args.tag} Finish; Run after all slices are completed panoptic_merge_shards.py Merge and package")
        return
    tar = Path.home() / "panoptic_review.tar"
    subprocess.run(["tar", "cf", str(tar), "-C", str(OUT_ROOT.parent), OUT_ROOT.name], check=True)
    subprocess.run(f"md5sum {tar} > {tar}.md5", shell=True, check=True)
    print(f"Pack: {tar} (+.md5)  passSGback: bash ingest_review_pkg.sh panoptic {tar.name}")


if __name__ == "__main__":
    main()
