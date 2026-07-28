#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CHI3D extract: Contact frame target map + Separate frame single person reference image candidate, Directly generate the scene board review package.

usage, mpie env):
  python chi3d_extract.py --sessions s02 --limit 4 --dump-annot 4   # smoke: 4action+Projected annotation map
  python chi3d_extract.py                                            # Whole amount train s02/s03/s04
product:
  ~/mpie_data/chi3d_review/chi3d/{flat/, ref_clusters.json, meta.json, debug/}
  ~/chi3d_review.tar (+.md5)  → passSGrun ingest_review_pkg.sh chi3d ~/chi3d_review.tar → /cc0scene/chi3d

design(docs/01_research/dataset_chi3d.md §5):
  scene = s0X session(fixed2people, video_id=chi3d_s0X, All session actions share a scene);
  target map = interaction_contact_signature of fr_id frame × 4 camera full frame;
  Reference image candidates = The frame with the smallest joint distance between two people in each action, projection GT 3D joints bbox layoff person,
              Full session press per person spacing×like high Pick TOP12(Scene board actor group upper limit MAX_GROUP=12);
  identity = joints3d_25 order of persons p0/p1(Assuming stability across actions; If two people are mixed into a certain actor candidate pool during the review,,
        It shows that the order of people is unstable, Need to come back to add ArcFace Merge - no preset complexity first)。
Coordinate convention(official imar-vision-datasets-tools): Xc = (Xw - T) @ R^T, Again w_distortion projection.
"""
import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

DATA = Path.home() / "mpie_data/datasets/chi3d/train"
OUT_ROOT = Path.home() / "mpie_data/chi3d_review"
PKG = OUT_ROOT / "chi3d"
MIN_REF_H = 200          # Reference picture bbox Minimum image height
REFS_PER_PERSON = 12     # signboard MAX_GROUP upper limit
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 0.25, 0.15, 0.06


def project(pts_w, cam):
    """World system(N,3) -> Pixel(N,2), Official distortion model. """
    R = np.array(cam["extrinsics"]["R"])
    T = np.array(cam["extrinsics"]["T"])
    intr = cam["intrinsics_w_distortion"]
    f = np.array(intr["f"])[0]
    c = np.array(intr["c"])[0]
    k = np.array(intr["k"])[0]
    p = np.array(intr["p"])[0][[1, 0]]
    Xc = (pts_w - T) @ R.T
    Xc = np.where(Xc[:, 2:3] < 1e-4, np.nan, Xc)
    x = Xc[:, :2] / Xc[:, 2:3]
    r2 = (x ** 2).sum(1)
    radial = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
    tan = x @ p
    xx = x * (radial + tan)[:, None] + r2[:, None] * p[None]
    return f * xx + c


def bbox_of(pts2d, w, h):
    """Joint points -> Add margin crop box; return None Indicates that the projection basically produces a picture. """
    ok = pts2d[(~np.isnan(pts2d).any(1))]
    ok = ok[(ok[:, 0] > -50) & (ok[:, 0] < w + 50) & (ok[:, 1] > -50) & (ok[:, 1] < h + 50)]
    if len(ok) < 15:
        return None
    x0, y0 = ok.min(0)
    x1, y1 = ok.max(0)
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - bw * MARGIN_X, x1 + bw * MARGIN_X
    y0, y1 = y0 - bh * MARGIN_TOP, y1 + bh * MARGIN_BOT
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 - x0 < 40 or y1 - y0 < MIN_REF_H:
        return None
    return x0, y0, x1, y1


def read_frame(video_path, idx):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, img = cap.read()
    cap.release()
    return img if ret else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["s02", "s03", "s04"])
    ap.add_argument("--limit", type=int, default=0, help="Each session only processes the firstNaction(smoke)")
    ap.add_argument("--dump-annot", type=int, default=0, help="forwardNThe target image overlays the projection joint to debug/")
    args = ap.parse_args()

    if PKG.exists():
        shutil.rmtree(PKG)
    flat = PKG / "flat"
    flat.mkdir(parents=True)
    debug = PKG / "debug"

    meta, groups = {}, []
    inb_all = []
    for sess in args.sessions:
        root = DATA / sess
        cams = sorted(p.name for p in (root / "videos").iterdir() if p.is_dir())
        sig = json.loads((root / "interaction_contact_signature.json").read_text())
        actions = sorted(sig)[: args.limit or None]
        vid = f"chi3d_{sess}"
        print(f"== {sess}: {len(actions)} action × {len(cams)} aircraft seat ==")

        # ---- first time(pure joint, No videoIO): Contact frames per action + split frame ----
        plan = {}
        for act in actions:
            j = np.array(json.loads((root / "joints3d_25" / f"{act}.json").read_text())["joints3d_25"])
            fr = int(sig[act]["fr_id"])
            fr = min(fr, j.shape[1] - 1)
            d = np.linalg.norm(j[0][:, :, None, :] - j[1][:, None, :, :], axis=-1).min((1, 2))
            t_sep = int(d.argmax())
            plan[act] = dict(j=j, fr=fr, t_sep=t_sep, d_sep=float(d[t_sep]))

        # ---- target map: fr_id × Whole camera, whole frame ----
        n_tgt = 0
        for act, p in plan.items():
            slug = act.replace(" ", "-")
            for cam_id in cams:
                vp = root / "videos" / cam_id / f"{act}.mp4"
                if not vp.exists():
                    continue
                img = read_frame(vp, p["fr"])
                if img is None:
                    continue
                cam = json.loads((root / "camera_parameters" / cam_id / f"{act}.json").read_text())
                pts = project(np.vstack([p["j"][0, p["fr"]], p["j"][1, p["fr"]]]), cam)
                h, w = img.shape[:2]
                inb = float(np.mean((pts[:, 0] > 0) & (pts[:, 0] < w) & (pts[:, 1] > 0) & (pts[:, 1] < h)))
                inb_all.append(inb)
                fn = f"{vid}_f{slug}_{cam_id}.jpg"
                cv2.imwrite(str(flat / fn), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                meta[fn] = dict(sess=sess, action=act, cam=cam_id, frame=p["fr"], kind="target")
                n_tgt += 1
                if args.dump_annot and n_tgt <= args.dump_annot:
                    debug.mkdir(exist_ok=True)
                    dbg = img.copy()
                    for x, y in pts[~np.isnan(pts).any(1)]:
                        cv2.circle(dbg, (int(x), int(y)), 4, (0, 0, 255), -1)
                    cv2.imwrite(str(debug / f"annot_{fn}"), dbg)

        # ---- Reference image candidates: per person TOP12 ----
        cands = defaultdict(list)   # person -> [(score, act, cam_id, bbox, frame)]
        for act, p in plan.items():
            for pi in (0, 1):
                best = None
                for cam_id in cams:
                    cp = root / "camera_parameters" / cam_id / f"{act}.json"
                    if not cp.exists():
                        continue
                    cam = json.loads(cp.read_text())
                    pts = project(p["j"][pi, p["t_sep"]], cam)
                    bb = bbox_of(pts, 900, 900)   # CHI3D 900x900, When cropping, press the actual frame sizeclip
                    if bb and (best is None or bb[3] - bb[1] > best[1][3] - best[1][1]):
                        best = (cam_id, bb)
                if best:
                    score = p["d_sep"] * (best[1][3] - best[1][1])
                    cands[pi].append((score, act, best[0], best[1], p["t_sep"]))
        for pi in (0, 1):
            members = []
            for score, act, cam_id, bb, t in sorted(cands[pi], reverse=True)[:REFS_PER_PERSON]:
                img = read_frame(root / "videos" / cam_id / f"{act}.mp4", t)
                if img is None:
                    continue
                h, w = img.shape[:2]
                x0, y0, x1, y1 = min(bb[0], w), min(bb[1], h), min(bb[2], w), min(bb[3], h)
                crop = img[y0:y1, x0:x1]
                if crop.shape[0] < MIN_REF_H:
                    continue
                slug = act.replace(" ", "-")
                fn = f"{vid}_rp{pi}_{slug}_{cam_id}.jpg"
                cv2.imwrite(str(flat / fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                meta[fn] = dict(sess=sess, action=act, cam=cam_id, frame=t, kind="ref", person=pi)
                members.append({"video_id": vid, "thumb": fn})
            groups.append({"avg_sim": 0.99, "members": members})
            print(f"  {sess} p{pi}: reference candidate {len(members)} open")
        print(f"  {sess} target map {n_tgt} open")

    (PKG / "ref_clusters.json").write_text(json.dumps({"groups": groups, "no_face": []}, ensure_ascii=False))
    (PKG / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    if inb_all:
        r = float(np.mean(inb_all))
        print(f"Mean projection rate {r:.2%}" + ("  !! too low, Projection convention may be wrong, look debug/ Annotation diagram" if r < 0.6 else ""))

    tar = Path.home() / "chi3d_review.tar"
    subprocess.run(["tar", "cf", str(tar), "-C", str(OUT_ROOT), "chi3d"], check=True)
    md5 = subprocess.run(["md5sum", str(tar)], capture_output=True, text=True).stdout
    (Path.home() / "chi3d_review.tar.md5").write_text(md5)
    print(f"Packaging completed: {tar}\n{md5.strip()}\npassSGback: bash ingest_review_pkg.sh chi3d ~/chi3d_review.tar")


if __name__ == "__main__":
    main()
