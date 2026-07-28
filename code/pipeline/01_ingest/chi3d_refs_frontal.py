#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CHI3D Supplementary collection of front face reference pictures v2: Face detection direct selection(No more guessing about skeleton orientation)。

v1 use joints_25 of 0/17/18 Dangbi/Ears and head direction——CHI3D The joint order is not BODY_25, Fractional distortion
(Pick out the back view), Obsolete.v2 Change to: Press "Two people can separate" to select candidate cuts. → insightface Detection(Only detect,
non-clustered) → Requires detection of faces close to the top of the actor's projected joints(To prevent the opposite actor’s face from getting in on the background) →
Sort by face height TOP12。

usage, mpie env, GPUquickCPUAlso going):
  CUDA_VISIBLE_DEVICES=<free card> python chi3d_refs_frontal.py            # Default complement s03
  python chi3d_refs_frontal.py --sessions s02 s03 s04
product: ~/chi3d_refs_<sess>.tar → passSG, SGReplace the scene reference candidate on the Kanban board and regenerate it.
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

DATA = Path.home() / "mpie_data/datasets/chi3d/train"
MIN_REF_H = 200
TOPK = 12
MAX_PER_ACTION = 2
MARGIN_X, MARGIN_TOP, MARGIN_BOT = 0.25, 0.32, 0.06


def load_cam(root, cam_id, act):
    return json.loads((root / "camera_parameters" / cam_id / f"{act}.json").read_text())


def project(pts_w, cam):
    R = np.array(cam["extrinsics"]["R"]); T = np.array(cam["extrinsics"]["T"])
    intr = cam["intrinsics_w_distortion"]
    f = np.array(intr["f"])[0]; c = np.array(intr["c"])[0]
    k = np.array(intr["k"])[0]; p = np.array(intr["p"])[0][[1, 0]]
    Xc = (pts_w - T) @ R.T
    Xc = np.where(Xc[:, 2:3] < 1e-4, np.nan, Xc)
    x = Xc[:, :2] / Xc[:, 2:3]
    r2 = (x ** 2).sum(1)
    radial = 1 + k[0] * r2 + k[1] * r2 ** 2 + k[2] * r2 ** 3
    tan = x @ p
    xx = x * (radial + tan)[:, None] + r2[:, None] * p[None]
    return f * xx + c


def bbox_of(pts2d, w, h):
    ok = pts2d[(~np.isnan(pts2d).any(1))]
    ok = ok[(ok[:, 0] > -50) & (ok[:, 0] < w + 50) & (ok[:, 1] > -50) & (ok[:, 1] < h + 50)]
    if len(ok) < 15:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["s03"])
    args = ap.parse_args()

    fa = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    fa.prepare(ctx_id=0, det_size=(640, 640))

    for sess in args.sessions:
        root = DATA / sess
        cams = sorted(p.name for p in (root / "videos").iterdir() if p.is_dir())
        sig = json.loads((root / "interaction_contact_signature.json").read_text())
        vid = f"chi3d_{sess}"
        out = Path.home() / f"chi3d_refs_{sess}"
        out.mkdir(exist_ok=True)
        for f_ in out.glob("*"):
            f_.unlink()

        # candidate: For each action, take the "widest" frame, This frame has one crop for each person in each camera position.
        acts = []
        joints = {}
        for act in sorted(sig):
            j = np.array(json.loads((root / "joints3d_25" / f"{act}.json").read_text())["joints3d_25"])
            d = np.linalg.norm(j[0][:, :, None, :] - j[1][:, None, :, :], axis=-1).min((1, 2))
            t = int(d.argmax())
            acts.append((float(d[t]), act, t))
            joints[act] = j
        acts.sort(reverse=True)          # Actions with high separation are prioritized

        kept = {0: [], 1: []}            # pi -> [(face_h, det_score, fn, crop)]
        per_action = {0: {}, 1: {}}
        for d_sep, act, t in acts:
            if all(len(kept[pi]) >= TOPK * 3 for pi in (0, 1)):
                break                    # Stop when the candidate pool is rich enough, ProvinceIO
            for cam_id in cams:
                cp = root / "camera_parameters" / cam_id / f"{act}.json"
                vp = root / "videos" / cam_id / f"{act}.mp4"
                if not cp.exists() or not vp.exists():
                    continue
                cam = load_cam(root, cam_id, act)
                cap = cv2.VideoCapture(str(vp))
                cap.set(cv2.CAP_PROP_POS_FRAMES, t)
                ret, img = cap.read()
                cap.release()
                if not ret:
                    continue
                h, w = img.shape[:2]
                for pi in (0, 1):
                    if per_action[pi].get(act, 0) >= MAX_PER_ACTION:
                        continue
                    pts = project(joints[act][pi, t], cam)
                    bb = bbox_of(pts, w, h)
                    if not bb:
                        continue
                    x0, y0, x1, y1 = bb
                    crop = img[y0:y1, x0:x1]
                    faces = fa.get(crop)
                    if not faces:
                        continue
                    f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                    fx, fy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
                    ch, cw = crop.shape[:2]
                    # overhead position = The highest point of the actor's projected joint(Relative cropping box); The face must be close to it, Prevent background cross-face
                    top = pts[~np.isnan(pts).any(1)]
                    hx, hy = top[top[:, 1].argmin()] - np.array([x0, y0])
                    if abs(fy - hy) > 0.30 * ch or abs(fx - hx) > 0.35 * cw:
                        continue
                    face_h = float(f.bbox[3] - f.bbox[1])
                    if face_h < 28 or f.det_score < 0.62:
                        continue
                    yaw = abs(float(f.pose[1])) if getattr(f, "pose", None) is not None else 0.0
                    if yaw > 38:
                        continue
                    slug = act.replace(" ", "-")
                    fn = f"{vid}_rp{pi}_{slug}_{cam_id}_t{t}.jpg"
                    kept[pi].append((face_h, float(f.det_score), fn, crop))
                    per_action[pi][act] = per_action[pi].get(act, 0) + 1

        listing = {}
        for pi in (0, 1):
            final = sorted(kept[pi], reverse=True)[:TOPK]
            for face_h, det, fn, crop in final:
                cv2.imwrite(str(out / fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                listing.setdefault(str(pi), []).append({"fn": fn, "face_h": round(face_h), "det": round(det, 2)})
            print(f"{sess} p{pi}: A promising candidate {len(kept[pi])} → Pick {len(final)} open")
        (out / "listing.json").write_text(json.dumps(listing, ensure_ascii=False, indent=1))

        tar = Path.home() / f"chi3d_refs_{sess}.tar"
        subprocess.run(["tar", "cf", str(tar), "-C", str(Path.home()), out.name], check=True)
        print(f"Pack: {tar}  passSGLater, I will replace the reference candidate of the scene in the Kanban board")


if __name__ == "__main__":
    main()
