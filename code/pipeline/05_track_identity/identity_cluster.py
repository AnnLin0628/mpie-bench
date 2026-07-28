#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 5: Person detection and cropping + identity clustering(Full library identity index)。

Selected keyframes(selected=1)and the sparsely sampled frame of the video in which it is located:
  YOLO-person Layoffs → ArcFace human face embedding(host) + DINOv2 whole body embedding(auxiliary,Shameless)
  → Full database agglomerative clustering identity_id。
Harmony4D/Hi4D/CHI3D The subject list is closed and cross- session recurring, Clustering is feasible;
identity_id It is the subsequent cross-video reference image retrieval.(Stage 6)and identity isolation and segmentation(make_splits)the basis of.

usage: python identity_cluster.py --db ~/mpie_data/manifests/mpie.db \
        --out ~/mpie_data/crops --face-thr 0.45 [--shard/--n-shards Only takes effect in the feature extraction phase]
Clustering stage(--cluster)A single process needs to run the entire database at once.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.weights import local_or_name, load_dinov2  # noqa: E402
from common.manifest import connect, upsert, rows  # noqa: E402


def extract(args, conn):
    """Feature extraction: Crop the selected frame and save it crop + face/body embedding npy。"""
    det = YOLO(local_or_name("yolov8m.pt"))
    face = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider"])
    face.prepare(ctx_id=0, det_size=(640, 640))
    dino = load_dinov2().eval().cuda()

    out = Path(args.out); (out / "persons").mkdir(parents=True, exist_ok=True)
    emb_dir = out / "embeddings"; emb_dir.mkdir(exist_ok=True)

    kfs = rows(conn, "SELECT * FROM keyframes WHERE selected=1")
    kfs = [k for i, k in enumerate(kfs) if i % args.n_shards == args.shard]
    for kf in kfs:
        img = cv2.imread(kf["frame_path"])
        if img is None:
            continue
        r = det(img, classes=[0], conf=0.5, verbose=False)[0]
        for pi, box in enumerate(r.boxes.xyxy.cpu().numpy() if r.boxes is not None else []):
            x1, y1, x2, y2 = [int(v) for v in box]
            crop = img[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0 or (y2 - y1) < 128:
                continue
            track_id = f"{kf['kf_id']}_p{pi}"
            cp = out / "persons" / f"{track_id}.jpg"
            cv2.imwrite(str(cp), crop)
            # human face embedding(Maybe shameless)
            faces = face.get(crop)
            femb = faces[0].normed_embedding if faces else None
            # whole body DINOv2
            t = cv2.resize(crop, (224, 224))[:, :, ::-1].copy()
            t = torch.from_numpy(t).permute(2, 0, 1).float().unsqueeze(0).cuda() / 255
            t = (t - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()) / \
                torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
            with torch.no_grad():
                bemb = dino(t)[0].cpu().numpy()
            fe = emb_dir / f"{track_id}.npz"
            np.savez(fe, face=femb if femb is not None else np.array([]), body=bemb)
            upsert(conn, "persons", {"track_id": track_id, "video_id": kf["video_id"],
                                     "identity_id": None, "bbox_json": [x1, y1, x2, y2],
                                     "n_frames": 1, "face_emb_path": str(fe),
                                     "body_emb_path": str(fe)})
        conn.commit()
    print("extract done")


def _agg_labels(embs, thr):
    """A group embedding agglomerative clustering → Label.<2 directly clustered into clusters. """
    if len(embs) < 2:
        return [0] * len(embs)
    X = np.stack([e / (np.linalg.norm(e) + 1e-8) for e in embs])
    return AgglomerativeClustering(n_clusters=None, distance_threshold=thr,
                                   metric="cosine", linkage="average").fit_predict(X)


def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def cluster(args, conn):
    """Identity clustering. Default **press video inside** gather(CC0: The same person only appears in the same video, Cross-video merging is strictly prohibited);
    Press first within each video ArcFace Juyouface track, Then press the faceless track DINOv2 body Similarity **Affiliated** to already
    Have a decent identity("Government" has a face/Structural divisions caused by "faceless dividing pools" causing fandoms to be demolished), If you can't hang it, press it alone. body gather.
    --global-cluster Full database clustering that can be returned(multiple perspectivesGTThe data set does not follow this path; Harmony use ingest Bring your ownGTidentity)。
    """
    ps = rows(conn, "SELECT track_id, video_id, face_emb_path FROM persons")
    recs = {}
    for p in ps:
        fp = p["face_emb_path"]
        if not fp or not Path(fp).exists():
            continue
        with np.load(fp) as z:   # with Close handle promptly, Defend fd leakage(Errno24)
            recs[p["track_id"]] = {"face": np.asarray(z["face"]),
                                   "body": np.asarray(z["body"]),
                                   "vid": p["video_id"] or "novid"}
    # Group: Default is one group per video; global A set of full database of patterns
    groups = {}
    for tid, r in recs.items():
        key = "ALL" if args.global_cluster else r["vid"]
        groups.setdefault(key, []).append(tid)

    def assign(tid, ident):
        conn.execute("UPDATE persons SET identity_id=? WHERE track_id=?", (ident, tid))

    for gk, tids in groups.items():
        faced = [t for t in tids if recs[t]["face"].size]
        faceless = [t for t in tids if not recs[t]["face"].size]
        # 1) Have a face: according to ArcFace gather
        face_lb = {}
        if faced:
            for t, lb in zip(faced, _agg_labels([recs[t]["face"] for t in faced], args.face_thr)):
                face_lb[t] = int(lb)
                assign(t, f"{gk}_f{lb:03d}")
        # 2) Someone with a face and an identity body center of mass, For shameless affiliation
        cents = {}
        for t in faced:
            cents.setdefault(face_lb[t], []).append(recs[t]["body"])
        cents = {lb: np.mean(np.stack(v), 0) for lb, v in cents.items()}
        # 3) Shameless: Affiliate yourself to someone with a decent identity nearby, If it cannot be hung up, press it alone body gather
        rest = []
        for t in faceless:
            best_lb, best = None, -1.0
            for lb, c in cents.items():
                s = _cos(recs[t]["body"], c)
                if s > best:
                    best, best_lb = s, lb
            if best_lb is not None and best >= args.attach_sim:
                assign(t, f"{gk}_f{best_lb:03d}")           # Affiliated to a respectable identity
            else:
                rest.append(t)
        if rest:
            for t, lb in zip(rest, _agg_labels([recs[t]["body"] for t in rest], args.body_thr)):
                assign(t, f"{gk}_b{lb:03d}")
    conn.commit()
    n = rows(conn, "SELECT COUNT(DISTINCT identity_id) c FROM persons")[0]["c"]
    print(f"cluster done: {len(recs)} tracks -> {n} identities "
          f"({'global' if args.global_cluster else 'per-video'})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cluster", action="store_true", help="Run the whole database clustering(single process)")
    ap.add_argument("--face-thr", type=float, default=0.55, help="ArcFace cosine distance threshold, The bigger it is, the more mergers it has.")
    ap.add_argument("--body-thr", type=float, default=0.35, help="Shameless DINOv2 body distance threshold")
    ap.add_argument("--attach-sim", type=float, default=0.55, help="Faceless tracks are linked to those with faces and identities body Similarity lower limit")
    ap.add_argument("--global-cluster", action="store_true", help="Return to full database clustering(By default, press within the video)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()
    conn = connect(args.db)
    if args.cluster:
        cluster(args, conn)
    else:
        extract(args, conn)


if __name__ == "__main__":
    main()
