#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 1 (Harmony4D dedicated): Registration has been drawn exo frame + use poses3d True value peak selection frame selection.

Harmony4D and CC0 Paths are different: frame has been jpg(Don't cut video), And with frame-by-frame multiplayer 3D attitude truth,
So the contact density/Number of people/identity**All true value**, bypass Stage3 light stream + Stage5 ArcFace clustering.

Table of contents: extracted/<pkg>/<pkg>/<seq>_<action>/
        exo/camNN/images/FFFFF.jpg     # 22 aircraft seat, Frame number alignment across cameras
        processed_data/poses3d/FFFFF.npy  # (P,J,3[+conf]) frame-by-frame multiplayer 3D joint
practice(per sequence):
  1. poses3d Counting people frame by frame-The closest joint distance of a person → Contact density; find_peaks Peak selection frame
  2. Selected peak frame × Sampling machine position subset → Register keyframes(selected=1, GT n_person/Density file)
  3. poses3d of person The order is stable → Give directly identity_id(seq Inside), Cross-machine reference picture

Run first --probe confirm poses3d shape/image path, More formal ingest。
usage:
  python harmony4d_ingest.py --root ~/mpie_data/raw_video/harmony4d/extracted --probe
  python harmony4d_ingest.py --root <...> --db ~/mpie_data/manifests/mpie.db \
        --out ~/mpie_data --cams 6 --peaks-per-seq 8 [--packages 01_hugging,02_grappling]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, upsert  # noqa: E402

try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

# action name → (Interaction type, Default contact density a priori file). Authentic document by poses3d Distance is further subdivided.
ACTION_MAP = {
    "hugging": ("hug", "C2"),
    "grappling": ("wrestle_grapple", "C3"),
    "grappling2": ("wrestle_grapple", "C3"),
    "ballroom": ("dance", "C2"),
    "ballroom2": ("dance", "C2"),
    "karate": ("fight_combat", "C3"), "karate2": ("fight_combat", "C3"),
    "karate3": ("fight_combat", "C3"),
    "mma": ("fight_combat", "C3"), "mma2": ("fight_combat", "C3"),
    "mma3": ("fight_combat", "C3"), "mma4": ("fight_combat", "C3"),
    "mma5": ("fight_combat", "C3"),
    "sword": ("fight_combat", "C3"),
}


def action_of(seq_name: str):
    """'002_hugging' -> 'hugging'; '043_grappling2' -> 'grappling2'。"""
    tail = seq_name.split("_", 1)[-1]
    return tail


def find_sequences(root: Path):
    """Return all containing exo/ and processed_data/poses3d sequence directory. """
    seqs = []
    for pd in sorted(root.glob("*/*/*/processed_data/poses3d")):
        seq = pd.parents[1]
        if (seq / "exo").is_dir():
            seqs.append(seq)
    return seqs


def load_pose(npy_path: Path):
    """Load a frame poses3d → (pose (P,J,3)rice, valid (P,J)Boolean, keys Stable identity name)。

    Harmony4D Format: dict{aria01:(17,4), aria02:(17,4)}, The last column is the confidence level.
    aria Stable across frames, Take it directly as the true value of the identity.
    """
    d = np.load(npy_path, allow_pickle=True).item()
    keys = sorted(d.keys())
    arr = np.stack([np.asarray(d[k], float) for k in keys])   # (P,17,4)
    pose = arr[..., :3]
    valid = arr[..., 3] > 0.3 if arr.shape[-1] >= 4 else np.ones(arr.shape[:2], bool)
    return pose, valid, keys


def _pair_min_dist(pose, valid, i, j):
    """The minimum Euclidean distance of "effective joints" between two people(rice)Count with close contacts. """
    a, b = pose[i][valid[i]], pose[j][valid[j]]
    if len(a) == 0 or len(b) == 0:
        return np.inf, 0
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return float(d.min()), int((d < 0.15).sum())


def contact_density(pose, valid=None):
    """people-The reciprocal of the distance between the nearest joints of a person + Close contact joint pair count, Calculate the contact density. """
    P = pose.shape[0]
    if valid is None:
        valid = np.ones(pose.shape[:2], bool)
    if P < 2:
        return 0.0, P
    best, close = np.inf, 0
    for i in range(P):
        for j in range(i + 1, P):
            md, c = _pair_min_dist(pose, valid, i, j)
            best = min(best, md); close += c
    if not np.isfinite(best):
        return 0.0, P
    score = (1.0 / (best + 1e-3)) + 0.05 * close
    return float(score), P


def density_band(min_dist):
    """people-The closest joint distance of a person(rice) → C Rough classification. truth driven, More reliable than heuristics.
    Original C3(<0.20) and C4(<0.08) merge into high touch C3。"""
    if min_dist < 0.20:
        return "C3"
    if min_dist < 0.40:
        return "C2"
    if min_dist < 0.80:
        return "C1"
    return "C0"


def pick_cameras(exo: Path, n: int):
    cams = sorted([d.name for d in exo.iterdir() if d.is_dir()])
    if len(cams) <= n:
        return cams
    idx = np.linspace(0, len(cams) - 1, n).round().astype(int)
    return [cams[i] for i in idx]


def frame_img(exo: Path, cam: str, fid: str):
    for ext in (".jpg", ".png"):
        p = exo / cam / "images" / f"{fid}{ext}"
        if p.exists():
            return p
    return None


def process_seq(seq: Path, args, conn):
    action = action_of(seq.name)
    itype, _prior = ACTION_MAP.get(action, ("other", "C2"))
    pose_dir = seq / "processed_data" / "poses3d"
    poses = sorted(pose_dir.glob("*.npy"))
    if not poses:
        return 0
    # frame-by-frame contact density(Use effective joints)
    fids, scores, mindists, npers, keyss = [], [], [], [], []
    for pp in poses:
        pose, valid, keys = load_pose(pp)
        P = pose.shape[0]
        md = 9.9
        if P >= 2:
            md = min(_pair_min_dist(pose, valid, i, j)[0]
                     for i in range(P) for j in range(i + 1, P))
            if not np.isfinite(md):
                md = 9.9
        sc, _ = contact_density(pose, valid)
        fids.append(pp.stem); scores.append(sc); mindists.append(md)
        npers.append(P); keyss.append(keys)
    scores = np.array(scores)
    # Peak selection
    if find_peaks is not None and len(scores) > 5:
        pk, _ = find_peaks(scores, distance=max(1, len(scores) // (args.peaks_per_seq * 2)))
        if len(pk) == 0:
            pk = np.argsort(scores)[::-1][:args.peaks_per_seq]
    else:
        pk = np.argsort(scores)[::-1][:args.peaks_per_seq]
    pk = sorted(pk, key=lambda i: scores[i], reverse=True)[:args.peaks_per_seq]

    exo = seq / "exo"
    cams = pick_cameras(exo, args.cams)
    vid_base = f"h4d_{seq.name}"
    # video OK(One per sequence, Remember the scene/Interaction type)
    upsert(conn, "videos", {"video_id": vid_base, "dataset": "harmony4d",
                            "path": str(seq), "license_tier": "restricted",
                            "fps": 0, "n_frames": len(poses),
                            "meta_json": {"action": action, "interaction_type": itype,
                                          "n_cams": len(cams)}})
    n_written = 0
    for i in pk:
        fid = fids[i]; band = density_band(mindists[i]); P = npers[i]; keys = keyss[i]
        for cam in cams:
            img = frame_img(exo, cam, fid)
            if img is None:
                continue
            kf_id = f"{vid_base}_{cam}_{fid}"
            upsert(conn, "keyframes", {
                "kf_id": kf_id, "video_id": vid_base, "shot_id": f"{vid_base}_{cam}",
                "frame_idx": int(fid), "n_person": int(P),
                "density_score": float(scores[i]), "density_level": band,
                "sharpness": 0.0, "frame_path": str(img), "selected": 1})
            # GT identity: aria Name spans frames/Stable across aircraft locations → directly as identity_id
            for pidx, ak in enumerate(keys):
                upsert(conn, "persons", {
                    "track_id": f"{kf_id}_p{pidx}", "video_id": vid_base,
                    "identity_id": f"{vid_base}_{ak}", "bbox_json": None,
                    "n_frames": 1, "face_emb_path": None, "body_emb_path": None})
            n_written += 1
    conn.commit()
    return n_written


def probe(root: Path):
    seqs = find_sequences(root)
    print(f"found {len(seqs)} sequences")
    if not seqs:
        return
    s = seqs[0]
    print("sample seq:", s)
    pose_dir = s / "processed_data" / "poses3d"
    pp = sorted(pose_dir.glob("*.npy"))
    print(f"  poses3d frames: {len(pp)}, first: {pp[0].name if pp else None}")
    if pp:
        pose, valid, keys = load_pose(pp[0])
        print(f"  pose shape (P,J,3): {pose.shape}, keys={keys}, valid={valid.sum()}/{valid.size}")
        sc, P = contact_density(pose, valid)
        md = min((_pair_min_dist(pose, valid, i, j)[0] for i in range(P)
                  for j in range(i + 1, P)), default=9.9)
        print(f"  contact_density={sc:.3f} n_person={P} min_dist={md:.3f}m band={density_band(md)}")
    exo = s / "exo"
    cams = pick_cameras(exo, 6)
    print(f"  exo cams total={len(list(exo.iterdir()))}, picked6={cams}")
    if pp:
        img = frame_img(exo, cams[0], pp[0].stem)
        print(f"  sample image for cam {cams[0]} frame {pp[0].stem}: {img}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--db", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--cams", type=int, default=6)
    ap.add_argument("--peaks-per-seq", type=int, default=8)
    ap.add_argument("--packages", default="", help="Comma separated only process certain packages(like 01_hugging)")
    ap.add_argument("--probe", action="store_true")
    args = ap.parse_args()
    root = Path(args.root).expanduser()
    if args.probe:
        probe(root)
        return
    conn = connect(args.db)
    seqs = find_sequences(root)
    if args.packages:
        keep = set(args.packages.split(","))
        seqs = [s for s in seqs if s.parents[1].name in keep]
    print(f"ingesting {len(seqs)} sequences")
    total = 0
    for k, s in enumerate(seqs):
        n = process_seq(s, args, conn)
        total += n
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(seqs)} seqs, {total} keyframes", flush=True)
    print(f"done: {total} keyframes selected from {len(seqs)} sequences")


if __name__ == "__main__":
    main()
