#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extended anatomy metrics for MPIE mesh v3.

Covers (beyond joint/bone/self/shape):
  - **structure overcount** (image-side hand/torso count vs R#) — MUST-catch
  - part-mesh self-collision (limb vs torso)
  - hand finger-chain ratios
  - cross-person scale consistency
  - contact-region Anat×Inter
  - limb-ownership confusion at contact
  - explain-residual proxy (2D bbox vs image foreground)
  - optional AbHuman/YOLO hook (weights path; no-op if missing)

Skin-tone / appearance bleed → VQA, not this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from mesh_metrics import (
    _J,
    _as_float_j3d,
    _as_float_verts,
    _seg_len,
    _subsample,
    joint_limit_score,
)


# Joint groups for part labeling / ownership
_TORSO = ("pelvis", "spine1", "neck", "head", "left_hip", "right_hip")
_L_ARM = ("left_shoulder", "left_elbow", "left_wrist")
_R_ARM = ("right_shoulder", "right_elbow", "right_wrist")
_L_LEG = ("left_hip", "left_knee", "left_ankle")
_R_LEG = ("right_hip", "right_knee", "right_ankle")


def _joint_xyz(j: np.ndarray, name: str) -> Optional[np.ndarray]:
    idx = _J.get(name)
    if idx is None or idx >= len(j):
        return None
    return j[idx]


def _person_height(j3d: Any) -> Optional[float]:
    j = _as_float_j3d(j3d)
    if j is None:
        return None
    head = _joint_xyz(j, "head")
    la = _joint_xyz(j, "left_ankle")
    ra = _joint_xyz(j, "right_ankle")
    if head is None:
        return None
    ankles = [a for a in (la, ra) if a is not None]
    if not ankles:
        pelvis = _joint_xyz(j, "pelvis")
        if pelvis is None:
            return None
        return float(np.linalg.norm(head - pelvis) * 2.2)
    ankle = np.mean(np.stack(ankles, 0), 0)
    h = float(np.linalg.norm(head - ankle))
    return h if h > 0.3 else None


def hand_fine_score(j3d: Optional[Any]) -> Dict[str, Any]:
    """Finger-chain length ratios (wrist→index3 vs forearm). Needs SMPL-X hand joints."""
    j = _as_float_j3d(j3d)
    if j is None or j.shape[0] < 43:
        return {"score": None, "available": False, "ratios": {}}
    scores = []
    ratios = {}
    for side, tip in (("left", "left_index"), ("right", "right_index")):
        wrist = f"{side}_wrist"
        elbow = f"{side}_elbow"
        finger = _seg_len(j, wrist, tip)
        forearm = _seg_len(j, elbow, wrist)
        if finger is None or forearm is None or forearm < 1e-4:
            continue
        r = finger / forearm
        ratios[f"{side}_finger_over_forearm"] = float(r)
        # adult finger chain << forearm; 0.25–0.75 soft band
        if 0.25 <= r <= 0.75:
            scores.append(1.0)
        elif r < 0.25:
            scores.append(max(0.0, 1.0 - (0.25 - r) / 0.25))
        else:
            scores.append(max(0.0, 1.0 - (r - 0.75) / 0.75))
    if not scores:
        return {"score": None, "available": False, "ratios": ratios}
    return {"score": float(np.mean(scores)), "available": True, "ratios": ratios}


def part_mesh_self_collision(
    verts: Any,
    j3d: Optional[Any],
    *,
    n_sample: int = 256,
    tau: float = 0.015,
) -> Dict[str, Any]:
    """Limb cloud vs torso cloud proximity (non-adjacent parts).

    Approximates HumanScore self-collision without full face BVH.
    """
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return {"score": None, "available": False, "hit_ratio": None}

    v = _as_float_verts(verts)
    j = _as_float_j3d(j3d)
    if j is None:
        return {"score": None, "available": False, "hit_ratio": None}

    centers = {}
    for name in _TORSO + _L_ARM + _R_ARM + _L_LEG + _R_LEG:
        p = _joint_xyz(j, name)
        if p is not None:
            centers[name] = p
    if len(centers) < 6:
        return {"score": None, "available": False, "hit_ratio": None}

    names = list(centers.keys())
    pts = np.stack([centers[n] for n in names], 0)
    # assign each vert to nearest joint
    tree = cKDTree(pts)
    vs = _subsample(v, min(len(v), 2048), 3)
    _, nn = tree.query(vs, k=1, workers=-1)
    labels = [names[i] for i in nn]

    def cloud(group):
        idx = [i for i, lb in enumerate(labels) if lb in group]
        if len(idx) < 8:
            return None
        return _subsample(vs[idx], n_sample, 4)

    torso = cloud(set(_TORSO))
    hits = []
    for limb in (set(_L_ARM), set(_R_ARM), set(_L_LEG), set(_R_LEG)):
        # exclude hip from leg-torso adjacency noise: use knee/ankle only for legs
        if limb & set(_L_LEG) or limb & set(_R_LEG):
            limb = limb - {"left_hip", "right_hip"}
        lc = cloud(limb)
        if torso is None or lc is None:
            continue
        d, _ = cKDTree(torso).query(lc, k=1, workers=-1)
        hits.append(float(np.mean(d <= tau)))
    if not hits:
        return {"score": None, "available": False, "hit_ratio": None}
    hit = float(np.max(hits))
    # 0% OK; ≥8% of limb samples glued into torso → 0
    score = float(max(0.0, 1.0 - hit / 0.08))
    return {"score": score, "available": True, "hit_ratio": hit}


def cross_person_scale_score(j3ds: Sequence[Any]) -> Dict[str, Any]:
    """Two adults in one photo should not differ wildly in mesh height/limb scale."""
    heights = []
    uarms = []
    for j3d in j3ds:
        h = _person_height(j3d)
        if h is not None:
            heights.append(h)
        j = _as_float_j3d(j3d)
        if j is not None:
            la = _seg_len(j, "left_shoulder", "left_elbow")
            ra = _seg_len(j, "right_shoulder", "right_elbow")
            if la and ra:
                uarms.append(0.5 * (la + ra))
            elif la or ra:
                uarms.append(float(la or ra))
    if len(heights) < 2:
        return {"score": None, "available": False, "height_ratio": None, "arm_ratio": None}
    hr = max(heights) / max(min(heights), 1e-6)
    # allow up to 1.25 (adult variance / child-adult soft); fail by 1.55
    if hr <= 1.25:
        s_h = 1.0
    elif hr >= 1.55:
        s_h = 0.0
    else:
        s_h = 1.0 - (hr - 1.25) / 0.30
    s_a = 1.0
    ar = None
    if len(uarms) >= 2:
        ar = max(uarms) / max(min(uarms), 1e-6)
        if ar <= 1.30:
            s_a = 1.0
        elif ar >= 1.70:
            s_a = 0.0
        else:
            s_a = 1.0 - (ar - 1.30) / 0.40
    return {
        "score": float(0.6 * s_h + 0.4 * s_a),
        "available": True,
        "height_ratio": float(hr),
        "arm_ratio": float(ar) if ar is not None else None,
    }


def contact_region_anat_score(
    humans_verts: Sequence[Any],
    j3ds: Sequence[Any],
    body_poses: Sequence[Any],
    *,
    tau_contact: float = 0.05,
    n_sample: int = 256,
) -> Dict[str, Any]:
    """Anat restricted to joints near person–person contact band (Anat×Inter)."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return {"score": None, "available": False, "n_contact_joints": 0}

    n = len(humans_verts)
    if n < 2:
        return {"score": None, "available": False, "n_contact_joints": 0}

    joint_scores = []
    n_cj = 0
    for i in range(n):
        for j in range(i + 1, n):
            va = _subsample(_as_float_verts(humans_verts[i]), n_sample, 5)
            vb = _subsample(_as_float_verts(humans_verts[j]), n_sample, 6)
            d_a, _ = cKDTree(vb).query(va, k=1, workers=-1)
            if float(np.min(d_a)) > tau_contact:
                continue
            contact_pts = va[d_a <= tau_contact]
            if len(contact_pts) < 3:
                continue
            for pi, j3d, pose in (
                (i, j3ds[i] if i < len(j3ds) else None, body_poses[i] if i < len(body_poses) else None),
                (j, j3ds[j] if j < len(j3ds) else None, body_poses[j] if j < len(body_poses) else None),
            ):
                jj = _as_float_j3d(j3d)
                if jj is None:
                    continue
                # joints near contact cloud
                near = []
                for name, idx in _J.items():
                    if idx >= len(jj) or name.endswith("_index"):
                        continue
                    dist = float(np.min(np.linalg.norm(contact_pts - jj[idx], axis=1)))
                    if dist <= 0.12:
                        near.append(name)
                        n_cj += 1
                # if contact involves arms, emphasize arm joint limits via full pose score
                # (per-joint isolation needs pose indexing; use global joint score as proxy
                #  when any arm/torso joint is in the band)
                if near:
                    joint_scores.append(joint_limit_score(pose))
    if not joint_scores:
        return {"score": None, "available": False, "n_contact_joints": 0}
    return {
        "score": float(np.mean(joint_scores)),
        "available": True,
        "n_contact_joints": int(n_cj),
    }


def ownership_confusion_score(
    humans_verts: Sequence[Any],
    j3ds: Sequence[Any],
    *,
    tau_contact: float = 0.04,
    n_sample: int = 256,
) -> Dict[str, Any]:
    """Contact-band verts of A closer to B's limbs than A's → ownership confusion."""
    try:
        from scipy.spatial import cKDTree
    except Exception:
        return {"score": None, "available": False, "confused_frac": None}

    n = len(humans_verts)
    if n < 2:
        return {"score": None, "available": False, "confused_frac": None}

    fracs = []
    for i in range(n):
        for j in range(i + 1, n):
            ji = _as_float_j3d(j3ds[i] if i < len(j3ds) else None)
            jj = _as_float_j3d(j3ds[j] if j < len(j3ds) else None)
            if ji is None or jj is None:
                continue
            va = _subsample(_as_float_verts(humans_verts[i]), n_sample, 7)
            vb = _subsample(_as_float_verts(humans_verts[j]), n_sample, 8)
            d_ab, _ = cKDTree(vb).query(va, k=1, workers=-1)
            band = va[d_ab <= tau_contact]
            if len(band) < 5:
                continue

            def limb_pts(j3d):
                pts = []
                for name in _L_ARM + _R_ARM + _L_LEG + _R_LEG:
                    p = _joint_xyz(j3d, name)
                    if p is not None:
                        pts.append(p)
                return np.stack(pts, 0) if pts else None

            li, lj = limb_pts(ji), limb_pts(jj)
            if li is None or lj is None:
                continue
            di, _ = cKDTree(li).query(band, k=1, workers=-1)
            dj, _ = cKDTree(lj).query(band, k=1, workers=-1)
            # A's contact verts nearer to B's skeleton
            fracs.append(float(np.mean(dj + 0.02 < di)))
    if not fracs:
        return {"score": None, "available": False, "confused_frac": None}
    confused = float(np.max(fracs))
    score = float(max(0.0, 1.0 - confused / 0.35))
    return {"score": score, "available": True, "confused_frac": confused}


def explain_residual_proxy(
    img_path: Optional[Path],
    j2ds: Sequence[Any],
    *,
    img_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Image–mesh residual via multi-cue fg ∩ joint-bbox (discrimination-first).

    SMPL priors wash joint/bone; this checks fit vs image person-mass.
    Multi-cue max reduces style-driven false lows; still fails when all agree.
    """
    if not img_path or not Path(img_path).is_file() or not j2ds:
        return {"score": None, "available": False, "iou": None}
    try:
        from PIL import Image
    except Exception:
        return {"score": None, "available": False, "iou": None}

    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    # defined below; resolved at call time — unpad Multi-HMR canvas j2d
    j2ds_px = _j2ds_to_original_pixels(j2ds, (w, h), img_size=img_size)
    if not j2ds_px:
        return {"score": None, "available": False, "iou": None}
    pts = np.concatenate(j2ds_px, 0)
    if len(pts) < 4:
        return {"score": None, "available": False, "iou": None}

    rgb = np.asarray(im, dtype=np.float64) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    sat = rgb.max(-1) - rgb.min(-1)

    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    bw, bh = float(x1 - x0), float(y1 - y0)
    if bw < 4 or bh < 4 or (bw * bh) < 0.002 * w * h:
        return {
            "score": 0.0,
            "available": True,
            "iou": 0.0,
            "method": "collapsed_j2d",
            "bbox_frac": float((bw * bh) / (w * h + 1e-6)),
        }

    pad = 0.08 * max(w, h)
    x0i, y0i = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    x1i, y1i = int(min(w - 1, x1 + pad)), int(min(h - 1, y1 + pad))
    mask = np.zeros((h, w), dtype=bool)
    mask[y0i : y1i + 1, x0i : x1i + 1] = True

    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    fg_luma = np.abs(gray - bg) > 0.10

    cy0, cy1 = h // 5, 4 * h // 5
    cx0, cx1 = w // 5, 4 * w // 5
    center = gray[cy0:cy1, cx0:cx1]
    thr = float(np.median(center))
    fg_otsu = gray < thr if float(np.mean(center < thr)) > 0.45 else gray > thr
    fg_sat = sat > max(0.08, float(np.percentile(sat, 60)))

    ious = {}
    for name, fg in (("luma", fg_luma), ("otsu", fg_otsu), ("sat", fg_sat)):
        inter = float(np.logical_and(mask, fg).sum())
        union = float(np.logical_or(mask, fg).sum()) + 1e-6
        ious[name] = inter / union

    iou = float(max(ious.values()))
    score = float(np.clip((iou - 0.06) / 0.28, 0.0, 1.0))
    bbox_frac = float((x1i - x0i + 1) * (y1i - y0i + 1) / (w * h + 1e-6))
    if bbox_frac > 0.85:
        score *= 0.7
    return {
        "score": score,
        "available": True,
        "iou": iou,
        "iou_cues": {k: float(v) for k, v in ious.items()},
        "method": "multi_cue_max",
        "bbox_frac": bbox_frac,
    }


# Skeleton edges for "explained by N fitted people" mask (unified extra-limb).
_SKELETON_EDGES = (
    ("pelvis", "spine1"),
    ("spine1", "neck"),
    ("neck", "head"),
    ("neck", "left_shoulder"),
    ("neck", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),
    ("pelvis", "left_hip"),
    ("pelvis", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def _as_j2d_arr(j2: Any) -> Optional[np.ndarray]:
    if j2 is None:
        return None
    if hasattr(j2, "detach"):
        j2 = j2.detach().cpu().numpy()
    arr = np.asarray(j2, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or arr.shape[0] < 16:
        return None
    return arr


def j2d_canvas_to_original(
    pts: np.ndarray,
    orig_wh: Tuple[int, int],
    canvas_size: int,
) -> np.ndarray:
    """Inverse of Multi-HMR ``ImageOps.contain`` + ``pad`` to square canvas.

    ``j2d`` from the model lives in ``canvas_size×canvas_size`` (e.g. 896),
    **not** in original image pixels. Naive ``* w/canvas`` is wrong for
    non-square photos and was causing skeleton_empty → P_extra≡0.
    """
    w0, h0 = float(orig_wh[0]), float(orig_wh[1])
    S = float(canvas_size)
    scale = min(S / max(w0, 1.0), S / max(h0, 1.0))
    w1, h1 = w0 * scale, h0 * scale
    # PIL ImageOps.pad centers the contained image
    off_x = (S - w1) / 2.0
    off_y = (S - h1) / 2.0
    out = np.asarray(pts, dtype=np.float64).copy()
    out[:, 0] = (out[:, 0] - off_x) / max(scale, 1e-9)
    out[:, 1] = (out[:, 1] - off_y) / max(scale, 1e-9)
    return out


def _j2ds_to_original_pixels(
    j2ds: Sequence[Any],
    orig_wh: Tuple[int, int],
    *,
    img_size: Optional[int] = None,
) -> List[np.ndarray]:
    w, h = orig_wh
    out: List[np.ndarray] = []
    canvas = int(img_size) if img_size else 0
    for j2 in j2ds or []:
        arr = _as_j2d_arr(j2)
        if arr is None:
            continue
        pts = arr[:, :2].copy()
        mx = float(np.nanmax(np.abs(pts)))
        if mx <= 1.5:
            pts[:, 0] *= w
            pts[:, 1] *= h
        elif canvas > 0 and mx <= canvas * 1.25:
            pts = j2d_canvas_to_original(pts, (w, h), canvas)
        # else: already looks like original pixels
        out.append(pts)
    return out


def _paint_disk(mask: np.ndarray, x: int, y: int, r: int) -> None:
    h, w = mask.shape
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask[y0:y1, x0:x1] |= (yy - y) ** 2 + (xx - x) ** 2 <= r * r


def _paint_thick_line(
    mask: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int
) -> None:
    """Numpy thick line (no OpenCV)."""
    n = int(max(abs(x1 - x0), abs(y1 - y0), 1))
    xs = np.linspace(x0, x1, n + 1)
    ys = np.linspace(y0, y1, n + 1)
    for x, y in zip(xs, ys):
        _paint_disk(mask, int(round(x)), int(round(y)), r)


def _dilate_bool(mask: np.ndarray, r: int) -> np.ndarray:
    """Disk dilate without (2r+1)^2 footprint (OOM on large r / images)."""
    r = int(max(0, min(int(r), 48)))
    if r <= 0:
        return mask
    try:
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(~mask) <= float(r)
    except Exception:
        pass
    try:
        from scipy.ndimage import binary_dilation

        out = mask
        cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
        for _ in range(r):
            out = binary_dilation(out, structure=cross)
        return out
    except Exception:
        out = mask.copy()
        h, w = mask.shape
        ys, xs = np.where(mask)
        for y, x in zip(ys[::4], xs[::4]):
            out[
                max(0, y - r) : min(h, y + r + 1),
                max(0, x - r) : min(w, x + r + 1),
            ] = True
        return out


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Connected components; prefer scipy, else cv2, else single-blob fallback."""
    try:
        from scipy.ndimage import label

        lab, n = label(mask)
        return lab, int(n)
    except Exception:
        pass
    try:
        import cv2

        n, lab = cv2.connectedComponents(mask.astype(np.uint8))
        return lab, int(n - 1)
    except Exception:
        # treat whole leftover as one component
        lab = mask.astype(np.int32)
        return lab, (1 if mask.any() else 0)


def _fill_convex_quad(mask: np.ndarray, pts: Sequence[Tuple[int, int]]) -> None:
    """Fill a convex quad/triangle on a bool mask (scanline; no cv2)."""
    if len(pts) < 3:
        return
    h, w = mask.shape
    ys = [p[1] for p in pts]
    y0, y1 = max(0, min(ys)), min(h - 1, max(ys))
    pts_f = [(float(x), float(y)) for x, y in pts]
    for y in range(y0, y1 + 1):
        xs: List[float] = []
        for i in range(len(pts_f)):
            x_a, y_a = pts_f[i]
            x_b, y_b = pts_f[(i + 1) % len(pts_f)]
            if abs(y_a - y_b) < 1e-6:
                continue
            if (y_a <= y <= y_b) or (y_b <= y <= y_a):
                t = (y - y_a) / (y_b - y_a)
                xs.append(x_a + t * (x_b - x_a))
        if len(xs) < 2:
            continue
        xa, xb = int(np.floor(min(xs))), int(np.ceil(max(xs)))
        xa, xb = max(0, xa), min(w - 1, xb)
        if xa <= xb:
            mask[y, xa : xb + 1] = True


def _draw_explained_skeleton(
    j2ds_px: Sequence[np.ndarray],
    h: int,
    w: int,
    *,
    thickness: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Raster thick stick-figure + torso fill from kept people (numpy)."""
    thr = thickness if thickness is not None else max(8, int(0.045 * max(h, w)))
    r = max(2, thr // 2)
    mask = np.zeros((h, w), dtype=bool)
    drew = False
    for arr in j2ds_px or []:
        if arr is None or len(arr) < 16:
            continue
        # torso plate: shoulders → hips (keeps body-fg attached to skeleton)
        torso_names = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")
        torso_pts: List[Tuple[int, int]] = []
        for name in torso_names:
            ii = _J.get(name)
            if ii is None or ii >= len(arr):
                torso_pts = []
                break
            torso_pts.append(
                (
                    int(np.clip(round(float(arr[ii, 0])), 0, w - 1)),
                    int(np.clip(round(float(arr[ii, 1])), 0, h - 1)),
                )
            )
        if len(torso_pts) == 4:
            _fill_convex_quad(mask, torso_pts)
            drew = True
        for a, b in _SKELETON_EDGES:
            ia, ib = _J.get(a), _J.get(b)
            if ia is None or ib is None or ia >= len(arr) or ib >= len(arr):
                continue
            x0 = int(np.clip(round(float(arr[ia, 0])), 0, w - 1))
            y0 = int(np.clip(round(float(arr[ia, 1])), 0, h - 1))
            x1 = int(np.clip(round(float(arr[ib, 0])), 0, w - 1))
            y1 = int(np.clip(round(float(arr[ib, 1])), 0, h - 1))
            _paint_thick_line(mask, x0, y0, x1, y1, r)
            _paint_disk(mask, x0, y0, r + 1)
            _paint_disk(mask, x1, y1, r + 1)
            drew = True
    if not drew:
        return None
    # fatten so normal body silhouette stays connected to the stick figure
    return _dilate_bool(mask, max(3, r))


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Person-mass mask. Avoid median-tie Otsu that marks the whole image fg."""
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    sat = rgb.max(-1) - rgb.min(-1)
    h, w = gray.shape
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    bg = float(np.median(border))
    cy0, cy1 = h // 5, 4 * h // 5
    cx0, cx1 = w // 5, 4 * w // 5
    center = gray[cy0:cy1, cx0:cx1]
    cmed = float(np.median(center))
    # contrast vs border (main cue)
    fg_luma = np.abs(gray - bg) > 0.10
    # person side of mid-threshold between border and center (no tie flip)
    mid = 0.5 * (bg + cmed)
    fg_mid = gray < mid if cmed < bg else gray > mid
    fg_sat = sat > max(0.10, float(np.percentile(sat, 70)))
    # require luma contrast OR (sat ∧ mid); never OR a full-frame mid mask alone
    return fg_luma | (fg_sat & fg_mid)


def _project_verts_with_joints(
    verts: np.ndarray,
    j3d: np.ndarray,
    j2d: np.ndarray,
) -> Optional[np.ndarray]:
    """Project mesh verts into the same 2D frame as ``j2d`` (canvas or pixels)."""
    if verts is None or j3d is None or j2d is None:
        return None
    v = np.asarray(verts, dtype=np.float64)
    j3 = np.asarray(j3d, dtype=np.float64)
    j2 = np.asarray(j2d, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] < 3 or j3.ndim != 2 or j2.ndim != 2:
        return None
    n = min(len(j3), len(j2))
    if n < 8:
        return None
    j3, j2 = j3[:n], j2[:n]
    ok = (
        np.isfinite(j3).all(axis=1)
        & np.isfinite(j2).all(axis=1)
        & (j3[:, 2] > 1e-4)
    )
    if int(ok.sum()) < 8:
        return None
    xz = j3[ok, 0] / j3[ok, 2]
    yz = j3[ok, 1] / j3[ok, 2]
    u, vv = j2[ok, 0], j2[ok, 1]
    Ax = np.stack([xz, np.ones_like(xz)], axis=1)
    Ay = np.stack([yz, np.ones_like(yz)], axis=1)
    bx, *_ = np.linalg.lstsq(Ax, u, rcond=None)
    by, *_ = np.linalg.lstsq(Ay, vv, rcond=None)
    f = float(0.5 * (bx[0] + by[0]))
    if abs(f) < 1e-3:
        return None
    cx, cy = float(bx[1]), float(by[1])
    z = np.clip(v[:, 2], 1e-4, None)
    return np.stack([f * v[:, 0] / z + cx, f * v[:, 1] / z + cy], axis=1)


def _rasterize_point_sil(
    pts_list: Sequence[np.ndarray],
    h: int,
    w: int,
    *,
    radius: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Scatter projected verts then one dilate (no per-vert disk loops)."""
    r = radius if radius is not None else max(3, min(14, int(0.012 * max(h, w))))
    mask = np.zeros((h, w), dtype=bool)
    n_pts = 0
    for pts in pts_list:
        if pts is None or len(pts) == 0:
            continue
        arr = np.asarray(pts, dtype=np.float64)
        step = max(1, len(arr) // 2500)
        xs = np.rint(arr[::step, 0]).astype(np.int32)
        ys = np.rint(arr[::step, 1]).astype(np.int32)
        ok = (
            np.isfinite(arr[::step, 0])
            & np.isfinite(arr[::step, 1])
            & (xs >= 0)
            & (xs < w)
            & (ys >= 0)
            & (ys < h)
        )
        if not np.any(ok):
            continue
        mask[ys[ok], xs[ok]] = True
        n_pts += int(ok.sum())
    if n_pts < 30:
        return None
    return _dilate_bool(mask, r)


def structure_overcount_score(
    img_path: Optional[Path],
    n_expected: int,
    *,
    j2ds: Optional[Sequence[Any]] = None,
    j3ds: Optional[Sequence[Any]] = None,
    verts: Optional[Sequence[Any]] = None,
    yolo_weights: Optional[Path] = None,
    img_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Unified extra-structure: image person-mass not explained by fitted humans.

    Prefer **mesh silhouette** (project Multi-HMR verts via j3d↔j2d); fall back
    to thick skeleton+torso. Then:

      leftover = person-fg ∧ ¬explained
      orphan   = leftover blobs that do not touch explained
      P_extra  ← calibrated leftover_frac + orphan blobs

    Skin-tone → VQA. ``yolo_weights`` ignored (API compat).
    """
    del yolo_weights  # unified path; kept for call-site compat
    if not img_path or not Path(img_path).is_file() or int(n_expected) <= 0:
        return {
            "score": None,
            "available": False,
            "leftover_frac": None,
            "n_leftover_blobs": None,
            "note": "no_image_or_n",
        }
    if not j2ds:
        return {
            "score": None,
            "available": False,
            "leftover_frac": None,
            "n_leftover_blobs": None,
            "note": "no_j2d",
        }
    try:
        from PIL import Image
    except Exception as e:
        return {
            "score": None,
            "available": False,
            "leftover_frac": None,
            "n_leftover_blobs": None,
            "note": f"deps:{e}",
        }

    im = Image.open(img_path).convert("RGB")
    w0, h0 = im.size
    # ≤768 long side — full-res dilate/splat previously OOM-killed the process
    max_side = 768
    scale = min(1.0, float(max_side) / float(max(w0, h0)))
    if scale < 1.0:
        w, h = int(round(w0 * scale)), int(round(h0 * scale))
        im = im.resize((w, h), Image.BILINEAR)
    else:
        w, h = w0, h0
    rgb = np.asarray(im, dtype=np.float32) / 255.0

    # Multi-HMR j2d is on padded square canvas → map back to original pixels
    j2ds_px = _j2ds_to_original_pixels(j2ds, (w0, h0), img_size=img_size)
    if not j2ds_px:
        return {
            "score": None,
            "available": False,
            "leftover_frac": None,
            "n_leftover_blobs": None,
            "note": "j2d_unusable",
        }
    if scale < 1.0:
        j2ds_px = [p * scale for p in j2ds_px]

    method = "skeleton_torso"
    explained = None
    # Mesh sil optional; stick+torso is enough for orphan detection and safer.
    # Prefer skeleton first for stability; mesh only if skeleton fails.
    explained = _draw_explained_skeleton(j2ds_px, h, w)
    if explained is None and verts and j3ds and len(verts) == len(j2ds):
        proj_px: List[np.ndarray] = []
        for v, j3, j2_raw, j2_px in zip(verts, j3ds, j2ds, j2ds_px):
            j2_arr = _as_j2d_arr(j2_raw)
            if j2_arr is None or j3 is None or v is None:
                continue
            canvas = int(img_size) if img_size else None
            if canvas and float(np.nanmax(np.abs(j2_arr))) > canvas * 1.25:
                canvas = None
            if canvas:
                pv = _project_verts_with_joints(
                    np.asarray(v, dtype=np.float64),
                    np.asarray(j3, dtype=np.float64),
                    j2_arr,
                )
                if pv is not None:
                    pv = j2d_canvas_to_original(pv, (w0, h0), canvas)
                    if scale < 1.0:
                        pv = pv * scale
            else:
                pv = _project_verts_with_joints(
                    np.asarray(v, dtype=np.float64),
                    np.asarray(j3, dtype=np.float64),
                    j2_px,
                )
            if pv is not None:
                proj_px.append(pv)
        explained = _rasterize_point_sil(proj_px, h, w)
        if explained is not None:
            method = "mesh_sil"

    if explained is None:
        return {
            "score": None,
            "available": False,
            "leftover_frac": None,
            "n_leftover_blobs": None,
            "note": "skeleton_empty",
            "j2d_max": float(max(np.nanmax(np.abs(p)) for p in j2ds_px)),
            "img_size_arg": img_size,
        }

    fg = _foreground_mask(rgb)
    ys, xs = np.where(explained)
    if len(xs) < 8:
        return {
            "score": None,
            "available": False,
            "leftover_frac": None,
            "n_leftover_blobs": None,
            "note": "explained_too_small",
        }
    # Bbox ROI (not huge dilate): include nearby extra limbs without OOM.
    pad = max(24, int(0.22 * max(h, w)))
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad + 1)
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad + 1)
    roi = np.zeros_like(fg, dtype=bool)
    roi[y0:y1, x0:x1] = True

    fg_roi = fg & roi
    leftover = fg_roi & (~explained)
    fg_area = float(fg_roi.sum()) + 1e-6
    leftover_frac = float(leftover.sum()) / fg_area

    # Orphans = leftover islands NOT touching explained.
    # touch≈2% side: reconnect body/clothing gaps; still leave detached hands.
    touch_r = max(4, min(16, int(0.022 * max(h, w))))
    explained_touch = _dilate_bool(explained, touch_r)
    n_blobs = 0
    max_blob_frac = 0.0
    orphan_area = 0.0
    lab, nlab = _label_components(leftover)
    for i in range(1, nlab + 1):
        comp = lab == i
        a = float(comp.sum())
        frac_i = a / fg_area
        # ignore crumbs (GT false positives ~1–2%); keep limb-scale extras
        if frac_i < 0.025:
            continue
        if bool(np.any(comp & explained_touch)):
            continue
        n_blobs += 1
        orphan_area += a
        max_blob_frac = max(max_blob_frac, frac_i)
    orphan_frac = orphan_area / fg_area

    # Soft mapping — old floors (0.55/0.80) turned GT orphan≈0.024 into P≈1.
    # leftover is secondary; orphan blobs are the main overcount signal.
    base = 0.48
    span = 0.32
    p_left = float(np.clip((leftover_frac - base) / span, 0.0, 1.0)) * 0.4
    p_blob = float(np.clip((max_blob_frac - 0.025) / 0.055, 0.0, 1.0))
    if n_blobs >= 2:
        p_blob = max(p_blob, min(0.65, 0.28 * n_blobs))
    p = float(max(p_left, p_blob))
    score = float(1.0 - p)
    return {
        "score": score,
        "available": True,
        "leftover_frac": float(leftover_frac),
        "orphan_frac": float(orphan_frac),
        "n_leftover_blobs": int(n_blobs),
        "max_blob_frac": float(max_blob_frac),
        "P_extra_area": p_left,
        "P_extra_blob": p_blob,
        "method": method,
        "note": f"unexplained_via_{method}",
        "work_scale": float(scale),
        "n_hands": None,
        "n_torso": None,
    }

def compose_anat_score(
    *,
    s_residual: Optional[float] = None,
    s_overcount: Optional[float] = None,
    s_scale: Optional[float] = None,
    s_ownership: Optional[float] = None,
    s_part_mesh: Optional[float] = None,
    s_person: Optional[float] = None,
    s_abhuman: Optional[float] = None,
    under_detect: bool = False,
    recon_fail: bool = False,
    n_detected_raw: Optional[int] = None,
    n_expected: Optional[int] = None,
    # anat_v3.1 defaults: stronger detect/struct for attached fused limbs
    w_extra: float = 0.40,
    w_resid: float = 0.20,
    w_struct: float = 0.20,
    w_detect: float = 0.20,
    overdetect_ok: float = 2.0,
    overdetect_span: float = 1.5,
    ownership_amp: float = 2.0,
    # Attached-extra signature: high leftover area but no orphan blobs + mild overdetect
    leftover_frac: Optional[float] = None,
    n_leftover_blobs: Optional[int] = None,
    attached_extra_leftover: float = 0.70,
    attached_extra_min_ratio: float = 2.0,
    attached_extra_p_floor: float = 0.55,
    protocol: str = "anat_v3.1",
) -> Dict[str, Any]:
    """Anatomy score (additive main form; penalty algebraically equivalent).

    anat_v3.1 Main: ``S = 0.40 s_extra + 0.20 s_resid + 0.20 s_struct + 0.20 s_detect``
    with ``s = 1 - P``.

    **A1**: soft ``P_detect`` from over-detection
    ``clip((n_raw/n_exp - overdetect_ok) / overdetect_span, 0, 1)`` (plus hard under-detect);
    plus a floor when leftover looks like *attached* extras (high leftover_frac, 0 blobs).
    **A2**: larger ``w_struct`` and amplified ownership penalty
    ``min(1, (1-s_own)*ownership_amp)``.

    **Must-catch**: unexplained structure (`s_overcount` = fg−skeleton).
    Skin → VQA.
    """
    if recon_fail:
        return {
            "S_anat_mesh": 0.0,
            "P_anat_extra": 1.0,
            "P_anat_resid": 1.0,
            "P_anat_struct": 0.0,
            "P_anat_detect": 1.0,
            "anat_formula": "recon_fail→0",
            "anat_protocol": protocol,
            "w_extra": w_extra,
            "w_resid": w_resid,
            "w_struct": w_struct,
            "w_detect": w_detect,
        }

    we, wr, ws, wd = float(w_extra), float(w_resid), float(w_struct), float(w_detect)

    # A2: amplify ownership; keep other struct cues as 1-s
    struct_penalties: List[float] = []
    if s_scale is not None:
        struct_penalties.append(float(1.0 - float(s_scale)))
    if s_ownership is not None:
        struct_penalties.append(
            float(min(1.0, (1.0 - float(s_ownership)) * float(ownership_amp)))
        )
    if s_part_mesh is not None:
        struct_penalties.append(float(1.0 - float(s_part_mesh)))
    if s_abhuman is not None:
        struct_penalties.append(float(1.0 - float(s_abhuman)))
    p_struct = float(max(struct_penalties)) if struct_penalties else 0.0
    if p_struct < 0.02:
        p_struct = 0.0

    # A1: under-detect hard fail + soft over-detect from raw person count
    ratio: Optional[float] = None
    if under_detect:
        p_detect = 1.0
    elif (
        n_detected_raw is not None
        and n_expected is not None
        and int(n_expected) > 0
    ):
        ratio = float(n_detected_raw) / float(n_expected)
        p_detect = float(
            np.clip(
                (ratio - float(overdetect_ok)) / max(float(overdetect_span), 1e-6),
                0.0,
                1.0,
            )
        )
    else:
        p_detect = 0.0

    # Attached fused limbs: leftover mass high but not fragmented into orphan blobs
    if (
        leftover_frac is not None
        and n_leftover_blobs is not None
        and float(leftover_frac) >= float(attached_extra_leftover)
        and int(n_leftover_blobs) == 0
        and ratio is not None
        and float(ratio) >= float(attached_extra_min_ratio)
    ):
        p_detect = float(max(p_detect, float(attached_extra_p_floor)))

    p_extra = (1.0 - float(s_overcount)) if s_overcount is not None else None
    if s_residual is not None:
        p_resid = float(1.0 - float(s_residual))
    else:
        p_resid = float(1.0 - float(s_person)) if s_person is not None else None

    # redistribute if overcount backend missing
    if p_extra is None:
        wr = wr + we * 0.5
        ws = ws + we * 0.5
        we = 0.0
        p_extra = 0.0
        formula = "no_overcount_backend: resid+struct+detect"
    else:
        formula = (
            f"{protocol}: 0.40s_extra+0.20s_resid+0.20s_struct+0.20s_detect "
            f"(overdetect_ok={overdetect_ok}, ownership_amp={ownership_amp})"
        )

    if p_resid is None:
        we = we + wr * 0.5
        ws = ws + wr * 0.5
        wr = 0.0
        p_resid = 0.0

    s = float(
        np.clip(
            1.0 - we * p_extra - wr * p_resid - ws * p_struct - wd * p_detect,
            0.0,
            1.0,
        )
    )
    return {
        "S_anat_mesh": s,
        "P_anat_extra": float(p_extra),
        "P_anat_resid": float(p_resid),
        "P_anat_struct": float(p_struct),
        "P_anat_detect": float(p_detect),
        "anat_formula": formula,
        "anat_protocol": protocol,
        "w_extra": we,
        "w_resid": wr,
        "w_struct": ws,
        "w_detect": wd,
        "n_detected_raw": n_detected_raw,
        "n_expected": n_expected,
        "detect_ratio": (
            float(n_detected_raw) / float(n_expected)
            if n_detected_raw is not None and n_expected not in (None, 0)
            else None
        ),
    }


def compose_anat_score_v4(
    *,
    s_residual: Optional[float] = None,
    s_scale: Optional[float] = None,
    s_ownership: Optional[float] = None,
    s_part_mesh: Optional[float] = None,
    s_person: Optional[float] = None,
    s_abhuman: Optional[float] = None,
    under_detect: bool = False,
    recon_fail: bool = False,
    n_detected_raw: Optional[int] = None,
    n_expected: Optional[int] = None,
    leftover_frac: Optional[float] = None,
    n_leftover_blobs: Optional[int] = None,
    orphan_frac: Optional[float] = None,
    pen_inside_ratio: Optional[float] = None,
    p_fuse: Optional[float] = None,
    # Weights: attach / orphan / struct / resid
    w_attach: float = 0.40,
    w_orphan: float = 0.20,
    w_struct: float = 0.25,
    w_resid: float = 0.15,
    ownership_amp: float = 2.0,
    # leftover_band: 0 at leftover_ok, 1 at leftover_bad
    leftover_ok: float = 0.55,
    leftover_bad: float = 0.80,
    # fuse_band from pen_inside (preferred) or P_fuse
    inside_ok: float = 0.20,
    inside_bad: float = 0.50,
    # gated over-detect (only with attach evidence)
    overdetect_ok: float = 2.5,
    overdetect_span: float = 2.0,
    gate_leftover: float = 0.60,
    gate_inside: float = 0.25,
    gate_ownership: float = 0.98,
    # orphan: blobs / orphan_frac
    orphan_blob_ok: float = 0.0,
    orphan_blob_bad: float = 3.0,
    orphan_frac_ok: float = 0.02,
    orphan_frac_bad: float = 0.15,
    # leftover alone is often clothing; full strength only when attached (blobs==0)
    leftover_alone_scale: float = 0.20,
    fuse_alone_scale: float = 0.25,
    attached_blobs_max: int = 0,
    attached_leftover_min: float = 0.65,
    # extreme over-detect (ungated mild floor) for fragment storms without leftover
    extreme_ratio: float = 3.5,
    extreme_span: float = 1.5,
    extreme_cap: float = 0.85,
    protocol: str = "anat_v4_exp",
) -> Dict[str, Any]:
    """Experimental Anatomy compose (offline; does not replace anat_v3/v3.1).

    ``S = 1 - w_a P_attach - w_o P_orphan - w_s P_struct - w_r P_resid``

    * **P_attach** — interaction of leftover / fuse / gated over-detect:
      leftover & fuse are *full strength* only for attached signature
      (high leftover + zero orphan blobs); alone they are down-scaled so
      clothing clutter on clean edits is not over-penalized.
    * **P_orphan** — free-floating leftover blobs / orphan mass.
    * **P_struct** — ownership (amplified) + scale + part-mesh + optional AbHuman.
    * **P_resid** — explain-residual (down-weighted vs v3).

    Returns ``S_anat_v4``. Does **not** mutate paper judgments.
    """
    if recon_fail:
        return {
            "S_anat_v4": 0.0,
            "S_anat_mesh": 0.0,
            "P_anat_attach": 1.0,
            "P_anat_orphan": 0.0,
            "P_anat_struct": 0.0,
            "P_anat_resid": 1.0,
            "P_anat_leftover": 1.0,
            "P_anat_fuse": 0.0,
            "P_anat_overdetect_gated": 1.0,
            "anat_formula": "recon_fail→0",
            "anat_protocol": protocol,
            "w_attach": w_attach,
            "w_orphan": w_orphan,
            "w_struct": w_struct,
            "w_resid": w_resid,
        }

    def _band_high_bad(val: Optional[float], ok: float, bad: float) -> float:
        if val is None:
            return 0.0
        v = float(val)
        if v != v:  # NaN
            return 0.0
        if bad <= ok:
            return 1.0 if v >= bad else 0.0
        return float(np.clip((v - ok) / max(bad - ok, 1e-6), 0.0, 1.0))

    # --- leftover / fuse bands ---
    p_leftover_raw = _band_high_bad(leftover_frac, leftover_ok, leftover_bad)
    if pen_inside_ratio is not None:
        p_fuse_raw = _band_high_bad(pen_inside_ratio, inside_ok, inside_bad)
    elif p_fuse is not None:
        p_fuse_raw = float(np.clip(float(p_fuse), 0.0, 1.0))
    else:
        p_fuse_raw = 0.0

    blobs = int(n_leftover_blobs) if n_leftover_blobs is not None else None
    attached = (
        leftover_frac is not None
        and blobs is not None
        and int(blobs) <= int(attached_blobs_max)
        and float(leftover_frac) >= float(attached_leftover_min)
    )
    # Full leftover/fuse only on attached (blobs==0 + high leftover); else down-scale
    if attached:
        p_leftover = float(p_leftover_raw)
        p_fuse_c = float(p_fuse_raw)
    else:
        p_leftover = float(leftover_alone_scale) * float(p_leftover_raw)
        p_fuse_c = float(fuse_alone_scale) * float(p_fuse_raw)

    ratio: Optional[float] = None
    p_over_raw = 0.0
    if under_detect:
        p_over_raw = 1.0
        ratio = 0.0
    elif (
        n_detected_raw is not None
        and n_expected is not None
        and int(n_expected) > 0
    ):
        ratio = float(n_detected_raw) / float(n_expected)
        p_over_raw = float(
            np.clip(
                (ratio - float(overdetect_ok)) / max(float(overdetect_span), 1e-6),
                0.0,
                1.0,
            )
        )

    # Gate: over-detect with attach evidence OR attached signature
    own = float(s_ownership) if s_ownership is not None else 1.0
    gate = bool(attached) or bool(under_detect)
    if not gate:
        if leftover_frac is not None and float(leftover_frac) >= float(gate_leftover):
            gate = True
        if pen_inside_ratio is not None and float(pen_inside_ratio) >= float(gate_inside):
            gate = True
        if s_ownership is not None and own < float(gate_ownership):
            gate = True
        if blobs is not None and int(blobs) <= int(attached_blobs_max):
            # zero-blob + mild leftover still gates over-detect
            if leftover_frac is not None and float(leftover_frac) >= 0.45:
                gate = True
    p_over_gated = float(p_over_raw if gate else 0.0)

    # Ungated mild floor for extreme fragment storms (e.g. UNO n_raw≫n_exp, low leftover)
    p_extreme = 0.0
    if ratio is not None and float(ratio) >= float(extreme_ratio):
        p_extreme = float(
            min(
                float(extreme_cap),
                np.clip(
                    (float(ratio) - float(extreme_ratio))
                    / max(float(extreme_span), 1e-6),
                    0.0,
                    1.0,
                )
                * float(extreme_cap),
            )
        )

    p_attach = float(max(p_leftover, p_fuse_c, p_over_gated, p_extreme))

    # --- P_orphan ---
    p_blob = 0.0
    if n_leftover_blobs is not None:
        p_blob = _band_high_bad(
            float(n_leftover_blobs), orphan_blob_ok, orphan_blob_bad
        )
    p_orph_frac = _band_high_bad(orphan_frac, orphan_frac_ok, orphan_frac_bad)
    p_orphan = float(max(p_blob, p_orph_frac))

    # --- P_struct ---
    struct_penalties: List[float] = []
    if s_scale is not None:
        struct_penalties.append(float(1.0 - float(s_scale)))
    if s_ownership is not None:
        struct_penalties.append(
            float(min(1.0, (1.0 - float(s_ownership)) * float(ownership_amp)))
        )
    if s_part_mesh is not None:
        struct_penalties.append(float(1.0 - float(s_part_mesh)))
    if s_abhuman is not None:
        struct_penalties.append(float(1.0 - float(s_abhuman)))
    p_struct = float(max(struct_penalties)) if struct_penalties else 0.0
    if p_struct < 0.02:
        p_struct = 0.0

    # --- P_resid ---
    if s_residual is not None:
        p_resid = float(1.0 - float(s_residual))
    elif s_person is not None:
        p_resid = float(1.0 - float(s_person))
    else:
        p_resid = 0.0

    wa, wo, ws, wr = (
        float(w_attach),
        float(w_orphan),
        float(w_struct),
        float(w_resid),
    )
    s = float(
        np.clip(
            1.0 - wa * p_attach - wo * p_orphan - ws * p_struct - wr * p_resid,
            0.0,
            1.0,
        )
    )
    formula = (
        f"{protocol}: 1-({wa:.2f}P_attach+{wo:.2f}P_orphan+"
        f"{ws:.2f}P_struct+{wr:.2f}P_resid); "
        f"leftover_ok/bad={leftover_ok}/{leftover_bad}; "
        f"gated_overdetect={overdetect_ok}+span{overdetect_span}"
    )
    return {
        "S_anat_v4": s,
        "S_anat_mesh": s,  # convenience for summarize tools; callers must not overwrite mesh_v3
        "P_anat_attach": float(p_attach),
        "P_anat_orphan": float(p_orphan),
        "P_anat_struct": float(p_struct),
        "P_anat_resid": float(p_resid),
        "P_anat_leftover": float(p_leftover),
        "P_anat_fuse": float(p_fuse_c),
        "P_anat_overdetect_gated": float(p_over_gated),
        "P_anat_overdetect_raw": float(p_over_raw),
        "P_anat_extreme_overdetect": float(p_extreme),
        "overdetect_gated": bool(gate),
        "attached_signature": bool(attached),
        "anat_formula": formula,
        "anat_protocol": protocol,
        "w_attach": wa,
        "w_orphan": wo,
        "w_struct": ws,
        "w_resid": wr,
        "n_detected_raw": n_detected_raw,
        "n_expected": n_expected,
        "detect_ratio": ratio,
        "leftover_ok": float(leftover_ok),
        "leftover_bad": float(leftover_bad),
        "inside_ok": float(inside_ok),
        "inside_bad": float(inside_bad),
        "overdetect_ok": float(overdetect_ok),
        "overdetect_span": float(overdetect_span),
        "gate_leftover": float(gate_leftover),
        "gate_inside": float(gate_inside),
        "gate_ownership": float(gate_ownership),
        "leftover_alone_scale": float(leftover_alone_scale),
        "fuse_alone_scale": float(fuse_alone_scale),
        "attached_leftover_min": float(attached_leftover_min),
        "extreme_ratio": float(extreme_ratio),
    }


def abhuman_score(
    img_path: Optional[Path],
    *,
    weights: Optional[Path] = None,
    conf: float = 0.25,
) -> Dict[str, Any]:
    """Optional AbHuman/YOLO anatomical-anomaly detector.

    If ultralytics + weights are present, score = 1 - clip(n_anom/5).
    Otherwise available=False (does not poison the aggregate).
    """
    if not img_path or not Path(img_path).is_file():
        return {"score": None, "available": False, "n_anom": None, "note": "no_image"}
    wpath = Path(weights) if weights else Path.home() / "models" / "abhuman" / "best.pt"
    if not wpath.is_file():
        return {
            "score": None,
            "available": False,
            "n_anom": None,
            "note": f"missing_weights:{wpath}",
        }
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        return {"score": None, "available": False, "n_anom": None, "note": f"no_ultralytics:{e}"}
    try:
        model = YOLO(str(wpath))
        res = model.predict(str(img_path), conf=conf, verbose=False)
        n = 0
        if res:
            boxes = getattr(res[0], "boxes", None)
            n = int(len(boxes)) if boxes is not None else 0
        score = float(max(0.0, 1.0 - n / 5.0))
        return {"score": score, "available": True, "n_anom": n, "note": "ok"}
    except Exception as e:
        return {"score": None, "available": False, "n_anom": None, "note": repr(e)}


def aggregate_extended_anat(
    *,
    per_person: List[Dict[str, Any]],
    scene: Dict[str, Any],
    under_detect: bool = False,
    recon_fail: bool = False,
    n_detected_raw: Optional[int] = None,
    n_expected: Optional[int] = None,
) -> Dict[str, Any]:
    """Penalty Anat: overcount (hands/torsos) first, then residual."""
    person_scores = [
        p["S_anat_mesh"] for p in per_person if p.get("S_anat_mesh") is not None
    ]
    s_person = float(np.mean(person_scores)) if person_scores else None

    def _sc(key: str) -> Optional[float]:
        block = scene.get(key) or {}
        if block.get("available") and block.get("score") is not None:
            return float(block["score"])
        return None

    part_scores = [
        float(p["part_mesh_self_score"])
        for p in per_person
        if p.get("part_mesh_self_score") is not None
    ]
    s_part = float(np.mean(part_scores)) if part_scores else None

    over = scene.get("structure_overcount") or {}
    composed = compose_anat_score(
        s_residual=_sc("explain_residual"),
        s_overcount=_sc("structure_overcount"),
        s_scale=_sc("cross_person_scale"),
        s_ownership=_sc("ownership"),
        s_part_mesh=s_part,
        s_person=s_person,
        s_abhuman=_sc("abhuman"),
        under_detect=under_detect,
        recon_fail=recon_fail,
        n_detected_raw=n_detected_raw,
        n_expected=n_expected,
        leftover_frac=over.get("leftover_frac"),
        n_leftover_blobs=over.get("n_leftover_blobs"),
    )
    return {
        **composed,
        "S_anat_person": float(s_person) if s_person is not None else None,
        "S_anat_scene": _sc("explain_residual"),
        "S_anat_overcount": _sc("structure_overcount"),
        "scene_detail": scene,
    }