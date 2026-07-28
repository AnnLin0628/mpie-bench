#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mesh-space geometric metrics for MPIE eval protocol v3.

Fast path uses scipy.cKDTree on subsampled vertices — no trimesh ProximityQuery
(that path can hang minutes on SMPL-X meshes).

Person selection (top-k by prompt R#) is done in score_mesh_v3 before calling
score_humans — this module only scores the humans it is given.

Count is diagnostic only; main-table Count stays VLM.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# SMPL-X body joint indices (smplx.joint_names.JOINT_NAMES prefix)
# Used for bone-length / symmetry when j3d is available.
_J = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "spine1": 3,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
    # distal index phalanx (tip proxy); NOT index1 (25/40)
    "left_index": 27,
    "right_index": 42,
}

# Anthropometric ratio bands (lo, hi) for adult-like proportions.
_RATIO_BANDS: Dict[str, Tuple[float, float]] = {
    "upper_arm_over_forearm": (0.85, 1.45),
    "thigh_over_shin": (0.90, 1.50),
    "upper_arm_over_thigh": (0.55, 0.95),
    "forearm_over_shin": (0.40, 0.95),
    "finger_over_shin": (0.12, 0.50),
    "torso_over_leg": (0.55, 1.10),
}

# Per-joint axis-angle magnitude caps (degrees) for body joints after global orient.
# Uniform 150° was too loose — Multi-HMR almost never exceeds it.
_JOINT_MAX_DEG = {
    # index in rotvec joints AFTER skipping global (so 0 = left_hip in body-only
    # addressing). We apply by absolute joint index in full rotvec instead:
    1: 90,   # left_hip
    2: 90,   # right_hip
    3: 60,   # spine1
    4: 150,  # left_knee (flexion large)
    5: 150,  # right_knee
    6: 60,   # spine2
    7: 70,   # left_ankle
    8: 70,   # right_ankle
    9: 60,   # spine3
    12: 80,  # neck
    15: 80,  # head
    16: 120, # left_shoulder
    17: 120, # right_shoulder
    18: 150, # left_elbow
    19: 150, # right_elbow
    20: 90,  # left_wrist
    21: 90,  # right_wrist
}
_DEFAULT_JOINT_MAX_DEG = 120.0


def _as_float_verts(v: Any) -> np.ndarray:
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"verts must be (V,3), got {v.shape}")
    return v


def _as_float_j3d(j: Any) -> Optional[np.ndarray]:
    if j is None:
        return None
    if hasattr(j, "detach"):
        j = j.detach().cpu().numpy()
    j = np.asarray(j, dtype=np.float64)
    if j.ndim != 2 or j.shape[1] != 3 or j.shape[0] < 22:
        return None
    return j


def _subsample(verts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    if len(verts) <= n:
        return verts
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(verts), size=n, replace=False)
    return verts[idx]


def _seg_len(j: np.ndarray, a: str, b: str) -> Optional[float]:
    ia, ib = _J.get(a), _J.get(b)
    if ia is None or ib is None or ia >= len(j) or ib >= len(j):
        return None
    d = float(np.linalg.norm(j[ia] - j[ib]))
    if not np.isfinite(d):
        return None
    return d  # allow 0 (coincident joints = hard collision)


def _ratio_score(val: float, lo: float, hi: float) -> float:
    if lo <= val <= hi:
        return 1.0
    # relative excess beyond nearest band edge
    if val < lo:
        excess = (lo - val) / max(lo, 1e-6)
    else:
        excess = (val - hi) / max(hi, 1e-6)
    return float(max(0.0, 1.0 - excess))


def bone_proportion_score(j3d: Optional[Any]) -> Dict[str, Any]:
    """Anthropometric limb ratios + L/R length symmetry (static-image Anat).

    Note: SMPL-X bone lengths are mostly shape-driven and stay near-human, so
    this rarely alone separates models — still useful for extreme shape/fit fails.
    Missing j3d → available=False (do NOT pretend score=1).
    """
    j = _as_float_j3d(j3d)
    if j is None:
        return {"score": None, "available": False, "ratios": {}, "symmetry": {}}

    def mean_pair(la, lb, ra, rb):
        l = _seg_len(j, la, lb)
        r = _seg_len(j, ra, rb)
        if l is None or r is None:
            return None, None, None
        return 0.5 * (l + r), l, r

    upper_arm, ua_l, ua_r = mean_pair(
        "left_shoulder", "left_elbow", "right_shoulder", "right_elbow"
    )
    forearm, fa_l, fa_r = mean_pair(
        "left_elbow", "left_wrist", "right_elbow", "right_wrist"
    )
    thigh, th_l, th_r = mean_pair("left_hip", "left_knee", "right_hip", "right_knee")
    shin, sh_l, sh_r = mean_pair("left_knee", "left_ankle", "right_knee", "right_ankle")
    torso = _seg_len(j, "pelvis", "neck")
    leg = (thigh + shin) if (thigh is not None and shin is not None) else None

    finger = None
    fl = _seg_len(j, "left_wrist", "left_index")
    fr = _seg_len(j, "right_wrist", "right_index")
    if fl is not None and fr is not None:
        finger = 0.5 * (fl + fr)
    elif fl is not None:
        finger = fl
    elif fr is not None:
        finger = fr

    raw_ratios: Dict[str, float] = {}
    if upper_arm and forearm:
        raw_ratios["upper_arm_over_forearm"] = upper_arm / forearm
    if thigh and shin:
        raw_ratios["thigh_over_shin"] = thigh / shin
    if upper_arm and thigh:
        raw_ratios["upper_arm_over_thigh"] = upper_arm / thigh
    if forearm and shin:
        raw_ratios["forearm_over_shin"] = forearm / shin
    if finger and shin:
        raw_ratios["finger_over_shin"] = finger / shin
    if torso and leg:
        raw_ratios["torso_over_leg"] = torso / leg

    ratio_scores = []
    for name, val in raw_ratios.items():
        lo, hi = _RATIO_BANDS[name]
        ratio_scores.append(_ratio_score(val, lo, hi))

    sym: Dict[str, float] = {}
    for key, pair in {
        "upper_arm": (ua_l, ua_r),
        "forearm": (fa_l, fa_r),
        "thigh": (th_l, th_r),
        "shin": (sh_l, sh_r),
    }.items():
        a, b = pair
        if a is None or b is None:
            continue
        rel = abs(a - b) / max(0.5 * (a + b), 1e-6)
        sym[key] = float(rel)
        ratio_scores.append(float(max(0.0, 1.0 - rel / 0.18)))

    if not ratio_scores:
        return {"score": None, "available": False, "ratios": {}, "symmetry": sym}
    return {
        "score": float(np.mean(ratio_scores)),
        "available": True,
        "ratios": {k: float(v) for k, v in raw_ratios.items()},
        "symmetry": sym,
    }


def joint_limit_score(body_pose: Optional[Any], max_deg: float = 150.0) -> float:
    """Axis-angle magnitude vs per-joint anatomical caps.

    Multi-HMR ``rotvec`` is [global_orient + body…] — **skip joint 0** (global).
    Old code used a flat 150° on all joints → nearly always 1.0 (no discrimination).
    """
    if body_pose is None:
        return 1.0
    if hasattr(body_pose, "detach"):
        body_pose = body_pose.detach().cpu().numpy()
    p = np.asarray(body_pose, dtype=np.float64).reshape(-1)
    if p.size % 3 != 0:
        return 1.0
    aa = p.reshape(-1, 3)
    if len(aa) <= 1:
        return 1.0
    # skip global orient
    aa = aa[1:]
    angles = np.linalg.norm(aa, axis=1) * 180.0 / np.pi
    # map back to absolute joint indices (1..); only score body joints we care about
    viol = []
    for i, ang in enumerate(angles):
        j_abs = i + 1
        if j_abs > 21:
            # hands/face: softer, skip to avoid noise
            continue
        cap = float(_JOINT_MAX_DEG.get(j_abs, _DEFAULT_JOINT_MAX_DEG))
        # soft excess beyond cap
        if ang <= cap:
            viol.append(0.0)
        else:
            viol.append(min(1.0, (ang - cap) / max(cap, 1.0)))
    if not viol:
        # fallback: flat threshold on body joints only
        body = angles[:21] if len(angles) >= 21 else angles
        frac = float(np.mean(body > float(max_deg))) if len(body) else 0.0
        return float(max(0.0, 1.0 - frac))
    return float(max(0.0, 1.0 - float(np.mean(viol))))


def self_collision_proxy(j3d: Optional[Any]) -> Dict[str, Any]:
    """Strict self-intersection proxy — avoid punishing normal contact poses.

    BUGFIX: old wrist–pelvis <8cm fired on real hugs → GT Anat < Flux.
    Only flag near-impossible configurations (≈ inside torso core).
    """
    j = _as_float_j3d(j3d)
    if j is None:
        return {"score": None, "available": False, "n_bad": 0}

    # Extremely close to torso core / opposite limb through body
    checks = [
        ("left_wrist", "spine1", 0.04),
        ("right_wrist", "spine1", 0.04),
        ("left_elbow", "spine1", 0.05),
        ("right_elbow", "spine1", 0.05),
        ("left_wrist", "right_shoulder", 0.05),
        ("right_wrist", "left_shoulder", 0.05),
        ("left_ankle", "right_hip", 0.04),
        ("right_ankle", "left_hip", 0.04),
    ]
    bad = 0
    tested = 0
    for a, b, thr in checks:
        la = _seg_len(j, a, b)
        if la is None:
            continue
        tested += 1
        if la < thr:
            bad += 1
    if tested == 0:
        return {"score": None, "available": False, "n_bad": 0}
    frac = bad / tested
    return {"score": float(max(0.0, 1.0 - frac)), "available": True, "n_bad": bad}


def shape_plausibility_score(betas: Optional[Any]) -> Dict[str, Any]:
    """Penalize extreme SMPL-X shape coefficients (|β| ≫ typical)."""
    if betas is None:
        return {"score": None, "available": False, "beta_abs_mean": None, "beta_abs_max": None}
    if hasattr(betas, "detach"):
        betas = betas.detach().cpu().numpy()
    b = np.abs(np.asarray(betas, dtype=np.float64).reshape(-1))
    if b.size == 0 or not np.isfinite(b).all():
        return {"score": None, "available": False, "beta_abs_mean": None, "beta_abs_max": None}
    # soft: |β|≤2 OK; linear to 0 by |β|=5
    excess = np.clip((b - 2.0) / 3.0, 0.0, 1.0)
    return {
        "score": float(1.0 - float(np.mean(excess))),
        "available": True,
        "beta_abs_mean": float(np.mean(b)),
        "beta_abs_max": float(np.max(b)),
    }


def anatomy_score(
    body_pose: Optional[Any] = None,
    j3d: Optional[Any] = None,
    betas: Optional[Any] = None,
    *,
    w_joint: float = 0.45,
    w_bone: float = 0.25,
    w_self: float = 0.15,
    w_shape: float = 0.15,
) -> Dict[str, Any]:
    """Aggregate Anat. Missing components redistribute weight (no fake 1.0 fillers)."""
    s_joint = joint_limit_score(body_pose)
    bone = bone_proportion_score(j3d)
    selfc = self_collision_proxy(j3d)
    shape = shape_plausibility_score(betas)

    parts = [("joint", w_joint, s_joint, True)]
    parts.append(("bone", w_bone, bone["score"], bone["available"]))
    parts.append(("self", w_self, selfc["score"], selfc["available"]))
    parts.append(("shape", w_shape, shape["score"], shape["available"]))

    w_sum = 0.0
    acc = 0.0
    for _name, w, s, ok in parts:
        if not ok or s is None:
            continue
        w_sum += w
        acc += w * float(s)
    if w_sum <= 1e-9:
        # only joint always available if pose exists; if pose missing → 1.0 neutral
        s = float(s_joint)
    else:
        s = float(acc / w_sum)

    return {
        "S_anat_mesh": float(s),
        "joint_limit_score": float(s_joint),
        "bone_proportion_score": (
            float(bone["score"]) if bone["score"] is not None else None
        ),
        "self_collision_score": (
            float(selfc["score"]) if selfc["score"] is not None else None
        ),
        "shape_plausibility_score": (
            float(shape["score"]) if shape["score"] is not None else None
        ),
        "bone_detail": bone,
        "self_collision_detail": selfc,
        "shape_detail": shape,
    }


# ---------------------------------------------------------------------------
# Interpenetration
# ---------------------------------------------------------------------------

def penetration_pair(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    faces: Optional[np.ndarray] = None,
    *,
    n_sample: int = 512,
    tau_overlap: float = 0.01,
    compute_volume: bool = True,
    vol_sample: int = 384,
) -> Dict[str, float]:
    """Pairwise Inter geometry.

    Fast proximity proxy (always):
      pen_vert_ratio — frac of subsampled verts within tau_overlap of other cloud
      min_surf_dist  — NN distance between clouds (m)

    Optional volume approx (default on; uses trimesh.contains if available):
      pen_inside_ratio — frac of A samples strictly inside mesh B (sym max)
      pen_volume_m3   — inside_ratio * convex_hull_volume(A)  (intersection proxy)
    """
    va, vb = _as_float_verts(verts_a), _as_float_verts(verts_b)
    sa, sb = _subsample(va, n_sample, 0), _subsample(vb, n_sample, 1)

    try:
        from scipy.spatial import cKDTree
    except Exception as e:
        raise RuntimeError("scipy required for mesh metrics") from e

    tree_b = cKDTree(sb)
    dist_a, _ = tree_b.query(sa, k=1, workers=-1)
    tree_a = cKDTree(sa)
    dist_b, _ = tree_a.query(sb, k=1, workers=-1)

    min_surf_dist = float(min(np.min(dist_a), np.min(dist_b)))
    r_ab = float(np.mean(dist_a <= tau_overlap))
    r_ba = float(np.mean(dist_b <= tau_overlap))
    pen_vert_ratio = float(max(r_ab, r_ba))
    deep = dist_a[dist_a <= tau_overlap]
    pen_depth_mean = float(np.mean(tau_overlap - deep)) if len(deep) else 0.0

    out: Dict[str, Any] = {
        "pen_vert_ratio": pen_vert_ratio,
        "pen_depth_mean": pen_depth_mean,
        "min_surf_dist": min_surf_dist,
        "tau_overlap": tau_overlap,
        "n_sample": n_sample,
        "pen_inside_ratio": 0.0,
        "pen_volume_m3": 0.0,
        "volume_method": "none",
    }

    if compute_volume and faces is not None and len(faces) > 0:
        vol = _penetration_volume_approx(va, vb, faces, n_sample=vol_sample)
        out.update(vol)
    return out


def _mesh_volume_hull(verts: np.ndarray) -> float:
    """Convex-hull volume fallback (SMPL-X body ≈ closed; hull is upper bound)."""
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(_subsample(verts, min(len(verts), 1024), 2))
        return float(max(hull.volume, 0.0))
    except Exception:
        # AABB volume * packing factor
        mn, mx = verts.min(0), verts.max(0)
        return float(np.prod(np.maximum(mx - mn, 1e-6)) * 0.35)


def _contains_ratio(
    points: np.ndarray, verts: np.ndarray, faces: np.ndarray
) -> Tuple[float, str]:
    """Fraction of points inside mesh. Prefer trimesh; else AABB reject → 0."""
    try:
        import trimesh

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        # ray-based contains; check=False avoids repair hangs
        if not mesh.is_watertight:
            mesh = mesh.copy()
            try:
                mesh.fix_normals()
            except Exception:
                pass
        inside = mesh.contains(points)
        return float(np.mean(inside)), "trimesh.contains"
    except Exception:
        return 0.0, "fallback_zero"


def _penetration_volume_approx(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    faces: np.ndarray,
    *,
    n_sample: int = 384,
) -> Dict[str, Any]:
    """Approximate interpenetration volume (proposal interpenetrating volume).

    Method: sample verts of A; fraction strictly inside mesh B (and symmetric);
    V ≈ max(r_ab, r_ba) * min(Vol_hull(A), Vol_hull(B)).

    Not exact Boolean intersection, but correlates with deep fusion and stays
    in tens of ms for SMPL-X when trimesh is available.
    """
    sa = _subsample(verts_a, n_sample, 10)
    sb = _subsample(verts_b, n_sample, 11)
    mn_b, mx_b = verts_b.min(0) - 0.02, verts_b.max(0) + 0.02
    mn_a, mx_a = verts_a.min(0) - 0.02, verts_a.max(0) + 0.02
    in_bb_ab = np.all((sa >= mn_b) & (sa <= mx_b), axis=1)
    in_bb_ba = np.all((sb >= mn_a) & (sb <= mx_a), axis=1)

    method = "none"
    inside_a = np.zeros(len(sa), dtype=bool)
    if np.any(in_bb_ab):
        local, method = _contains_ratio(sa[in_bb_ab], verts_b, faces)
        if method.startswith("trimesh"):
            # re-fetch boolean mask (contains_ratio only returns mean)
            try:
                import trimesh

                mesh = trimesh.Trimesh(vertices=verts_b, faces=faces, process=False)
                inside_a[in_bb_ab] = mesh.contains(sa[in_bb_ab])
            except Exception:
                # approximate: scatter local rate uniformly on in-AABB points
                n_in = int(round(local * int(np.sum(in_bb_ab))))
                idx = np.flatnonzero(in_bb_ab)
                inside_a[idx[:n_in]] = True
        else:
            method = "fallback_zero"
    r_ab = float(np.mean(inside_a))

    inside_b = np.zeros(len(sb), dtype=bool)
    if np.any(in_bb_ba):
        local, m2 = _contains_ratio(sb[in_bb_ba], verts_a, faces)
        if m2.startswith("trimesh"):
            try:
                import trimesh

                mesh = trimesh.Trimesh(vertices=verts_a, faces=faces, process=False)
                inside_b[in_bb_ba] = mesh.contains(sb[in_bb_ba])
                method = "trimesh.contains"
            except Exception:
                n_in = int(round(local * int(np.sum(in_bb_ba))))
                idx = np.flatnonzero(in_bb_ba)
                inside_b[idx[:n_in]] = True
                method = m2
        elif method == "none":
            method = m2
    r_ba = float(np.mean(inside_b))

    pen_inside = float(max(r_ab, r_ba))
    vol_a = _mesh_volume_hull(verts_a)
    vol_b = _mesh_volume_hull(verts_b)
    pen_vol = pen_inside * float(min(vol_a, vol_b))
    return {
        "pen_inside_ratio": pen_inside,
        "pen_volume_m3": float(pen_vol),
        "vol_hull_a": float(vol_a),
        "vol_hull_b": float(vol_b),
        "volume_method": method,
    }


def contact_band_hit(min_surf_dist: float, tau_contact: float = 0.02) -> bool:
    if min_surf_dist != min_surf_dist:  # NaN
        return False
    return float(min_surf_dist) <= float(tau_contact)


# ---------------------------------------------------------------------------
# Prompt-conditioned interaction intent (NOT category labels)
# ---------------------------------------------------------------------------

# Clear physical-contact language in edit prompts (English smoke prompts).
_CONTACT_RES = [
    re.compile(p, re.I)
    for p in (
        r"\bhug(?:s|ging|ged)?\b",
        r"\bembrac(?:e|es|ing|ed)\b",
        r"\bkiss(?:es|ing|ed)?\b",
        r"\bhold(?:s|ing|held)? (?:her |his |their |the )?(?:hands?|hand)\b",
        r"\bholding hands\b",
        r"\bhand[- ]?holds?\b",
        r"\bhandshake\b",
        r"\bshake(?:s|ing|n)? hands?\b",
        r"\bhigh[- ]?fives?\b",
        r"\bfist bumps?\b",
        r"\barm(?:s)? around\b",
        r"\baround (?:her|his|their) (?:shoulder|waist|neck|back)\b",
        r"\bpiggyback\b",
        r"\bcarry(?:ing|ies|ied)?\b",
        r"\blift(?:ing|s|ed)? (?:her|him|them|the)\b",
        r"\bwrestl(?:e|es|ing|ed)\b",
        r"\bgrappl(?:e|es|ing|ed)\b",
        r"\bfight(?:s|ing|ers)?\b",
        r"\bpunch(?:es|ing|ed)?\b",
        r"\bkick(?:s|ing|ed)?\b",
        r"\bpush(?:es|ing|ed)?\b",
        r"\btouch(?:es|ing|ed)?\b",
        r"\blean(?:s|ing|ed)? (?:on|against)\b",
        r"\bhand(?:s)? (?:on|resting on|placed on)\b",
        r"\bgrab(?:s|bing|bed)?\b",
        r"\bclinch(?:es|ing|ed)?\b",
        r"\bwrap(?:s|ping|ped)? (?:an? |her |his |their )?arms?\b",
        r"\bagainst (?:her|his|their) shoulder\b",
        r"\barm gently against\b",
        r"\bholds? the raised hand\b",
        r"\bholding (?:onto|her|him|them)\b",
        r"\bin (?:a |his |her |their )?arms\b",
        r"\bchest[- ]to[- ]chest\b",
        r"\bsparring\b",
        r"\bpinned\b",
        r"\bsupport(?:s|ing|ed)? (?:her|him|them|his|the)\b",
    )
]

_FORBIDDEN_RES = [
    re.compile(p, re.I)
    for p in (
        r"\bno (?:physical )?contact\b",
        r"\bnot touching\b",
        r"\bwithout (?:any )?touching\b",
        r"\bstand(?:s|ing)? apart\b",
        r"\bkeep(?:s|ing)? (?:their |a )?distance\b",
        r"\bfew (?:feet|metres|meters) apart\b",
        r"\bno physical interaction\b",
    )
]

# Legacy fallback only when prompt text is unavailable (old jsons).
NO_CONTACT_CATS = frozenset(
    {
        "face_to_face_talk",
        "posing",
        "other_multi_person",
    }
)


def prompt_contact_intent(prompt: Optional[str]) -> str:
    """Infer interaction intent from the *actual edit prompt*.

    Returns:
      required   — prompt asserts physical contact → missing contact is a fault
      forbidden  — prompt asserts no/apart contact → unwanted fusion is a fault
      unspecified — no clear cue → only punish pathological fusion (never force contact)

    Category labels are intentionally NOT used: they are upstream taxonomy and
    often disagree with what the caption actually says.
    """
    text = prompt or ""
    if not text.strip():
        return "unspecified"
    n_c = sum(1 for r in _CONTACT_RES if r.search(text))
    n_f = sum(1 for r in _FORBIDDEN_RES if r.search(text))
    if n_f > 0 and n_c == 0:
        return "forbidden"
    if n_c >= 1:
        return "required"
    return "unspecified"


def infer_needs_contact(
    cat: Optional[str] = None,
    contact_density: Optional[str] = None,
    explicit: Optional[bool] = None,
    prompt: Optional[str] = None,
) -> bool:
    """True iff prompt (preferred) asserts required contact.

    Prefer ``prompt_contact_intent``; ``cat`` is last-resort fallback only.
    """
    if explicit is not None:
        return bool(explicit)
    if prompt is not None and str(prompt).strip():
        return prompt_contact_intent(prompt) == "required"
    dens = str(contact_density or "")
    if dens.upper().startswith("C0"):
        return False
    if (cat or "").lower() in NO_CONTACT_CATS:
        return False
    return True


def _ramp_high_good(x: float, lo: float, hi: float) -> float:
    """1 if x<=lo, 0 if x>=hi, linear otherwise (smaller x is better)."""
    if x != x:  # NaN
        return 0.0
    if x <= lo:
        return 1.0
    if x >= hi:
        return 0.0
    return float(1.0 - (x - lo) / max(hi - lo, 1e-9))


def s_proximity_contact(
    min_surf_dist: float,
    *,
    d_good: float = 0.05,
    d_fail: float = 0.40,
) -> float:
    """Closeness goodness for required-contact prompts."""
    d = float(min_surf_dist)
    if d != d:
        return 0.0
    if d < 0:
        d = 0.0
    return _ramp_high_good(d, d_good, d_fail)


def s_pen_band(
    value: float,
    *,
    ok: float,
    bad: float,
) -> float:
    """Acceptable overlap up to `ok`; full penalty by `bad` (GT-calibrated band)."""
    return _ramp_high_good(float(value), ok, bad)


def compose_inter_score(
    *,
    needs_contact: bool = False,
    min_surf_dist: float,
    pen_volume_m3: Optional[float] = None,
    pen_vert_ratio: Optional[float] = None,
    pen_inside_ratio: Optional[float] = None,
    under_detect: bool = False,
    vol_ok: float = 0.05,
    vol_bad: float = 0.15,
    tau_pen: float = 0.15,
    d_good: float = 0.05,
    d_fail: float = 0.40,
    w_prox: float = 0.55,
    w_pen: float = 0.45,
    prompt: Optional[str] = None,
    contact_intent: Optional[str] = None,
    w_fuse: float = 0.45,
    w_miss: float = 0.35,
    w_unwanted: float = 0.45,
    w_qual: float = 0.20,
    n_detected_raw: Optional[int] = None,
    n_expected: Optional[int] = None,
    s_ownership: Optional[float] = None,
    inside_ok: float = 0.20,
    inside_bad: float = 0.50,
    overdetect_ok: float = 2.0,
    overdetect_span: float = 1.5,
    ownership_amp: float = 2.0,
    protocol: str = "inter_v3.1",
) -> Dict[str, Any]:
    """Prompt-intent Inter score (additive form; penalty algebraically equivalent).

    Main equation (goodness terms ``s = 1 - P``), **inter_v3.1**:

      required:     S = 0.45 s_pen + 0.35 s_prox + 0.20 s_qual
      forbidden:    S = 0.55 s_pen + 0.45 s_clear   (qual folded into clear)
      unspecified:  S = 0.85 s_pen + 0.15 s_qual

    ``s_pen`` uses the *stricter* of volume/vert band and ``pen_inside_ratio`` band.
    ``s_qual`` penalizes ownership confusion and person over-detection (fused-limb /
    fragment contact that volume alone under-penalizes).

    Diagnostics:

      P_fuse / P_miss / P_unwanted / P_qual
    """
    intent = contact_intent or (
        prompt_contact_intent(prompt) if prompt is not None else None
    )
    if intent is None:
        intent = "required" if needs_contact else "unspecified"

    if pen_volume_m3 is not None:
        s_pen = s_pen_band(float(pen_volume_m3), ok=vol_ok, bad=vol_bad)
        pen_signal = "volume_band"
        fuse_value = float(pen_volume_m3)
        fuse_ok, fuse_bad = vol_ok, vol_bad
    elif pen_vert_ratio is not None:
        s_pen = s_pen_band(float(pen_vert_ratio), ok=tau_pen * 0.35, bad=tau_pen)
        pen_signal = "proximity_band"
        fuse_value = float(pen_vert_ratio)
        fuse_ok, fuse_bad = tau_pen * 0.35, tau_pen
    else:
        s_pen, pen_signal = 0.0, "none"
        fuse_value, fuse_ok, fuse_bad = 0.0, 0.0, 1.0

    # inter_v3.1: also listen to enclosure fraction (often high on fused limbs)
    if pen_inside_ratio is not None:
        s_pen_in = s_pen_band(
            float(pen_inside_ratio), ok=float(inside_ok), bad=float(inside_bad)
        )
        if s_pen_in < s_pen:
            s_pen = float(s_pen_in)
            pen_signal = pen_signal + "+inside_band"
            fuse_value = float(pen_inside_ratio)
            fuse_ok, fuse_bad = float(inside_ok), float(inside_bad)

    # Penalties in [0,1]
    p_fuse = float(1.0 - s_pen)
    s_prox = s_proximity_contact(min_surf_dist, d_good=d_good, d_fail=d_fail)
    p_miss = 0.0
    p_unwanted = 0.0

    # Contact-quality penalty (ownership + over-detect)
    p_qual = 0.0
    if s_ownership is not None:
        p_qual = float(max(p_qual, min(1.0, (1.0 - float(s_ownership)) * float(ownership_amp))))
    if (
        n_detected_raw is not None
        and n_expected is not None
        and int(n_expected) > 0
    ):
        ratio = float(n_detected_raw) / float(n_expected)
        p_over = float(
            np.clip(
                (ratio - float(overdetect_ok)) / max(float(overdetect_span), 1e-6),
                0.0,
                1.0,
            )
        )
        p_qual = float(max(p_qual, 0.75 * p_over))

    if intent == "required":
        p_miss = float(1.0 - s_prox)
        p_fuse_term = float(w_fuse) * p_fuse
        p_miss_term = float(w_miss) * p_miss
        p_unwanted_term = 0.0
        p_qual_term = float(w_qual) * p_qual
        regime = "prompt_contact_required"
    elif intent == "forbidden":
        # Unwanted contact: too near (S_prox high) OR fused — take the worse.
        # NOTE: must NOT use (1 - S_prox); that is P_miss (too far) for required.
        d = float(min_surf_dist)
        if d != d:
            p_near = 1.0  # unknown distance → treat as unsafe when contact forbidden
        else:
            # S_prox = ramp(dist): nearer → higher; use as nearness penalty
            p_near = float(s_prox)
        p_unwanted = float(max(p_near, p_fuse, p_qual))
        p_fuse_term = 0.55 * p_fuse
        p_miss_term = 0.0
        p_unwanted_term = float(w_unwanted) * p_unwanted
        p_qual_term = 0.0
        regime = "prompt_contact_forbidden"
    else:
        # unspecified: fusion + mild quality (over-detect / ownership)
        p_fuse_term = 0.85 * p_fuse
        p_miss_term = 0.0
        p_unwanted_term = 0.0
        p_qual_term = 0.15 * p_qual
        s_prox = 1.0  # N/A
        regime = "prompt_contact_unspecified"

    s_inter = float(
        max(0.0, 1.0 - p_fuse_term - p_miss_term - p_unwanted_term - p_qual_term)
    )
    if under_detect:
        s_inter *= 0.5

    return {
        "S_inter_mesh": float(s_inter),
        "S_prox": float(s_prox),
        "S_pen": float(s_pen),
        "P_fuse": float(p_fuse),
        "P_miss": float(p_miss),
        "P_unwanted": float(p_unwanted),
        "P_qual": float(p_qual),
        "inter_regime": regime,
        "contact_intent": intent,
        "needs_contact": intent == "required",
        "pen_signal": pen_signal,
        "inter_protocol": protocol,
        "vol_ok": float(vol_ok),
        "vol_bad": float(vol_bad),
        "d_good": float(d_good),
        "d_fail": float(d_fail),
        "w_prox": float(w_miss),  # kept for compat; means miss weight
        "w_pen": float(w_fuse),
        "w_fuse": float(w_fuse),
        "w_miss": float(w_miss),
        "w_unwanted": float(w_unwanted),
        "w_qual": float(w_qual),
        "fuse_value": float(fuse_value),
        "fuse_ok": float(fuse_ok),
        "fuse_bad": float(fuse_bad),
    }


def score_humans(
    humans_verts: Sequence[np.ndarray],
    faces: np.ndarray,
    n_expected: int,
    *,
    body_poses: Optional[Sequence[Any]] = None,
    j3ds: Optional[Sequence[Any]] = None,
    shapes: Optional[Sequence[Any]] = None,
    j2ds: Optional[Sequence[Any]] = None,
    img_path: Optional[Any] = None,
    abhuman_weights: Optional[Any] = None,
    hmr_img_size: Optional[int] = None,
    needs_contact: bool = True,
    prompt: Optional[str] = None,
    contact_intent: Optional[str] = None,
    tau_pen: float = 0.15,
    tau_contact: float = 0.02,
    tau_vol: float = 0.002,  # legacy; prefer vol_ok/vol_bad
    vol_ok: float = 0.05,
    vol_bad: float = 0.15,
    d_good: float = 0.05,
    d_fail: float = 0.40,
    use_volume: bool = True,
    use_anat_extended: bool = True,
    n_detected_raw: Optional[int] = None,
) -> Dict[str, Any]:
    """Score Anat/Inter on the *already selected* humans (top-k by R#)."""
    intent = contact_intent or (
        prompt_contact_intent(prompt) if prompt is not None else None
    )
    if intent is None:
        intent = "required" if needs_contact else "unspecified"
    n = len(humans_verts)
    recon_fail = n == 0 or any(
        (v is None) or (not np.isfinite(_as_float_verts(v)).all()) for v in humans_verts
    )
    # Count diagnostic: kept humans vs prompt-expected
    s_count = 1.0 if (not recon_fail and n == int(n_expected)) else 0.0
    # under-detection: kept < expected
    under_det = (not recon_fail) and n < int(n_expected)

    if recon_fail:
        return {
            "n_humans": n,
            "n_expected": int(n_expected),
            "n_detected_raw": n_detected_raw,
            "S_count_mesh": s_count,
            "S_anat_mesh": 0.0,
            "S_inter_mesh": 0.0,
            "pen_vert_ratio": None,
            "pen_inside_ratio": None,
            "pen_volume_m3": None,
            "pen_depth_mean": None,
            "min_surf_dist": None,
            "contact_band_hit": False,
            "recon_fail": True,
            "under_detect": True,
            "needs_contact": intent == "required",
            "contact_intent": intent,
            "inter_regime": f"prompt_contact_{intent}",
            "pairs": [],
        }

    anat_parts = []
    for i in range(n):
        pose = body_poses[i] if body_poses and i < len(body_poses) else None
        j3d = j3ds[i] if j3ds and i < len(j3ds) else None
        beta = shapes[i] if shapes and i < len(shapes) else None
        part = anatomy_score(pose, j3d, beta)
        # extended per-person
        try:
            from anat_extended import hand_fine_score, part_mesh_self_collision

            hand = hand_fine_score(j3d)
            pmesh = part_mesh_self_collision(humans_verts[i], j3d)
            part["hand_fine_score"] = hand.get("score")
            part["hand_detail"] = hand
            part["part_mesh_self_score"] = pmesh.get("score")
            part["part_mesh_self_detail"] = pmesh
            # soft-blend into person score if available
            extras = [
                x
                for x in (hand.get("score"), pmesh.get("score"))
                if x is not None
            ]
            if extras:
                part["S_anat_mesh"] = float(
                    0.75 * part["S_anat_mesh"] + 0.25 * float(np.mean(extras))
                )
        except Exception as e:
            part["anat_ext_error"] = repr(e)
        anat_parts.append(part)
    s_anat_person = (
        float(np.mean([a["S_anat_mesh"] for a in anat_parts])) if anat_parts else 1.0
    )

    def _anat_mean(key: str) -> Optional[float]:
        xs = [a[key] for a in anat_parts if a.get(key) is not None]
        return float(np.mean(xs)) if xs else None

    s_anat_joint = _anat_mean("joint_limit_score")
    s_anat_bone = _anat_mean("bone_proportion_score")
    s_anat_self = _anat_mean("self_collision_score")
    s_anat_shape = _anat_mean("shape_plausibility_score")
    s_anat_hand = _anat_mean("hand_fine_score")
    s_anat_pmesh = _anat_mean("part_mesh_self_score")

    pairs = []
    pens, insides, vols, dists, hits = [], [], [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            try:
                geo = penetration_pair(
                    humans_verts[i],
                    humans_verts[j],
                    faces,
                    compute_volume=use_volume,
                )
            except Exception as e:
                geo = {
                    "pen_vert_ratio": 0.0,
                    "pen_inside_ratio": 0.0,
                    "pen_volume_m3": 0.0,
                    "pen_depth_mean": 0.0,
                    "min_surf_dist": float("nan"),
                    "error": repr(e),
                    "volume_method": "error",
                }
            hit = contact_band_hit(geo["min_surf_dist"], tau_contact)
            pairs.append({"i": i, "j": j, **geo, "contact_band_hit": hit})
            pens.append(geo["pen_vert_ratio"])
            insides.append(geo.get("pen_inside_ratio") or 0.0)
            vols.append(geo.get("pen_volume_m3") or 0.0)
            dists.append(geo["min_surf_dist"])
            hits.append(hit)

    if pens:
        pen = float(np.max(pens))
        inside = float(np.max(insides)) if insides else 0.0
        vol = float(np.max(vols)) if vols else 0.0
        min_d = float(np.nanmin(dists)) if np.any(np.isfinite(dists)) else float("nan")
        any_hit = bool(any(hits))
    else:
        pen, inside, vol, min_d, any_hit = 0.0, 0.0, 0.0, float("nan"), False

    # If volume path failed entirely, fall back to vert-ratio band via tau_pen
    use_vol = use_volume and any(
        p.get("volume_method", "none") not in ("none", "error", "fallback_zero")
        for p in pairs
    )
    # Ownership before Inter so contact-quality term can use it (inter_v3.1)
    s_own_early: Optional[float] = None
    if use_anat_extended and n >= 2:
        try:
            from anat_extended import ownership_confusion_score

            j3ds_l_early = list(j3ds) if j3ds else [None] * n
            own_early = ownership_confusion_score(humans_verts, j3ds_l_early)
            if own_early.get("available") and own_early.get("score") is not None:
                s_own_early = float(own_early["score"])
        except Exception:
            s_own_early = None

    inter = compose_inter_score(
        needs_contact=(intent == "required"),
        contact_intent=intent,
        prompt=prompt,
        min_surf_dist=min_d,
        pen_volume_m3=vol if use_vol else None,
        pen_vert_ratio=pen if not use_vol else None,
        pen_inside_ratio=inside if insides else None,
        under_detect=under_det,
        vol_ok=vol_ok,
        vol_bad=vol_bad,
        tau_pen=tau_pen,
        d_good=d_good,
        d_fail=d_fail,
        n_detected_raw=n_detected_raw if n_detected_raw is not None else n,
        n_expected=int(n_expected),
        s_ownership=s_own_early,
    )

    # --- scene-level extended Anat ---
    scene: Dict[str, Any] = {}
    s_anat = s_anat_person
    s_anat_scale = s_anat_contact = s_anat_own = s_anat_resid = s_anat_ab = None
    s_anat_over = None
    if use_anat_extended:
        try:
            from anat_extended import (
                abhuman_score,
                aggregate_extended_anat,
                contact_region_anat_score,
                cross_person_scale_score,
                explain_residual_proxy,
                ownership_confusion_score,
                structure_overcount_score,
            )

            poses_l = list(body_poses) if body_poses else [None] * n
            j3ds_l = list(j3ds) if j3ds else [None] * n
            j2ds_l = list(j2ds) if j2ds else []
            scene["cross_person_scale"] = cross_person_scale_score(j3ds_l)
            scene["contact_region_anat"] = contact_region_anat_score(
                humans_verts, j3ds_l, poses_l
            )
            scene["ownership"] = ownership_confusion_score(humans_verts, j3ds_l)
            scene["explain_residual"] = explain_residual_proxy(
                Path(img_path) if img_path else None,
                j2ds_l,
                img_size=hmr_img_size,
            )
            # MUST-catch: fg − mesh sil / skeleton (j2d unpadded from canvas)
            scene["structure_overcount"] = structure_overcount_score(
                Path(img_path) if img_path else None,
                int(n_expected),
                j2ds=j2ds_l,
                j3ds=j3ds_l,
                verts=list(humans_verts) if humans_verts else None,
                img_size=hmr_img_size,
            )
            scene["abhuman"] = abhuman_score(
                Path(img_path) if img_path else None,
                weights=Path(abhuman_weights) if abhuman_weights else None,
            )
            agg = aggregate_extended_anat(
                per_person=anat_parts,
                scene=scene,
                under_detect=under_det,
                recon_fail=False,
                n_detected_raw=n_detected_raw if n_detected_raw is not None else n,
                n_expected=int(n_expected),
            )
            s_anat = float(agg["S_anat_mesh"])
            s_anat_scale = (scene["cross_person_scale"] or {}).get("score")
            s_anat_contact = (scene["contact_region_anat"] or {}).get("score")
            s_anat_own = (scene["ownership"] or {}).get("score")
            s_anat_resid = (scene["explain_residual"] or {}).get("score")
            s_anat_ab = (scene["abhuman"] or {}).get("score")
            s_anat_over = (scene["structure_overcount"] or {}).get("score")
            for k in (
                "P_anat_extra",
                "P_anat_resid",
                "P_anat_struct",
                "P_anat_detect",
                "anat_formula",
                "w_extra",
                "w_resid",
                "w_struct",
                "w_detect",
            ):
                if k in agg:
                    scene[k] = agg[k]
        except Exception as e:
            scene = {"error": repr(e)}

    return {
        "n_humans": n,
        "n_expected": int(n_expected),
        "n_detected_raw": n_detected_raw if n_detected_raw is not None else n,
        "S_count_mesh": s_count,
        "S_anat_mesh": s_anat,
        "S_anat_person": s_anat_person,
        "S_anat_joint": s_anat_joint,
        "S_anat_bone": s_anat_bone,
        "S_anat_self": s_anat_self,
        "S_anat_shape": s_anat_shape,
        "S_anat_hand": s_anat_hand,
        "S_anat_part_mesh": s_anat_pmesh,
        "S_anat_scale": s_anat_scale,
        "S_anat_contact_region": s_anat_contact,
        "S_anat_ownership": s_anat_own,
        "S_anat_residual": s_anat_resid,
        "S_anat_overcount": s_anat_over,
        "S_anat_abhuman": s_anat_ab,
        "anat_scene": scene,
        "anat_leftover_frac": (scene.get("structure_overcount") or {}).get(
            "leftover_frac"
        ),
        "anat_n_leftover_blobs": (scene.get("structure_overcount") or {}).get(
            "n_leftover_blobs"
        ),
        "anat_overcount_note": (scene.get("structure_overcount") or {}).get("note"),
        "P_anat_extra": scene.get("P_anat_extra"),
        "P_anat_resid": scene.get("P_anat_resid"),
        "P_anat_struct": scene.get("P_anat_struct"),
        "P_anat_detect": scene.get("P_anat_detect"),
        "anat_formula": scene.get("anat_formula"),
        "S_inter_mesh": float(inter["S_inter_mesh"]),
        "S_prox": inter["S_prox"],
        "S_pen": inter["S_pen"],
        "anat_parts": anat_parts,
        "pen_vert_ratio": pen,
        "pen_inside_ratio": inside,
        "pen_volume_m3": vol,
        "pen_depth_mean": float(np.mean([p.get("pen_depth_mean", 0.0) for p in pairs]))
        if pairs
        else 0.0,
        "min_surf_dist": min_d,
        "contact_band_hit": any_hit,
        "recon_fail": False,
        "under_detect": under_det,
        "pairs": pairs,
        "tau_pen": tau_pen,
        "tau_contact": tau_contact,
        "tau_vol": tau_vol,
        **{k: inter[k] for k in (
            "inter_regime", "needs_contact", "contact_intent", "pen_signal",
            "P_fuse", "P_miss", "P_unwanted",
            "vol_ok", "vol_bad", "d_good", "d_fail",
            "w_prox", "w_pen", "w_fuse", "w_miss", "w_unwanted",
        )},
    }
