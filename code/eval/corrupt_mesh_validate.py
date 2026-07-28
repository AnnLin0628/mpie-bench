#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled mesh corruptions → Anat/Inter monotonicity (CPU, no HMR).

Synthetic two-person body meshes + same score_humans algebra as mesh_v3
(proximity path; volume off). Use on any machine; no CUDA.

Example:
  python corrupt_mesh_validate.py \\
    --out ./analysis/out/analysis_corruption.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_metrics import score_humans  # noqa: E402


def _icosphere(radius: float = 0.12, refine: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1, t, 0],
            [1, t, 0],
            [-1, -t, 0],
            [1, -t, 0],
            [0, -1, t],
            [0, 1, t],
            [0, -1, -t],
            [0, 1, -t],
            [t, 0, -1],
            [t, 0, 1],
            [-t, 0, -1],
            [-t, 0, 1],
        ],
        dtype=np.float64,
    )
    faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ]

    def _mid(a: int, b: int, cache: Dict, vlist: List) -> int:
        key = (a, b) if a < b else (b, a)
        if key in cache:
            return cache[key]
        mid = 0.5 * (vlist[a] + vlist[b])
        mid = mid / np.linalg.norm(mid)
        idx = len(vlist)
        vlist.append(mid)
        cache[key] = idx
        return idx

    vlist = [verts[i] / np.linalg.norm(verts[i]) for i in range(len(verts))]
    flist = [list(f) for f in faces]
    for _ in range(refine):
        cache: Dict = {}
        new_faces = []
        for a, b, c in flist:
            ab = _mid(a, b, cache, vlist)
            bc = _mid(b, c, cache, vlist)
            ca = _mid(c, a, cache, vlist)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        flist = new_faces
    return np.stack(vlist, axis=0) * float(radius), np.asarray(flist, dtype=np.int64)


def _body_mesh(
    center: np.ndarray,
    *,
    scale: Tuple[float, float, float] = (0.22, 0.55, 0.14),
    refine: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    v, f = _icosphere(radius=1.0, refine=refine)
    v = v * np.asarray(scale, dtype=np.float64)[None, :]
    v = v + np.asarray(center, dtype=np.float64)[None, :]
    return v, f


def _canonical_j3d(center: np.ndarray) -> np.ndarray:
    j = np.zeros((55, 3), dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    j[0] = c
    j[1] = c + [-0.10, -0.05, 0]
    j[2] = c + [0.10, -0.05, 0]
    j[3] = c + [0, 0.12, 0]
    j[4] = c + [-0.10, -0.45, 0]
    j[5] = c + [0.10, -0.45, 0]
    j[7] = c + [-0.10, -0.85, 0]
    j[8] = c + [0.10, -0.85, 0]
    j[12] = c + [0, 0.45, 0]
    j[15] = c + [0, 0.62, 0]
    j[16] = c + [-0.18, 0.38, 0]
    j[17] = c + [0.18, 0.38, 0]
    j[18] = c + [-0.38, 0.20, 0]
    j[19] = c + [0.38, 0.20, 0]
    j[20] = c + [-0.55, 0.05, 0]
    j[21] = c + [0.55, 0.05, 0]
    j[27] = c + [-0.62, 0.00, 0]
    j[42] = c + [0.62, 0.00, 0]
    return j


def _score(
    humans: Sequence[np.ndarray],
    faces: np.ndarray,
    *,
    n_expected: int,
    j3ds: Optional[Sequence[np.ndarray]] = None,
    intent: str = "required",
) -> Dict[str, Any]:
    return score_humans(
        list(humans),
        faces,
        n_expected,
        body_poses=None,
        j3ds=list(j3ds) if j3ds is not None else None,
        shapes=None,
        j2ds=None,
        img_path=None,
        needs_contact=(intent == "required"),
        contact_intent=intent,
        use_volume=False,
        use_anat_extended=False,
        n_detected_raw=len(humans),
    )


def build_conditions() -> Tuple[List[Dict[str, Any]], List[bool], Dict[str, Any]]:
    _, faces = _body_mesh(np.zeros(3), refine=2)
    c0 = np.array([-0.18, 0.0, 0.0])
    c1 = np.array([0.18, 0.0, 0.0])

    def pack(name, expect, humans, j3ds, n_exp, intent):
        geo = _score(humans, faces, n_expected=n_exp, j3ds=j3ds, intent=intent)
        return {
            "name": name,
            "expect": expect,
            "n_humans": geo.get("n_humans"),
            "n_expected": n_exp,
            "contact_intent": intent,
            "S_anat_mesh": geo.get("S_anat_mesh"),
            "S_inter_mesh": geo.get("S_inter_mesh"),
            "P_fuse": geo.get("P_fuse"),
            "P_miss": geo.get("P_miss"),
            "P_unwanted": geo.get("P_unwanted"),
            "min_surf_dist": geo.get("min_surf_dist"),
            "pen_vert_ratio": geo.get("pen_vert_ratio"),
            "under_detect": geo.get("under_detect"),
            "recon_fail": geo.get("recon_fail"),
        }

    v0, _ = _body_mesh(c0)
    v1, _ = _body_mesh(c1)
    j0, j1 = _canonical_j3d(c0), _canonical_j3d(c1)

    rows: List[Dict[str, Any]] = []
    rows.append(
        pack(
            "baseline_contact",
            "high Inter (required contact, no fusion)",
            [v0, v1],
            [j0, j1],
            2,
            "required",
        )
    )
    # Identical centers → pen_vert_ratio≈1 under tau_overlap=0.01 (0.04m offset is too weak).
    vp0, _ = _body_mesh(np.zeros(3))
    vp1, _ = _body_mesh(np.zeros(3))
    rows.append(
        pack(
            "force_penetration",
            "Inter ↓ (P_fuse ↑)",
            [vp0, vp1],
            [_canonical_j3d(np.zeros(3)), _canonical_j3d(np.zeros(3))],
            2,
            "required",
        )
    )
    vf0, _ = _body_mesh(np.array([-0.9, 0, 0]))
    vf1, _ = _body_mesh(np.array([0.9, 0, 0]))
    rows.append(
        pack(
            "separate_far",
            "Inter ↓ (P_miss ↑)",
            [vf0, vf1],
            [_canonical_j3d(np.array([-0.9, 0, 0])), _canonical_j3d(np.array([0.9, 0, 0]))],
            2,
            "required",
        )
    )
    rows.append(
        pack(
            "drop_person",
            "Inter ↓ (under-detect ×0.5)",
            [v0],
            [j0],
            2,
            "required",
        )
    )
    jb = j0.copy()
    jb[4] = j0[1] + np.array([0, -1.80, 0])
    jb[7] = jb[4] + np.array([0, -0.05, 0])
    rows.append(
        pack(
            "bone_corrupt",
            "Anat ↓ (bone proportion)",
            [v0, v1],
            [jb, j1],
            2,
            "required",
        )
    )
    js = j0.copy()
    js[20] = js[21]
    rows.append(
        pack(
            "self_collide_joints",
            "Anat ↓ (self-collision proxy)",
            [v0, v1],
            [js, j1],
            2,
            "required",
        )
    )
    v2, _ = _body_mesh(np.array([0.0, 0.15, 0.0]), scale=(0.18, 0.40, 0.12))
    rows.append(
        pack(
            "extra_person",
            "diagnostic n>expected",
            [v0, v1, v2],
            [j0, j1, _canonical_j3d(np.array([0.0, 0.15, 0.0]))],
            2,
            "required",
        )
    )
    rows.append(
        pack(
            "forbidden_near",
            "Inter ↓ under forbidden+near",
            [v0, v1],
            [j0, j1],
            2,
            "forbidden",
        )
    )

    base = next(r for r in rows if r["name"] == "baseline_contact")
    ba, bi = float(base["S_anat_mesh"]), float(base["S_inter_mesh"])
    checks: List[bool] = []
    for r in rows:
        name = r["name"]
        a, i = float(r["S_anat_mesh"]), float(r["S_inter_mesh"])
        ok: Optional[bool] = None
        note = ""
        if name == "force_penetration":
            ok, note = i < bi - 1e-6, f"ΔInter={i - bi:+.3f}"
        elif name == "separate_far":
            ok, note = i < bi - 1e-6, f"ΔInter={i - bi:+.3f}"
        elif name == "drop_person":
            ok, note = i < bi - 1e-6, f"ΔInter={i - bi:+.3f}"
        elif name == "bone_corrupt":
            ok, note = a < ba - 1e-6, f"ΔAnat={a - ba:+.3f}"
        elif name == "self_collide_joints":
            ok, note = a < ba - 1e-6, f"ΔAnat={a - ba:+.3f}"
        elif name == "forbidden_near":
            ok, note = i < bi - 1e-6, f"ΔInter={i - bi:+.3f}"
        elif name == "extra_person":
            ok, note = True, f"ΔInter={i - bi:+.3f}, ΔAnat={a - ba:+.3f}"
        elif name == "baseline_contact":
            ok, note = True, "anchor"
        r["delta_vs_baseline"] = {"Anat": a - ba, "Inter": i - bi}
        r["check_pass"] = ok
        r["check_note"] = note
        if name not in ("baseline_contact", "extra_person") and ok is not None:
            checks.append(bool(ok))
    return rows, checks, base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(
            Path(".")
            / "analysis" / "out"
            / "analysis_corruption.json"
        ),
    )
    args = ap.parse_args()

    rows, checks, base = build_conditions()
    n_req, n_pass = len(checks), int(sum(checks))
    summary = {
        "method": "synthetic_two_person_meshes + score_humans (use_volume=False, use_anat_extended=False)",
        "note": (
            "Validates metric algebra on known geometric damage without Multi-HMR. "
            "Image-dependent Anat residual/overcount off; bone/self-collision and "
            "Inter fuse/miss on. Optional GPU follow-up: corrupt_mesh_on_recon.py"
        ),
        "baseline": {
            "S_anat_mesh": base["S_anat_mesh"],
            "S_inter_mesh": base["S_inter_mesh"],
            "min_surf_dist": base["min_surf_dist"],
        },
        "n_required_checks": n_req,
        "n_passed": n_pass,
        "all_required_pass": n_pass == n_req,
        "conditions": rows,
    }
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"baseline Anat={base['S_anat_mesh']:.3f} Inter={base['S_inter_mesh']:.3f}")
    for r in rows:
        flag = (
            "PASS"
            if r.get("check_pass")
            else ("FAIL" if r.get("check_pass") is False else "n/a")
        )
        print(
            f"  {r['name']:22s}  Anat={r['S_anat_mesh']:.3f}  "
            f"Inter={r['S_inter_mesh']:.3f}  {r.get('check_note','')}  [{flag}]"
        )
    print(f"required checks {n_pass}/{n_req}  wrote {out}")


if __name__ == "__main__":
    main()
