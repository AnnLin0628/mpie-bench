#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SMPLer-X frontend (multi-person SMPL-X).

Install:
  conda create -n smplerx python=3.8 -y && conda activate smplerx
  # follow https://github.com/caizhongang/SMPLer-X
  git clone https://github.com/caizhongang/SMPLer-X.git ~/models/SMPLer-X
  # install mmcv / mmdet / torch per their README; download smpler_x ckpt

This adapter calls their demo inference entry if present; otherwise raises
with exact next steps. Output verts are SMPL-X (10475).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


class SMPLerXFrontend:
    name = "smpler_x"
    model_name = "SMPLer-X"
    img_size = 512

    def __init__(
        self,
        repo: Optional[str | Path] = None,
        ckpt: Optional[str | Path] = None,
        device: str = "cuda",
        det_thresh: float = 0.3,
    ):
        import torch

        self.torch = torch
        assert torch.cuda.is_available(), "SMPLerXFrontend needs CUDA"
        self.device = device
        self.det_thresh = float(det_thresh)

        repo_p = Path(repo).expanduser().resolve() if repo else (
            Path.home() / "models" / "SMPLer-X"
        )
        if not repo_p.is_dir():
            raise FileNotFoundError(
                f"SMPLer-X repo not found: {repo_p}\n"
                "  git clone https://github.com/caizhongang/SMPLer-X.git ~/models/SMPLer-X\n"
                "  Install deps + download ckpt per their README."
            )
        self.repo = repo_p
        if str(repo_p) not in sys.path:
            sys.path.insert(0, str(repo_p))
        # common subpaths
        for sub in ("main", "main/transformer_utils", "."):
            p = repo_p / sub
            if p.is_dir() and str(p) not in sys.path:
                sys.path.insert(0, str(p))

        ckpt_p = Path(ckpt).expanduser() if ckpt else self._find_ckpt(repo_p)
        if ckpt_p is None or not ckpt_p.is_file():
            raise FileNotFoundError(
                "SMPLer-X checkpoint not found. Pass --smplerx-ckpt /path/to/*.pth\n"
                f"Searched under {repo_p}/pretrained_models and {repo_p}/ckpts"
            )
        self.ckpt = ckpt_p
        self.model_name = ckpt_p.stem

        self._infer_fn = self._build_inferencer()
        self.faces = self._load_faces()

    def _find_ckpt(self, repo: Path) -> Optional[Path]:
        for d in (
            repo / "pretrained_models",
            repo / "ckpts",
            repo / "checkpoints",
            Path.home() / "mpie_weights" / "smpler_x",
        ):
            if not d.is_dir():
                continue
            cands = sorted(d.rglob("*.pth")) + sorted(d.rglob("*.pt"))
            if cands:
                # prefer names containing smpler
                pref = [c for c in cands if "smpler" in c.name.lower()]
                return (pref or cands)[0]
        return None

    def _load_faces(self) -> np.ndarray:
        try:
            import smplx

            m = smplx.create(
                model_path=os.environ.get(
                    "SMPLX_MODEL_PATH",
                    str(Path.home() / "models" / "smplx"),
                ),
                model_type="smplx",
                gender="neutral",
                use_face_contour=False,
                num_betas=10,
            )
            return np.asarray(m.faces, dtype=np.int64)
        except Exception:
            # last resort: Multi-HMR smplx faces if available
            faces_np = Path.home() / "models" / "multi-hmr" / "data" / "smplx_faces.npy"
            if faces_np.is_file():
                return np.load(faces_np).astype(np.int64)
            raise RuntimeError(
                "Cannot load SMPL-X faces. Set SMPLX_MODEL_PATH to SMPL-X model dir "
                "(containing SMPLX_NEUTRAL.npz) or place smplx_faces.npy."
            )

    def _build_inferencer(self):
        """Try several known entry points from SMPLer-X demos."""
        # Path A: project-local helper we may add later
        try:
            from demo.mpie_infer import build_smplerx_inferencer  # type: ignore

            return build_smplerx_inferencer(
                self.repo, self.ckpt, device=self.device, det_thresh=self.det_thresh
            )
        except Exception:
            pass

        # Path B: lightweight torch hub style — load model class if packaged
        try:
            return self._build_via_main_config()
        except Exception as e:
            self._last_err = e
            raise RuntimeError(
                "Could not construct SMPLer-X inferencer automatically.\n"
                "Recommended: follow SMPLer-X README to run their demo once, then\n"
                "either export a thin `demo/mpie_infer.py` with\n"
                "  build_smplerx_inferencer(repo, ckpt, device, det_thresh)\n"
                "  -> callable(img_path) returning list[dict] with keys:\n"
                "     verts (Vx3), score (float), j3d?, j2d?, shape?\n"
                f"Last import error: {e}"
            ) from e

    def _build_via_main_config(self):
        """Best-effort: use SMPLer-X `main` demo modules if present."""
        # Many installs expose something like main.base.tool.inference
        # We keep this narrow to avoid fragile deep imports; raise if missing.
        raise ImportError(
            "No packaged MPIE inferencer; add demo/mpie_infer.py in SMPLer-X repo "
            "(see docs/exp_alt_mesh.md for a template)."
        )

    def infer(
        self, img_path: Path
    ) -> Tuple[
        List[np.ndarray], List[Any], List[Any], List[Any], List[Any], List[float], float
    ]:
        t0 = time.time()
        humans = self._infer_fn(str(img_path))
        verts_list, poses, j3ds, shapes, j2ds, scores = [], [], [], [], [], []
        for h in humans or []:
            v = h.get("verts") or h.get("vertices") or h.get("verts_smplx")
            if v is None:
                continue
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            verts_list.append(np.asarray(v, dtype=np.float64))
            poses.append(h.get("pose"))
            j = h.get("j3d")
            if j is not None and hasattr(j, "detach"):
                j = j.detach().cpu().numpy()
            j3ds.append(np.asarray(j, dtype=np.float64) if j is not None else None)
            sh = h.get("shape") or h.get("betas")
            if sh is not None and hasattr(sh, "detach"):
                sh = sh.detach().cpu().numpy()
            shapes.append(np.asarray(sh, dtype=np.float64) if sh is not None else None)
            j2 = h.get("j2d")
            if j2 is not None and hasattr(j2, "detach"):
                j2 = j2.detach().cpu().numpy()
            j2ds.append(np.asarray(j2, dtype=np.float64) if j2 is not None else None)
            sc = h.get("score") or h.get("scores") or 0.0
            if hasattr(sc, "detach"):
                sc = float(sc.detach().cpu().numpy().reshape(-1)[0])
            scores.append(float(sc))
        return verts_list, poses, j3ds, shapes, j2ds, scores, (time.time() - t0) * 1000.0
