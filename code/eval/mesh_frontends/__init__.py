"""Alternate mesh reconstruction frontends for MPIE Anat/Inter.

All frontends expose the same interface as MultiHMRBackend.infer:
  infer(img_path) -> verts, poses, j3ds, shapes, j2ds, scores, elapsed_ms
  .faces : (F,3) int64
  .name / .model_name / .img_size
"""
from __future__ import annotations

from typing import Any


def build_frontend(name: str, **kwargs: Any):
    name = (name or "multi_hmr").strip().lower().replace("-", "_")
    if name in ("multi_hmr", "multihmr", "mhmr"):
        from score_mesh_v3 import MultiHMRBackend

        repo = kwargs.get("multihmr_repo") or kwargs.get("repo")
        backend = MultiHMRBackend(
            repo,
            model_name=kwargs.get("backend_model", "multiHMR_896_L"),
            det_thresh=float(kwargs.get("det_thresh", 0.2)),
            device=kwargs.get("device", "cuda"),
        )
        backend.name = "multi_hmr"
        backend.img_size = int(getattr(backend.model, "img_size", 896) or 896)
        return backend
    if name in ("hmr2", "hmr_2", "4dhumans", "fourdhumans"):
        from mesh_frontends.hmr2_frontend import HMR2Frontend

        return HMR2Frontend(
            repo=kwargs.get("hmr2_repo") or kwargs.get("repo"),
            device=kwargs.get("device", "cuda"),
            det_thresh=float(kwargs.get("det_thresh", 0.5)),
        )
    if name in ("smpler_x", "smplerx", "smpler-x"):
        from mesh_frontends.smplerx_frontend import SMPLerXFrontend

        return SMPLerXFrontend(
            repo=kwargs.get("smplerx_repo") or kwargs.get("repo"),
            ckpt=kwargs.get("smplerx_ckpt") or kwargs.get("ckpt"),
            device=kwargs.get("device", "cuda"),
            det_thresh=float(kwargs.get("det_thresh", 0.3)),
        )
    raise ValueError(
        f"unknown frontend {name!r}; choose multi_hmr | hmr2 | smpler_x"
    )


__all__ = ["build_frontend"]
