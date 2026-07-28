#!/usr/bin/env python3
"""Drop this file into SMPLer-X as:  <SMPLer-X>/demo/mpie_infer.py

MPIE calls:
  build_smplerx_inferencer(repo, ckpt, device, det_thresh)
    -> callable(img_path: str) -> List[dict]

Each dict must provide at least:
  verts: (V,3) float ndarray   # SMPL-X vertices in a shared camera/world frame
  score: float                 # detection / confidence for top-k keep
Optional: j3d, j2d, shape/betas, pose

Fill `_infer_one` using whatever demo API you already have working on this machine.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List


def build_smplerx_inferencer(
    repo: Path,
    ckpt: Path,
    device: str = "cuda",
    det_thresh: float = 0.3,
) -> Callable[[str], List[Dict[str, Any]]]:
    """Construct a per-image inferencer.

    TODO: wire to your local SMPLer-X demo. Sketch below mirrors a typical
    detect → SMPLer-X forward loop; replace with the exact calls that work
    in your install (after you can run their official demo on one image).
    """
    repo = Path(repo)
    ckpt = Path(ckpt)

    # ---- BEGIN: replace with real loading ----
    # Example skeleton (pseudo; names differ across SMPLer-X versions):
    #
    #   from main.config import cfg
    #   cfg.set_args(...)
    #   model = load_model(ckpt, device)
    #   detector = load_detector(det_thresh)
    #
    # def _infer_one(img_path: str) -> List[Dict[str, Any]]:
    #     img = load_rgb(img_path)
    #     boxes, scores = detector(img)
    #     humans = []
    #     for box, sc in zip(boxes, scores):
    #         out = model(img, box)   # -> verts, joints, ...
    #         humans.append({
    #             "verts": out["vertices"],   # (10475, 3)
    #             "score": float(sc),
    #             "j3d": out.get("joints"),
    #             "j2d": out.get("joints2d"),
    #             "shape": out.get("betas"),
    #         })
    #     return humans
    # ---- END ----

    def _infer_one(img_path: str) -> List[Dict[str, Any]]:
        raise NotImplementedError(
            f"Wire SMPLer-X demo inference in {__file__}. "
            f"repo={repo} ckpt={ckpt} device={device} det_thresh={det_thresh}. "
            f"Tried image={img_path}"
        )

    return _infer_one
