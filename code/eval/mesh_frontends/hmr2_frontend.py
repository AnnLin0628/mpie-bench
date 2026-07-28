#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HMR 2.0 / 4DHumans frontend (detect → crop → SMPL).

Install (separate conda recommended):
  conda create -n hmr2 python=3.10 -y && conda activate hmr2
  # follow https://github.com/shubham-goel/4D-Humans  (detectron2 + hmr2)
  git clone https://github.com/shubham-goel/4D-Humans.git ~/models/4D-Humans
  cd ~/models/4D-Humans && pip install -e .
  # download DEFAULT_CHECKPOINT via their download_models()

Notes:
  - Returns SMPL meshes (6890 verts), not SMPL-X.
  - Multi-person via external detector boxes; contact occlusion is brittle
    (expected; this is why Multi-HMR is the primary frontend).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np


class HMR2Frontend:
    name = "hmr2"
    model_name = "HMR2"
    img_size = 256

    def __init__(
        self,
        repo: Optional[str | Path] = None,
        device: str = "cuda",
        det_thresh: float = 0.5,
    ):
        import torch

        self.torch = torch
        assert torch.cuda.is_available(), "HMR2Frontend needs CUDA"
        self.device = torch.device(device)
        self.det_thresh = float(det_thresh)

        repo_p = Path(repo).expanduser().resolve() if repo else (
            Path.home() / "models" / "4D-Humans"
        )
        if not repo_p.is_dir():
            raise FileNotFoundError(
                f"HMR2 repo not found: {repo_p}\n"
                "  git clone https://github.com/shubham-goel/4D-Humans.git ~/models/4D-Humans\n"
                "  cd ~/models/4D-Humans && pip install -e ."
            )
        self.repo = repo_p
        if str(repo_p) not in sys.path:
            sys.path.insert(0, str(repo_p))

        try:
            from hmr2.configs import CACHE_DIR_4DHUMANS
            from hmr2.models import DEFAULT_CHECKPOINT, download_models, load_hmr2
            from hmr2.models.smpl_wrapper import SMPL
        except Exception as e:
            raise RuntimeError(
                "Cannot import hmr2. Activate the hmr2 conda env and "
                f"`pip install -e {repo_p}`. Original: {e}"
            ) from e

        download_models(CACHE_DIR_4DHUMANS)
        self._ensure_smpl(repo_p, CACHE_DIR_4DHUMANS)
        # Headless machines may have no EGL; scoring only needs verts, not pyrender.
        self.model, self.model_cfg = self._load_hmr2_no_renderer(
            DEFAULT_CHECKPOINT, load_hmr2
        )
        self.model = self.model.to(self.device).eval()
        self.model_name = Path(DEFAULT_CHECKPOINT).stem

        # SMPL faces (shared)
        smpl = SMPL(self.model_cfg.SMPL.MODEL_PATH)
        self.faces = np.asarray(smpl.faces, dtype=np.int64)

        # Detector: prefer vitdet demo helper if present, else detectron2 COCO-Keypoints
        self._detector = self._build_detector()

    @staticmethod
    def _load_hmr2_no_renderer(checkpoint_path: str, load_hmr2_fn):
        """Like hmr2.models.load_hmr2 but init_renderer=False (skip pyrender/EGL)."""
        from pathlib import Path as _Path

        from hmr2.configs import get_config
        from hmr2.models import HMR2, check_smpl_exists

        model_cfg_path = str(_Path(checkpoint_path).parent.parent / "model_config.yaml")
        model_cfg = get_config(model_cfg_path, update_cachedir=True)
        if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and (
            "BBOX_SHAPE" not in model_cfg.MODEL
        ):
            model_cfg.defrost()
            assert model_cfg.MODEL.IMAGE_SIZE == 256
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()
        check_smpl_exists()
        try:
            model = HMR2.load_from_checkpoint(
                checkpoint_path,
                strict=False,
                cfg=model_cfg,
                init_renderer=False,
            )
        except TypeError:
            # Older PL / unexpected signature — fall back then stub renderers.
            model, model_cfg = load_hmr2_fn(checkpoint_path)
            model.renderer = None
            model.mesh_renderer = None
        return model, model_cfg

    @staticmethod
    def _ensure_smpl(repo: Path, cache_dir: str) -> None:
        """Place SMPL_NEUTRAL.pkl where 4D-Humans expects it.

        Official check looks at:
          1) {CACHE_DIR_4DHUMANS}/data/smpl/SMPL_NEUTRAL.pkl
          2) cwd-relative data/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
        Scoring runs from code/eval, so (2) usually misses. We stage into (1).
        """
        target = Path(cache_dir) / "data" / "smpl" / "SMPL_NEUTRAL.pkl"
        if target.is_file():
            return

        env_src = os.environ.get("SMPL_MODEL_PATH") or os.environ.get("SMPL_NEUTRAL_PKL")
        candidates: List[Path] = []
        if env_src:
            candidates.append(Path(env_src).expanduser())
        candidates.extend(
            [
                repo / "data" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
                repo / "data" / "SMPL_NEUTRAL.pkl",
                Path.home() / "models" / "smpl" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
                Path.home() / "models" / "smpl" / "SMPL_NEUTRAL.pkl",
                Path.home() / "models" / "body_models" / "smpl" / "SMPL_NEUTRAL.pkl",
                Path.home() / "models" / "body_models" / "smpl" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
                Path.home() / "mpie_weights" / "smpl" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl",
                Path.home() / "mpie_weights" / "smpl" / "SMPL_NEUTRAL.pkl",
            ]
        )
        src = next((p for p in candidates if p is not None and p.is_file()), None)
        if src is None:
            dest_hint = repo / "data" / "basicModel_neutral_lbs_10_207_0_v1.0.0.pkl"
            raise FileNotFoundError(
                "SMPL neutral model missing for HMR2.\n"
                "Download from https://smplify.is.tue.mpg.de/ (register → Downloads),\n"
                f"then place basicModel_neutral_lbs_10_207_0_v1.0.0.pkl at:\n"
                f"  {dest_hint}\n"
                "or set SMPL_MODEL_PATH=/path/to/that.pkl (or SMPL_NEUTRAL.pkl).\n"
                f"Target cache path: {target}"
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        name = src.name.lower()
        if name == "smpl_neutral.pkl":
            if not target.exists():
                try:
                    target.symlink_to(src.resolve())
                except OSError:
                    import shutil

                    shutil.copy2(src, target)
            print(f"[hmr2] SMPL → {target} (from {src})")
            return

        # Python2→3 convert (same as 4D-Humans check_smpl_exists)
        from hmr2.models import convert_pkl

        convert_pkl(str(src), str(target))
        print(f"[hmr2] converted SMPL {src} → {target}")

    def _build_detector(self):
        """Return callable(img_bgr_uint8) -> Nx4 xyxy + scores."""
        # Try 4DHumans vitdet
        try:
            from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
            from detectron2.config import LazyConfig
            import detectron2.data.transforms as T  # noqa: F401

            cfg_path = self.repo / "hmr2" / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
            if cfg_path.is_file():
                cfg = LazyConfig.load(str(cfg_path))
                cfg.train.init_checkpoint = (
                    "https://dl.fbaipublicfiles.com/detectron2/"
                    "ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
                )
                for i in range(3):
                    cfg.model.roi_heads.box_predictors[i].test_score_thresh = self.det_thresh
                predictor = DefaultPredictor_Lazy(cfg)

                def _det(img_bgr):
                    out = predictor(img_bgr)["instances"]
                    keep = out.pred_classes == 0  # person
                    boxes = out.pred_boxes.tensor[keep].detach().cpu().numpy()
                    scores = out.scores[keep].detach().cpu().numpy()
                    return boxes, scores

                return _det
        except Exception as e:
            print(f"[hmr2] vitdet unavailable ({e}); falling back to detectron2 zoo", flush=True)

        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor

            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
                )
            )
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.det_thresh
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"
            )
            cfg.MODEL.DEVICE = str(self.device)
            predictor = DefaultPredictor(cfg)

            def _det(img_bgr):
                out = predictor(img_bgr)["instances"].to("cpu")
                keep = out.pred_classes.numpy() == 0
                boxes = out.pred_boxes.tensor.numpy()[keep]
                scores = out.scores.numpy()[keep]
                return boxes, scores

            return _det
        except Exception as e:
            print(f"[hmr2] detectron2 zoo unavailable ({e}); falling back to torchvision", flush=True)

        # Torchvision Faster R-CNN — no detectron2; enough for alt-frontend corr
        try:
            import torch
            from torchvision.models.detection import (
                fasterrcnn_resnet50_fpn_v2,
                FasterRCNN_ResNet50_FPN_V2_Weights,
            )

            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            det = fasterrcnn_resnet50_fpn_v2(weights=weights).to(self.device).eval()
            tfm = weights.transforms()

            def _det(img_bgr):
                import cv2

                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                # weights.transforms expects PIL or tensor; use torch from numpy
                t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
                t = tfm(t).to(self.device)
                with torch.no_grad():
                    out = det([t])[0]
                labels = out["labels"].detach().cpu().numpy()
                scores = out["scores"].detach().cpu().numpy()
                boxes = out["boxes"].detach().cpu().numpy()
                keep = (labels == 1) & (scores >= self.det_thresh)  # COCO person=1
                return boxes[keep], scores[keep]

            print("[hmr2] using torchvision FasterRCNN person detector", flush=True)
            return _det
        except Exception as e2:
            raise RuntimeError(
                "No person detector for HMR2. Tried detectron2 then torchvision.\n"
                f"Last error: {e2}"
            ) from e2

    def infer(
        self, img_path: Path
    ) -> Tuple[
        List[np.ndarray], List[Any], List[Any], List[Any], List[Any], List[float], float
    ]:
        import cv2
        from hmr2.datasets.vitdet_dataset import ViTDetDataset
        from hmr2.utils import recursive_to
        from torch.utils.data import DataLoader

        t0 = time.time()
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise FileNotFoundError(img_path)
        boxes, det_scores = self._detector(img_bgr)
        if boxes is None or len(boxes) == 0:
            return [], [], [], [], [], [], (time.time() - t0) * 1000.0

        dataset = ViTDetDataset(self.model_cfg, img_bgr, boxes)
        loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        verts_list: List[np.ndarray] = []
        poses: List[Any] = []
        j3ds: List[Any] = []
        shapes: List[Any] = []
        j2ds: List[Any] = []
        scores: List[float] = []

        # align scores with boxes order (ViTDetDataset keeps box order)
        score_iter = list(map(float, np.asarray(det_scores).reshape(-1)))

        with self.torch.no_grad():
            idx = 0
            for batch in loader:
                batch = recursive_to(batch, self.device)
                out = self.model(batch)
                pred = out["pred_vertices"]  # (B,V,3)
                for i in range(pred.shape[0]):
                    v = pred[i].detach().cpu().numpy().astype(np.float64)
                    verts_list.append(v)
                    # optional joints
                    pj = out.get("pred_keypoints_3d")
                    if pj is not None:
                        j3ds.append(pj[i].detach().cpu().numpy().astype(np.float64))
                    else:
                        j3ds.append(None)
                    pj2 = out.get("pred_keypoints_2d")
                    if pj2 is not None:
                        j2ds.append(pj2[i].detach().cpu().numpy().astype(np.float64))
                    else:
                        j2ds.append(None)
                    sh = out.get("pred_smpl_params", {}).get("betas") if isinstance(
                        out.get("pred_smpl_params"), dict
                    ) else None
                    if sh is not None:
                        shapes.append(sh[i].detach().cpu().numpy().astype(np.float64))
                    else:
                        shapes.append(None)
                    poses.append(None)
                    scores.append(score_iter[idx] if idx < len(score_iter) else 0.0)
                    idx += 1

        return verts_list, poses, j3ds, shapes, j2ds, scores, (time.time() - t0) * 1000.0
