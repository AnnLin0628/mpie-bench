#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Offline weight analysis: priority $MPIE_WEIGHTS_DIR local files under, Otherwise, it will fall back to online automatic download.

for inaccessible github/pytorch Download source machine(Such as machines with restricted downloads):
setup_cn_mirror.sh The offline package will be retrieved ~/mpie_weights and export MPIE_WEIGHTS_DIR。
"""
import os
from pathlib import Path


def local_or_name(name: str) -> str:
    d = os.environ.get("MPIE_WEIGHTS_DIR", "")
    if d:
        p = Path(d) / name
        if p.exists():
            return str(p)
    return name


def load_dinov2(model_name: str = "dinov2_vitb14"):
    """DINOv2 load: Prioritize local hub cache(completely offline, Adapter cannot connect github machine)。

    The offline package will repo snapshot into ~/.cache/torch/hub/facebookresearch_dinov2_main、
    checkpoint put in hub/checkpoints/。torch.hub of github source even skip_validation
    Also urlopen Probing the default branch, Offline machines must go source='local'。
    """
    import torch
    local_repo = Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
    if local_repo.exists():
        return torch.hub.load(str(local_repo), model_name, source="local")
    return torch.hub.load("facebookresearch/dinov2", model_name,
                          trust_repo=True, skip_validation=True)
