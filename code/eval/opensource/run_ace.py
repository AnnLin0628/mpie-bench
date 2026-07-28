#!/usr/bin/env python3
"""ACE++ Full quantity of pictures (conda: ace). default subject LoRA + First reference.

24GB GPUs must --offload model|sequential(Official model .to(cuda) meeting OOM）。

  conda activate ace
  export FLUX_FILL_PATH=~/mpie_weights/flux1-fill-dev
  export SUBJECT_MODEL_PATH=~/mpie_weights/ace_plus/subject/...safetensors
  CUDA_VISIBLE_DEVICES=0 python run_ace.py --pack ~/mpie_testset_pack --limit 2 --offload model
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from common import (
    add_common_args,
    iter_todo,
    make_meta,
    pack_root,
    resolve_refs,
    sample_out_path,
    seed_model_id,
    weights_root,
    write_meta,
)
from runner_lib import code_root, prepend_sys_path

MODEL_ID = "ace"


def setup_ace_env():
    w = weights_root()
    if not os.environ.get("FLUX_FILL_PATH"):
        fill = w / "flux1-fill-dev"
        if fill.exists():
            os.environ["FLUX_FILL_PATH"] = str(fill)
            print(f"[env] FLUX_FILL_PATH={fill}", flush=True)
    if not os.environ.get("SUBJECT_MODEL_PATH"):
        cands = list((w / "ace_plus").rglob("*.safetensors")) if (w / "ace_plus").is_dir() else []
        subj = [p for p in cands if "subject" in str(p).lower()]
        pick = subj[0] if subj else (cands[0] if cands else None)
        if pick is not None:
            os.environ["SUBJECT_MODEL_PATH"] = str(pick)
            print(f"[env] SUBJECT_MODEL_PATH={pick}", flush=True)


def load_pipe(ace_repo: Path, offload: str = "model"):
    """offload: model | sequential | none。24GB GPUs must model/sequential。"""
    prepend_sys_path(ace_repo)
    os.chdir(ace_repo)
    setup_ace_env()
    import torch
    from diffusers import FluxFillPipeline
    from transformers import T5TokenizerFast
    from scepter.modules.utils.config import Config
    from scepter.modules.utils.distribute import we
    from scepter.modules.utils.file_system import FS
    import inference.ace_plus_diffusers as ace_inf
    from inference.ace_plus_diffusers import ACEPlusDiffuserInference
    from inference.utils import ACEPlusImageProcessor

    fs_cfgs = [
        Config(cfg_dict={"NAME": "HuggingfaceFs", "TEMP_DIR": "./cache"}, load=False),
        Config(cfg_dict={"NAME": "ModelscopeFs", "TEMP_DIR": "./cache"}, load=False),
        Config(cfg_dict={"NAME": "HttpFs", "TEMP_DIR": "./cache"}, load=False),
        Config(cfg_dict={"NAME": "LocalFs", "TEMP_DIR": "./cache"}, load=False),
    ]
    for one in fs_cfgs:
        FS.init_fs_client(one)

    # Cover official init:Remove .to(cuda), use diffusers offload（24G required)
    def init_from_cfg(self, cfg):
        self.max_seq_len = cfg.get("MAX_SEQ_LEN", 4096)
        self.image_processor = ACEPlusImageProcessor(max_seq_len=self.max_seq_len)
        local_folder = FS.get_dir_to_local_dir(cfg.MODEL.PRETRAINED_MODEL)
        self.pipe = FluxFillPipeline.from_pretrained(local_folder, torch_dtype=torch.bfloat16)
        if offload == "none":
            self.pipe.to(we.device_id)
        elif offload == "sequential":
            self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe.enable_model_cpu_offload()
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            os.path.join(local_folder, "tokenizer_2"),
            additional_special_tokens=["{image}"],
        )
        self.pipe.tokenizer_2 = tokenizer_2
        self.load_default(cfg.DEFAULT_PARAS)
        print(f"[ace] offload={offload}", flush=True)

    ace_inf.ACEPlusDiffuserInference.init_from_cfg = init_from_cfg

    cfg_path = ace_repo / "config" / "ace_plus_diffusers_infer.yaml"
    if not cfg_path.is_file():
        yamls = list((ace_repo / "config").glob("*diffuser*.yaml"))
        if not yamls:
            raise FileNotFoundError(f"no ACE diffuser yaml under {ace_repo}/config")
        cfg_path = yamls[0]
    pipe_cfg = Config(load=True, cfg_file=str(cfg_path))
    pipe = ACEPlusDiffuserInference()
    pipe.init_from_cfg(pipe_cfg)

    # Offline HF_HUB_OFFLINE：load_lora_weights Must be explicit weight_name
    _orig_load = pipe.pipe.load_lora_weights

    def _load_lora(pretrained_model_name_or_path_or_dict, weight_name=None, **kwargs):
        path = pretrained_model_name_or_path_or_dict
        if weight_name is None and isinstance(path, (str, os.PathLike)):
            path = str(path)
            if os.path.isfile(path):
                return _orig_load(
                    os.path.dirname(path) or ".",
                    weight_name=os.path.basename(path),
                    **kwargs,
                )
            if os.path.isdir(path):
                cands = sorted(
                    f for f in os.listdir(path)
                    if f.endswith(".safetensors") and "subject" in f.lower()
                ) or sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
                if cands:
                    return _orig_load(path, weight_name=cands[0], **kwargs)
        return _orig_load(pretrained_model_name_or_path_or_dict, weight_name=weight_name, **kwargs)

    pipe.pipe.load_lora_weights = _load_lora

    # new version FluxFill: Transmit only masked_image_latents still sometimes preprocess(image=None) Report an error
    # ACE The official calling method is not passed image, need to correct None Return placeholder tensor（strength=1 is only used as the starting point of the noise)
    _orig_prep = pipe.pipe.image_processor.preprocess

    def _preprocess(image, height=None, width=None, **kwargs):
        if image is None:
            h = int(height or 1024)
            w = int(width or 1024)
            return torch.zeros(1, 3, h, w, dtype=torch.float32)
        return _orig_prep(image, height=height, width=width, **kwargs)

    pipe.pipe.image_processor.preprocess = _preprocess

    print(f"ACE pipe ready cfg={cfg_path} offload={offload}", flush=True)
    return pipe


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--guide-scale", type=float, default=50.0)
    ap.add_argument("--lora-path", default="", help="cover SUBJECT_MODEL_PATH")
    ap.add_argument(
        "--offload",
        choices=("model", "sequential", "none"),
        default="sequential",
        help="24GB GPUs: recommend sequential；model may still be denoising OOM；high-VRAM Available none",
    )
    args = ap.parse_args()

    ace_repo = code_root() / "ACE_plus"
    pipe = load_pipe(ace_repo, offload=args.offload)
    lora = args.lora_path or os.environ.get("SUBJECT_MODEL_PATH") or ""
    if not lora:
        raise SystemExit("lack SUBJECT_MODEL_PATH / --lora-path（ace_plus subject safetensors）")

    root = pack_root(args.pack)
    mid = seed_model_id(MODEL_ID, args.seed, args.seed_tag)
    tag = f"shard{args.shard_id}/{args.num_shards}"
    n_ok = n_fail = 0
    print(f"=== {mid} seed={args.seed} {tag} pack={root} ===", flush=True)

    from PIL import Image

    for row in iter_todo(root, mid, limit=args.limit, shard_id=args.shard_id, num_shards=args.num_shards):
        sid = row["sample_id"]
        try:
            refs = resolve_refs(root, row)
            # Do not use scepter pillow_convert:The return type may not be PIL，diffusers Will refuse
            ref_img = Image.open(refs[0]).convert("RGB")
            t1 = time.time()
            image, _seed = pipe(
                reference_image=ref_img,
                edit_image=None,
                edit_mask=None,
                prompt=row["prompt"],
                output_height=args.size,
                output_width=args.size,
                sampler="flow_euler",
                sample_steps=args.steps,
                guide_scale=args.guide_scale,
                seed=int(args.seed),
                repainting_scale=1.0,
                lora_path=lora,
            )
            dest = sample_out_path(root, mid, sid)
            image.save(dest)
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                seconds=time.time() - t1, n_refs=len(refs), ok=True,
                extra={"shard_id": args.shard_id, "lora": str(lora),
                       "seed": int(args.seed), "base_model_id": MODEL_ID},
            ))
            n_ok += 1
            print(f"[OK {tag}] {sid} ({time.time()-t1:.0f}s)", flush=True)
        except Exception as e:
            n_fail += 1
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                ok=False, error=f"{type(e).__name__}: {e}"[:500],
                extra={"shard_id": args.shard_id, "seed": int(args.seed)},
            ))
            print(f"[ERR {tag}] {sid}: {e}", flush=True)
    print(f"DONE {mid} {tag} ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
