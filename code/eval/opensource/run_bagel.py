#!/usr/bin/env python3
"""BAGEL-7B-MoT Full quantity of pictures (conda: bagel). Edit Mode: First Reference + prompt。

24GB single GPU:use max_mem_per_gpu=20GiB + offload_folder。

  conda activate bagel
  CUDA_VISIBLE_DEVICES=0 python run_bagel.py --pack ~/mpie_testset_pack --limit 2
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image

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
from runner_lib import code_root, limit_refs, prepend_sys_path

MODEL_ID = "bagel"


def load_inferencer(model_path: Path, max_mem: str, offload_folder: str):
    bagel_repo = code_root() / "Bagel"
    prepend_sys_path(bagel_repo)
    os.chdir(bagel_repo)

    from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
    from data.transforms import ImageTransform
    from data.data_utils import add_special_tokens
    from modeling.bagel import (
        BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel,
    )
    from modeling.qwen2 import Qwen2Tokenizer
    from modeling.autoencoder import load_ae
    from inferencer import InterleaveInferencer

    t0 = time.time()
    llm_config = Qwen2Config.from_json_file(str(model_path / "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(str(model_path / "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=str(model_path / "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, connector_act="gelu_pytorch_tanh",
        latent_patch_size=2, max_latent_size=64,
    )
    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(str(model_path))
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    device_map = infer_auto_device_map(
        model,
        max_memory={0: max_mem},
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        "language_model.model.embed_tokens", "time_embedder", "latent_pos_embed",
        "vae2llm", "llm2vae", "connector", "vit_pos_embed",
    ]
    first_device = device_map.get(same_device_modules[0], "cuda:0")
    for k in same_device_modules:
        if k in device_map:
            device_map[k] = first_device

    Path(offload_folder).mkdir(parents=True, exist_ok=True)
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=str(model_path / "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=offload_folder,
    ).eval()

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    print(f"bagel ready {time.time()-t0:.0f}s", flush=True)
    return inferencer


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--weights", default="")
    ap.add_argument("--max-refs", type=int, default=1,
                    help="The editing interface defaults to a single picture;>1 try when interleave_inference")
    ap.add_argument("--max-mem", default="20GiB")
    ap.add_argument("--offload-folder", default="/tmp/bagel_offload")
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    w = Path(args.weights) if args.weights else weights_root() / "BAGEL-7B-MoT"
    inferencer = load_inferencer(w, args.max_mem, args.offload_folder)
    hyper = dict(
        cfg_text_scale=4.0, cfg_img_scale=2.0, cfg_interval=[0.0, 1.0],
        timestep_shift=3.0, num_timesteps=args.steps,
        cfg_renorm_min=0.0, cfg_renorm_type="text_channel",
    )

    root = pack_root(args.pack)
    mid = seed_model_id(MODEL_ID, args.seed, args.seed_tag)
    tag = f"shard{args.shard_id}/{args.num_shards}"
    n_ok = n_fail = 0
    print(f"=== {mid} seed={args.seed} {tag} pack={root} ===", flush=True)
    # Bagel API has no explicit seed; pin torch/numpy for best-effort determinism.
    import torch
    import numpy as np
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    for row in iter_todo(root, mid, limit=args.limit, shard_id=args.shard_id, num_shards=args.num_shards):
        sid = row["sample_id"]
        try:
            refs = limit_refs(resolve_refs(root, row), args.max_refs)
            imgs = [Image.open(p).convert("RGB") for p in refs]
            t1 = time.time()
            if len(imgs) == 1:
                out_dict = inferencer(image=imgs[0], text=row["prompt"], **hyper)
            else:
                # More reference: Picture... + text
                seq = list(imgs) + [row["prompt"]]
                out_dict = inferencer.interleave_inference(input_lists=seq, **hyper)
                if isinstance(out_dict, list):
                    out_dict = {"image": out_dict[-1] if out_dict else None}
            out = out_dict["image"] if isinstance(out_dict, dict) else out_dict
            if isinstance(out, list):
                out = out[-1]
            dest = sample_out_path(root, mid, sid)
            out.save(dest)
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                seconds=time.time() - t1, n_refs=len(refs), ok=True,
                extra={"shard_id": args.shard_id, "seed": int(args.seed),
                       "base_model_id": MODEL_ID, "seed_note": "torch/numpy only"},
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
