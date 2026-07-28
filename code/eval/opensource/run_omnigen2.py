#!/usr/bin/env python3
"""OmniGen2 Full quantity of pictures (conda: omnigen2）。

  conda activate omnigen2
  CUDA_VISIBLE_DEVICES=0 python run_omnigen2.py --pack ~/mpie_testset_pack --limit 2
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

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

MODEL_ID = "omnigen2"


def load_pipe(weights: Path, offload: str):
    prepend_sys_path(code_root() / "OmniGen2")
    from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
    from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

    t0 = time.time()
    pipe = OmniGen2Pipeline.from_pretrained(str(weights), torch_dtype=torch.bfloat16, trust_remote_code=True)
    pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
        str(weights), subfolder="transformer", torch_dtype=torch.bfloat16,
    )
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif offload == "model":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    print(f"pipe ready {time.time()-t0:.0f}s offload={offload}", flush=True)
    return pipe


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--weights", default="")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--max-refs", type=int, default=4)
    ap.add_argument("--offload", choices=("sequential", "model", "none"), default="model")
    ap.add_argument("--text-guidance", type=float, default=5.0)
    ap.add_argument("--image-guidance", type=float, default=2.0)
    args = ap.parse_args()

    root = pack_root(args.pack)
    mid = seed_model_id(MODEL_ID, args.seed, args.seed_tag)
    w = Path(args.weights) if args.weights else weights_root() / "omnigen2"
    pipe = load_pipe(w, args.offload)
    tag = f"shard{args.shard_id}/{args.num_shards}"
    n_ok = n_fail = 0
    print(f"=== {mid} seed={args.seed} {tag} pack={root} ===", flush=True)

    for row in iter_todo(root, mid, limit=args.limit, shard_id=args.shard_id, num_shards=args.num_shards):
        sid = row["sample_id"]
        try:
            refs = limit_refs(resolve_refs(root, row), args.max_refs)
            imgs = [ImageOps.exif_transpose(Image.open(p).convert("RGB")) for p in refs]
            t1 = time.time()
            out = pipe(
                prompt=row["prompt"],
                input_images=imgs,
                width=args.size,
                height=args.size,
                num_inference_steps=args.steps,
                max_sequence_length=1024,
                text_guidance_scale=args.text_guidance,
                image_guidance_scale=args.image_guidance,
                num_images_per_prompt=1,
                generator=torch.Generator(device="cpu").manual_seed(int(args.seed)),
                output_type="pil",
            ).images[0]
            dest = sample_out_path(root, mid, sid)
            out.save(dest)
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                seconds=time.time() - t1, n_refs=len(refs), ok=True,
                extra={"shard_id": args.shard_id, "seed": int(args.seed),
                       "base_model_id": MODEL_ID},
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
