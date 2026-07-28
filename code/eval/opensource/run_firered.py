#!/usr/bin/env python3
"""FireRed-Image-Edit-1.1 full-pack generation (conda: firered or mpie_edit).

Natively supports 1–3 reference images; truncate to the first 3 when more are present
(do not use the official Gemini Agent).

  conda activate firered
  CUDA_VISIBLE_DEVICES=0 python run_firered.py \\
    --pack /path/to/mpie_testset_pack --limit 2 \\
    --weights /path/to/mpie_FireRed-Image-Edit-1.1 --offload none
"""
from __future__ import annotations

import argparse
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
from runner_lib import limit_refs

MODEL_ID = "firered"


def load_pipe(weights: Path, offload: str = "none"):
    t0 = time.time()
    try:
        from diffusers import QwenImageEditPlusPipeline as Pipe
    except Exception:
        from diffusers import DiffusionPipeline as Pipe

    pipe = Pipe.from_pretrained(str(weights), torch_dtype=torch.bfloat16)
    if offload == "none":
        pipe.to("cuda")
    elif offload == "model" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    elif hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)
    print(f"firered ready {time.time()-t0:.0f}s offload={offload}", flush=True)
    return pipe


def generate(pipe, prompt: str, imgs: list[Image.Image], steps: int, seed: int, cfg: float):
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    kwargs = dict(
        image=imgs if len(imgs) > 1 else imgs[0],
        prompt=prompt,
        negative_prompt=" ",
        num_inference_steps=steps,
        generator=gen,
        true_cfg_scale=cfg,
        guidance_scale=1.0,
        num_images_per_prompt=1,
    )
    with torch.inference_mode():
        try:
            return pipe(**kwargs).images[0]
        except TypeError:
            kwargs.pop("true_cfg_scale", None)
            kwargs["image"] = imgs
            return pipe(**kwargs).images[0]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--weights", default="")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--true-cfg-scale", type=float, default=4.0)
    ap.add_argument("--max-refs", type=int, default=3, help="FireRed native range: 1–3 refs")
    ap.add_argument(
        "--offload",
        choices=("sequential", "model", "none"),
        default="none",
        help="use none on high-VRAM GPUs; sequential when VRAM is tight",
    )
    args = ap.parse_args()

    root = pack_root(args.pack)
    mid = seed_model_id(MODEL_ID, args.seed, args.seed_tag)
    w = Path(args.weights) if args.weights else weights_root() / "FireRed-Image-Edit-1.1"
    pipe = load_pipe(w, offload=args.offload)

    n_ok = n_fail = 0
    tag = f"shard{args.shard_id}/{args.num_shards}"
    print(f"=== {mid} seed={args.seed} {tag} pack={root} weights={w} ===", flush=True)

    for row in iter_todo(
        root, mid, limit=args.limit,
        shard_id=args.shard_id, num_shards=args.num_shards,
    ):
        sid = row["sample_id"]
        try:
            refs = limit_refs(resolve_refs(root, row), args.max_refs)
            imgs = [Image.open(p).convert("RGB") for p in refs]
            t1 = time.time()
            out = generate(
                pipe, row["prompt"], imgs, args.steps,
                seed=args.seed, cfg=args.true_cfg_scale,
            )
            dest = sample_out_path(root, mid, sid)
            out.save(dest)
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                seconds=time.time() - t1, n_refs=len(refs), ok=True,
                extra={
                    "shard_id": args.shard_id, "seed": int(args.seed),
                    "base_model_id": MODEL_ID, "steps": args.steps,
                    "true_cfg_scale": args.true_cfg_scale,
                },
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
