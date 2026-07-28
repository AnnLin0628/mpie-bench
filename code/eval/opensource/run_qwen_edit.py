#!/usr/bin/env python3
"""Qwen-Image-Edit-2511 Batch production of pictures.

24GB GPU: default sequential cpu offload(avoid model_cpu_offload of device mismatch）
high-VRAM GPU: add --no-offload

  conda activate mpie_edit
  CUDA_VISIBLE_DEVICES=5 python run_qwen_edit.py --pack "$MPIE_TEST_PACK" --limit 2
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

MODEL_ID = "qwen-image-edit-2511"


def load_pipe(weights: Path, offload: str = "sequential"):
    """offload: sequential | model | none"""
    t0 = time.time()
    try:
        from diffusers import QwenImageEditPipeline as Pipe
    except Exception:
        try:
            from diffusers import QwenImageEditPlusPipeline as Pipe
        except Exception:
            from diffusers import DiffusionPipeline as Pipe

    pipe = Pipe.from_pretrained(str(weights), torch_dtype=torch.bfloat16)

    # Some versions default true_cfg_scale>1 But not opened CFG, turn it off first to avoid meaningless warnings/fork in the road
    if hasattr(pipe, "config") and hasattr(pipe.config, "true_cfg_scale"):
        pass

    if offload == "none":
        pipe.to("cuda")
    elif offload == "model" and hasattr(pipe, "enable_model_cpu_offload"):
        # in part diffusers+Qwen The combination will trigger index_select device mismatch
        pipe.enable_model_cpu_offload()
    elif hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload()
    elif hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    print(f"pipe ready {time.time()-t0:.0f}s offload={offload}", flush=True)
    return pipe


def generate(pipe, prompt: str, img: Image.Image, steps: int, seed: int = 0):
    """Compatible with different diffusers signature; compulsory CPU generator,avoid offload Devices are inconsistent. """
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    base = dict(
        prompt=prompt,
        image=img,
        num_inference_steps=steps,
        generator=gen,
    )
    # Explicitly turn off true CFG(none negative Time should not pass CFG）
    for extra in (
        dict(true_cfg_scale=1.0),
        dict(guidance_scale=1.0, true_cfg_scale=1.0),
        {},
    ):
        kwargs = {**base, **extra}
        try:
            return pipe(**kwargs).images[0]
        except TypeError:
            continue
    # old API: images=list
    return pipe(prompt=prompt, images=[img], num_inference_steps=steps, generator=gen).images[0]


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--weights", default="", help="Weight directory; default MPIE_WEIGHTS/qwen-image-edit-2511")
    ap.add_argument(
        "--offload",
        choices=("sequential", "model", "none"),
        default="sequential",
        help="24GB: use sequential；high-VRAM use none；model Easy to trigger cuda/cpu mismatch",
    )
    ap.add_argument("--no-offload", action="store_true", help="equivalence --offload none（high-VRAM）")
    args = ap.parse_args()
    if args.no_offload:
        args.offload = "none"

    root = pack_root(args.pack)
    mid = seed_model_id(MODEL_ID, args.seed, args.seed_tag)
    w = Path(args.weights) if args.weights else weights_root() / "qwen-image-edit-2511"
    pipe = load_pipe(w, offload=args.offload)
    n_ok = n_fail = 0
    tag = f"shard{args.shard_id}/{args.num_shards}"
    print(f"=== {mid} seed={args.seed} {tag} pack={root} offload={args.offload} ===", flush=True)
    for row in iter_todo(
        root, mid, limit=args.limit,
        shard_id=args.shard_id, num_shards=args.num_shards,
    ):
        sid = row["sample_id"]
        try:
            refs = resolve_refs(root, row)
            img = Image.open(refs[0]).convert("RGB")
            t1 = time.time()
            out = generate(pipe, row["prompt"], img, args.steps, seed=args.seed)
            dest = sample_out_path(root, mid, sid)
            out.save(dest)
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                seconds=time.time() - t1, n_refs=len(refs), ok=True,
                extra={"shard_id": args.shard_id, "offload": args.offload,
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
