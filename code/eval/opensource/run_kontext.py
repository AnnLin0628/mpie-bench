#!/usr/bin/env python3
"""GPU host / conda: mpie_edit
FLUX.1-Kontext-dev Batch production of pictures (multiple references: take the first reference) image，prompt with the rest of the character description).

  conda activate mpie_edit
  python run_kontext.py --pack ~/mpie_testset_pack --limit 2   # smoke first
  python run_kontext.py --pack ~/mpie_testset_pack            # Whole amount
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
from diffusers import FluxKontextPipeline
from optimum.quanto import freeze, qfloat8, quantize

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

MODEL_ID = "flux1-kontext-dev"


def load_pipe(weights: Path, no_quant: bool = False):
    t0 = time.time()
    pipe = FluxKontextPipeline.from_pretrained(str(weights), torch_dtype=torch.bfloat16)
    # high-VRAM GPU Card can be loaded directly;24GB GPU Still used fp8 + offload
    if no_quant:
        pipe.to("cuda")
    else:
        quantize(pipe.transformer, weights=qfloat8)
        freeze(pipe.transformer)
        quantize(pipe.text_encoder_2, weights=qfloat8)
        freeze(pipe.text_encoder_2)
        pipe.enable_model_cpu_offload()
    print(f"pipe ready {time.time()-t0:.0f}s quant={not no_quant}", flush=True)
    return pipe


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--weights", default="", help="Weight directory; default MPIE_WEIGHTS/flux1-kontext-dev")
    ap.add_argument("--no-quant", action="store_true",
                    help="high-VRAM Recommendation: Not good fp8, on the whole mold GPU")
    args = ap.parse_args()
    root = pack_root(args.pack)
    mid = seed_model_id(MODEL_ID, args.seed, args.seed_tag)
    w = Path(args.weights) if args.weights else weights_root() / "flux1-kontext-dev"
    pipe = load_pipe(w, no_quant=args.no_quant)
    n_ok = n_fail = 0
    tag = f"shard{args.shard_id}/{args.num_shards}"
    print(f"=== {mid} seed={args.seed} {tag} pack={root} ===", flush=True)
    for row in iter_todo(
        root, mid, limit=args.limit,
        shard_id=args.shard_id, num_shards=args.num_shards,
    ):
        sid = row["sample_id"]
        try:
            refs = resolve_refs(root, row)
            if not refs:
                raise RuntimeError("no refs")
            img = Image.open(refs[0]).convert("RGB")
            # Kontext Main input single picture; multi-person information dependence prompt in (R#) describe
            t1 = time.time()
            out = pipe(
                image=img,
                prompt=row["prompt"],
                height=args.size,
                width=args.size,
                num_inference_steps=args.steps,
                guidance_scale=2.5,
                generator=torch.Generator("cpu").manual_seed(int(args.seed)),
            ).images[0]
            dest = sample_out_path(root, mid, sid)
            out.save(dest)
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                seconds=time.time() - t1, n_refs=len(refs), ok=True,
                extra={"ref0": str(refs[0]), "shard_id": args.shard_id,
                       "seed": int(args.seed), "base_model_id": MODEL_ID},
            ))
            n_ok += 1
            print(f"[OK {tag}] {sid} -> {dest.name} ({time.time()-t1:.0f}s)", flush=True)
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
