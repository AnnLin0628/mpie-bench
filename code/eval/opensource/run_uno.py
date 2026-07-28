#!/usr/bin/env python3
"""UNO Full quantity of pictures (conda: uno). Multiple references subject-driven。

24GB GPUs: recommend: --offload --model-type flux-dev-fp8

  conda activate uno
  export FLUX_DEV=~/mpie_weights/flux1-dev/flux1-dev.safetensors   # or directory convention see UNO README
  CUDA_VISIBLE_DEVICES=0 python run_uno.py --pack ~/mpie_testset_pack --limit 2 --offload
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

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
from runner_lib import code_root, find_first, limit_refs, prepend_sys_path

MODEL_ID = "uno"


def setup_uno_env():
    """Try to ~/mpie_weights fill UNO Required environment variables (will not be overwritten if they already exist).

    Offline machines must be provided locally CLIP/T5, otherwise it will hit HF. Prioritize reuse diffusers Layout
    flux1-kontext-dev / flux1-fill-dev / flux1-dev down text_encoder(_2)。
    fp8 When the file is missing: put FLUX_DEV_FP8 point to bf16 safetensors（UNO Will transfer on the spot fp8）。
    """
    w = weights_root()
    flux = w / "flux1-dev"
    uno = w / "uno"
    home = Path.home()
    # CLIP / T5：UNO Read environment variables CLIP、T5(See uno/flux/util.py）
    clip = find_first(
        w / "clip-vit-large-patch14",
        w / "openai-clip-vit-large-patch14",
        flux / "text_encoder",
        w / "flux1-kontext-dev" / "text_encoder",
        w / "flux1-fill-dev" / "text_encoder",
        home / "models" / "clip-vit-large-patch14",
    )
    t5 = find_first(
        w / "xflux_text_encoders",
        w / "t5xxl",
        flux / "text_encoder_2",
        w / "flux1-kontext-dev" / "text_encoder_2",
        w / "flux1-fill-dev" / "text_encoder_2",
    )
    flux_bf16 = find_first(
        flux / "flux1-dev.safetensors",
        flux / "flux1-dev.fp8.safetensors",
        Path(flux) if flux.is_dir() else Path("/__missing__"),
    )
    mapping = {
        "FLUX_DEV": flux_bf16,
        # No independence fp8 File fallback bf16（UNO Official: Live transfer fp8）
        "FLUX_DEV_FP8": find_first(
            flux / "flux1-dev.fp8.safetensors",
            uno / "flux1-dev.fp8.safetensors",
            flux / "flux1-dev.safetensors",
        ),
        "AE": find_first(flux / "ae.safetensors", uno / "ae.safetensors"),
        "LORA": find_first(
            uno / "dit_lora.safetensors",
            uno / "uno_lora.safetensors",
            Path(uno) if uno.is_dir() else Path("/__missing__"),
        ),
        "CLIP": clip,
        "T5": t5,
    }
    for k, p in mapping.items():
        if p is not None and not os.environ.get(k):
            os.environ[k] = str(p)
            print(f"[env] {k}={p}", flush=True)
    missing = [k for k in ("CLIP", "T5", "FLUX_DEV", "AE", "LORA") if not os.environ.get(k)]
    if missing:
        print(
            f"[WARN] UNO missing env {missing}; offline will fail on HF download. "
            "Set CLIP/T5 to local dirs (e.g. flux1-kontext-dev/text_encoder[_2]).",
            flush=True,
        )


def load_pipe(model_type: str, offload: bool, lora_rank: int):
    prepend_sys_path(code_root() / "UNO")
    setup_uno_env()
    from uno.flux.pipeline import UNOPipeline

    t0 = time.time()
    device = "cuda"
    pipe = UNOPipeline(model_type, device, offload, only_lora=True, lora_rank=lora_rank)
    print(f"pipe ready {time.time()-t0:.0f}s type={model_type} offload={offload}", flush=True)
    return pipe


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--size", type=int, default=704)
    ap.add_argument("--max-refs", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--offload", action="store_true", default=True)
    ap.add_argument("--no-offload", action="store_true")
    ap.add_argument("--model-type", default="flux-dev-fp8",
                    choices=("flux-dev", "flux-dev-fp8", "flux-schnell"))
    ap.add_argument("--lora-rank", type=int, default=512)
    args = ap.parse_args()
    if args.no_offload:
        args.offload = False

    root = pack_root(args.pack)
    mid = seed_model_id(MODEL_ID, args.seed, args.seed_tag)
    pipe = load_pipe(args.model_type, args.offload, args.lora_rank)
    from uno.flux.pipeline import preprocess_ref

    tag = f"shard{args.shard_id}/{args.num_shards}"
    n_ok = n_fail = 0
    print(f"=== {mid} seed={args.seed} {tag} pack={root} ===", flush=True)

    for row in iter_todo(root, mid, limit=args.limit, shard_id=args.shard_id, num_shards=args.num_shards):
        sid = row["sample_id"]
        try:
            refs = limit_refs(resolve_refs(root, row), args.max_refs)
            ref_size = 512 if len(refs) == 1 else 320
            ref_imgs = [preprocess_ref(Image.open(p).convert("RGB"), ref_size) for p in refs]
            t1 = time.time()
            out = pipe(
                prompt=row["prompt"],
                width=args.size,
                height=args.size,
                guidance=args.guidance,
                num_steps=args.steps,
                seed=int(args.seed),
                ref_imgs=ref_imgs,
                pe="d",
            )
            dest = sample_out_path(root, mid, sid)
            out.save(dest)
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                seconds=time.time() - t1, n_refs=len(refs), ok=True,
                extra={"shard_id": args.shard_id, "model_type": args.model_type,
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
