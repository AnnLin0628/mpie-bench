#!/usr/bin/env python3
"""DreamO Full quantity of pictures (conda: dreamo). Multiple reference default IP task。

24GB GPU: --quant int8(or none + offload）

  conda activate dreamo
  cd ~/mpie_code/DreamO
  # It is recommended to flux1-dev Link to ./models/black-forest-labs/FLUX.1-dev
  CUDA_VISIBLE_DEVICES=0 python "$MPIE_ROOT/code/eval"/opensource/run_dreamo.py \\
    --pack ~/mpie_testset_pack --limit 2 --quant int8
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
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

MODEL_ID = "dreamo"


def ensure_flux_link(dreamo_repo: Path):
    """DreamO Find by default ./models/black-forest-labs/FLUX.1-dev or HF id。"""
    flux = weights_root() / "flux1-dev"
    dst_parent = dreamo_repo / "models" / "black-forest-labs"
    dst = dst_parent / "FLUX.1-dev"
    if flux.is_dir() and not dst.exists():
        dst_parent.mkdir(parents=True, exist_ok=True)
        try:
            dst.symlink_to(flux)
            print(f"[link] {dst} -> {flux}", flush=True)
        except OSError as e:
            print(f"[warn] cannot link flux1-dev: {e}", flush=True)


def load_generator(quant: str, offload: bool, version: str):
    dreamo_repo = code_root() / "DreamO"
    prepend_sys_path(dreamo_repo)
    os.chdir(dreamo_repo)
    ensure_flux_link(dreamo_repo)
    from dreamo_generator import Generator

    t0 = time.time()
    gen = Generator(version=version, offload=offload, quant=quant, no_turbo=False, device="cuda")
    print(f"dreamo ready {time.time()-t0:.0f}s quant={quant} offload={offload}", flush=True)
    return gen


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--ref-res", type=int, default=512)
    ap.add_argument("--max-refs", type=int, default=2, help="DreamO UI Default most 2 Zhang Reference")
    ap.add_argument("--quant", choices=("none", "int8", "nunchaku"), default="int8")
    ap.add_argument("--offload", action="store_true", default=False)
    ap.add_argument("--version", default="v1.1", choices=("v1.1", "v1"))
    ap.add_argument("--task", default="ip", choices=("ip", "id"))
    args = ap.parse_args()

    gen = load_generator(args.quant, args.offload, args.version)
    root = pack_root(args.pack)
    mid = seed_model_id(MODEL_ID, args.seed, args.seed_tag)
    tag = f"shard{args.shard_id}/{args.num_shards}"
    n_ok = n_fail = 0
    print(f"=== {mid} seed={args.seed} {tag} pack={root} ===", flush=True)

    for row in iter_todo(root, mid, limit=args.limit, shard_id=args.shard_id, num_shards=args.num_shards):
        sid = row["sample_id"]
        try:
            refs = limit_refs(resolve_refs(root, row), args.max_refs)
            # DreamO want numpy RGB
            ref_np = [np.array(Image.open(p).convert("RGB")) for p in refs]
            while len(ref_np) < 2:
                ref_np.append(None)
            tasks = [args.task] * 2
            if ref_np[1] is None:
                tasks[1] = args.task
            t1 = time.time()
            ref_conds, _dbg, seed = gen.pre_condition(
                ref_images=ref_np[:2],
                ref_tasks=tasks,
                ref_res=args.ref_res,
                seed=int(args.seed),
            )
            out = gen.dreamo_pipeline(
                prompt=row["prompt"],
                width=args.size,
                height=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                ref_conds=ref_conds,
                generator=torch.Generator(device="cpu").manual_seed(int(seed)),
                true_cfg_scale=1.0,
                true_cfg_start_step=0,
                true_cfg_end_step=0,
                negative_prompt="",
                neg_guidance_scale=3.5,
                first_step_guidance_scale=args.guidance,
            ).images[0]
            dest = sample_out_path(root, mid, sid)
            out.save(dest)
            write_meta(root, mid, sid, make_meta(
                sample_id=sid, model_id=mid, backend="opensource",
                seconds=time.time() - t1, n_refs=len(refs), ok=True,
                extra={"shard_id": args.shard_id, "quant": args.quant,
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
