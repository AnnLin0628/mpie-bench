#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy existing full-pack outputs into seed-subset pack as <base>_s0/.

Saves regenerating seed=0 when full 2500 outputs already exist on a GPU host.

Example:
  python bootstrap_multiseed_s0.py \\
    --subset-pack ~/mpie_testset_pack_seed150 \\
    --full-pack ~/mpie_testset_pack \\
    --bases flux1-kontext-dev,ace,omnigen2,dreamo,uno,bagel,qwen-image-edit-2511
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset-pack", required=True)
    ap.add_argument("--full-pack", required=True)
    ap.add_argument(
        "--bases",
        default="flux1-kontext-dev,ace,omnigen2,dreamo,uno,bagel,qwen-image-edit-2511",
    )
    ap.add_argument("--seed", type=int, default=0, help="usually 0")
    ap.add_argument("--link", action="store_true", help="symlink instead of copy")
    args = ap.parse_args()

    sub = Path(args.subset_pack).expanduser().resolve()
    full = Path(args.full_pack).expanduser().resolve()
    man = sub / "manifest.jsonl"
    if not man.is_file():
        raise SystemExit(f"missing {man}")
    sids = [
        json.loads(l)["sample_id"]
        for l in man.read_text().splitlines()
        if l.strip()
    ]
    bases = [b.strip() for b in args.bases.split(",") if b.strip()]
    seed = int(args.seed)

    for base in bases:
        src_dir = full / "outputs" / base
        dst_dir = sub / "outputs" / f"{base}_s{seed}"
        if not src_dir.is_dir():
            print(f"[skip] no {src_dir}", flush=True)
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        n_ok = n_miss = 0
        for sid in sids:
            src = src_dir / f"{sid}.png"
            dst = dst_dir / f"{sid}.png"
            if dst.exists() and dst.stat().st_size > 1000:
                n_ok += 1
                continue
            if not src.is_file() or src.stat().st_size <= 1000:
                n_miss += 1
                continue
            if args.link:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())
            else:
                shutil.copy2(src, dst)
            n_ok += 1
        print(
            f"[{base}_s{seed}] ok={n_ok}/{len(sids)} miss={n_miss} -> {dst_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
