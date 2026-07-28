#!/usr/bin/env python3
"""Other open source model access templates (OmniGen2 / UNO / ACE++ / BAGEL / DreamO）。

Copy as run_<model>.py Fill in later load_pipe()/generate_one()。
unified CLI with output layout to facilitate rsync times SG。

  python run_stub_template.py --model-id omnigen2 --pack ~/mpie_testset_pack --limit 1
"""
from __future__ import annotations

import argparse
import time

from PIL import Image

from common import (
    add_common_args,
    iter_todo,
    pack_root,
    resolve_refs,
    sample_out_path,
    write_meta,
)


def generate_one(row, refs, args):
    """return PIL.Image. This is a placeholder: copy the first reference to prove IO aisle. """
    return Image.open(refs[0]).convert("RGB")


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument("--model-id", required=True, help="Output the subdirectory name, such as omnigen2 / uno / ace / bagel / dreamo")
    args = ap.parse_args()
    root = pack_root(args.pack)
    n_ok = n_fail = 0
    for row in iter_todo(root, args.model_id, limit=args.limit):
        sid = row["sample_id"]
        try:
            refs = resolve_refs(root, row)
            t1 = time.time()
            out = generate_one(row, refs, args)
            dest = sample_out_path(root, args.model_id, sid)
            out.save(dest)
            write_meta(root, args.model_id, sid, {"seconds": round(time.time() - t1, 2), "stub": True})
            n_ok += 1
            print(f"[OK-STUB] {sid}", flush=True)
        except Exception as e:
            n_fail += 1
            write_meta(root, args.model_id, sid, {"error": str(e)[:500]})
            print(f"[ERR] {sid}: {e}", flush=True)
    print(f"DONE stub/{args.model_id} ok={n_ok} fail={n_fail}")


if __name__ == "__main__":
    main()
