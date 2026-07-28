#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Closed source API Batch graph generation: write the same as open source pack/outputs/<model_id>/ layout.

usage:
  python run_closed.py --model gpt-image-2 --pack "$MPIE_TEST_PACK" --workers 4
  python run_closed.py --model seedream-5-pro --pack "$MPIE_TEST_PACK" --workers 4
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_EVAL = Path(__file__).resolve().parents[1]
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from pack_io import (  # noqa: E402
    add_common_args,
    iter_todo,
    make_meta,
    pack_root,
    resolve_refs,
    sample_out_path,
    write_meta,
)
from gateway import gemini_3_pro_image, gpt_image2_edit, seedream_5_pro  # noqa: E402

MODELS = {
    "gpt-image-2": "gpt",
    "gemini-3-pro-image": "gemini",
    "nano-banana-pro": "gemini",
    "seedream-5-pro": "seedream",
    "seedream-5": "seedream",
}

_MODEL_ID_ALIAS = {
    "nano-banana-pro": "gemini-3-pro-image",
    "seedream-5": "seedream-5-pro",
}

_print_lock = threading.Lock()


def _generate(kind: str, prompt: str, refs: list[Path], args) -> tuple[bytes, dict]:
    if kind == "gpt":
        return gpt_image2_edit(
            prompt,
            refs,
            size=args.size,
            quality=args.quality,
            timeout=args.timeout,
        )
    if kind == "seedream":
        return seedream_5_pro(
            prompt,
            refs,
            size=args.seedream_size,
            timeout=args.timeout,
        )
    return gemini_3_pro_image(prompt, refs, timeout=args.timeout)


def _run_one(row: dict, *, root: Path, model_id: str, kind: str, args) -> bool:
    sid = row["sample_id"]
    # Check again in parallel to avoid running in a race condition with the old process.
    dest = sample_out_path(root, model_id, sid)
    if dest.exists() and dest.stat().st_size > 1000:
        with _print_lock:
            print(f"[SKIP] {sid}", flush=True)
        return True
    t0 = time.time()
    try:
        refs = resolve_refs(root, row)
        if not refs:
            raise RuntimeError("no refs")
        img_bytes, api_meta = _generate(kind, row["prompt"], refs, args)
        dest.write_bytes(img_bytes)
        write_meta(
            root,
            model_id,
            sid,
            make_meta(
                sample_id=sid,
                model_id=model_id,
                backend="closedsource",
                seconds=time.time() - t0,
                n_refs=len(refs),
                ok=True,
                extra=api_meta,
            ),
        )
        with _print_lock:
            print(f"[OK] {sid} -> {dest.name} ({time.time()-t0:.0f}s)", flush=True)
        if args.sleep > 0:
            time.sleep(args.sleep)
        return True
    except Exception as e:  # noqa: BLE001
        write_meta(
            root,
            model_id,
            sid,
            make_meta(
                sample_id=sid,
                model_id=model_id,
                backend="closedsource",
                seconds=time.time() - t0,
                n_refs=len(row.get("ref_relpaths") or []),
                ok=False,
                error=f"{type(e).__name__}: {e}"[:500],
            ),
        )
        with _print_lock:
            print(f"[ERR] {sid}: {e}", flush=True)
        return False


def main():
    ap = add_common_args(argparse.ArgumentParser())
    ap.add_argument(
        "--model",
        required=True,
        choices=sorted(MODELS.keys()),
        help="Placement model_id;see alias _MODEL_ID_ALIAS",
    )
    ap.add_argument("--size", default="1024x1024", help="only gpt-image-2")
    ap.add_argument(
        "--seedream-size",
        default="2K",
        help="only seedream-5-pro(recommend 2K）",
    )
    ap.add_argument(
        "--quality",
        default="low",
        choices=["low", "medium", "high"],
        help="only gpt-image-2",
    )
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="The number of seconds to sleep after each success (recommended in parallel 0）",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel requests (default 4）",
    )
    args = ap.parse_args()

    model_id = _MODEL_ID_ALIAS.get(args.model, args.model)
    kind = MODELS[args.model]
    root = pack_root(args.pack)
    workers = max(1, args.workers)
    todos = list(iter_todo(root, model_id, limit=args.limit))
    print(
        f"PACK={root} model_id={model_id} kind={kind} "
        f"todo={len(todos)} workers={workers}",
        flush=True,
    )
    if not todos:
        print(f"DONE {model_id} ok=0 fail=0 (nothing todo) pack={root}")
        return

    n_ok = n_fail = 0
    if workers == 1:
        for row in todos:
            if _run_one(row, root=root, model_id=model_id, kind=kind, args=args):
                n_ok += 1
            else:
                n_fail += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    _run_one, row, root=root, model_id=model_id, kind=kind, args=args
                )
                for row in todos
            ]
            for fut in as_completed(futs):
                if fut.result():
                    n_ok += 1
                else:
                    n_fail += 1

    print(f"DONE {model_id} ok={n_ok} fail={n_fail} pack={root}")


if __name__ == "__main__":
    main()
