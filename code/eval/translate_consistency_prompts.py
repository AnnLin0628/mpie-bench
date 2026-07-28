#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Consistency experiment subset in English edit prompt Translated into Chinese for annotation front-end display.

Place the order: $PACK/judgments/human_consistency/prompt_zh.json
  { sample_id: { "prompt_en", "prompt_zh", "translated_at" }, ... }

usage:
  set -a; source your env file; set +a
  python translate_consistency_prompts.py --pack "$MPIE_TEST_PACK"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from pack_io import pack_root  # noqa: E402

def translate_one_text(text: str, model: str) -> str:
    """Plain-text translation (not JSON) via same gateway as VQA."""
    import requests
    from instr_qa_common import api_key, api_url

    user = (
        "Please translate the following English image editing instructions into Simplified Chinese."
        "Retain faithfully R1/R2 Wait for character codes, actions and contact relationships; do not add or delete facts; only output the translation.\n\n"
        f"<<<\n{text}\n>>>"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }
    last = ""
    for attempt in range(4):
        try:
            r = requests.post(api_url(), headers=headers, json=payload, timeout=120)
            if r.status_code in (429, 502, 503):
                last = f"HTTP {r.status_code}"
                time.sleep(min(3 * 2**attempt, 30))
                continue
            r.raise_for_status()
            body = r.json()
            if body.get("error"):
                raise RuntimeError(str(body["error"])[:300])
            msg = (body.get("choices") or [{}])[0].get("message") or {}
            out = (msg.get("content") or "").strip()
            if out.startswith("```"):
                out = out.strip("`")
                if out.lower().startswith("text"):
                    out = out[4:].lstrip()
            if not out:
                raise ValueError("empty translation")
            return out
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"[:240]
            time.sleep(1 + attempt)
    raise RuntimeError(last or "translate failed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="")
    ap.add_argument("--judge-model", default="gpt-5.5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pack = pack_root(args.pack or None)
    hc = pack / "judgments" / "human_consistency"
    split = json.loads((hc / "_split.json").read_text(encoding="utf-8"))
    out_path = hc / "prompt_zh.json"
    cache = {}
    if out_path.is_file() and not args.force:
        cache = json.loads(out_path.read_text(encoding="utf-8"))

    todo = {}
    for name in ("guide", "pilot", "holdout", "main"):
        for it in split.get("splits", {}).get(name) or []:
            sid = it["sample_id"]
            en = it.get("prompt") or ""
            if not en:
                continue
            prev = cache.get(sid) or {}
            if prev.get("prompt_zh") and prev.get("prompt_en") == en and not args.force:
                continue
            todo[sid] = en
    items = list(todo.items())
    if args.limit:
        items = items[: args.limit]

    print(json.dumps({"cached": len(cache), "todo": len(items)}, ensure_ascii=False), flush=True)
    if not items:
        print("nothing to translate", flush=True)
        return

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {
            ex.submit(translate_one_text, en, args.judge_model): sid
            for sid, en in items
        }
        done = 0
        for fut in as_completed(futs):
            sid = futs[fut]
            done += 1
            try:
                zh = fut.result()
                cache[sid] = {
                    "sample_id": sid,
                    "prompt_en": todo[sid],
                    "prompt_zh": zh,
                    "translated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "judge_model": args.judge_model,
                }
                ok += 1
                print(f"[{done}/{len(items)}] {sid} -> ok", flush=True)
            except Exception as e:  # noqa: BLE001
                fail += 1
                print(f"[{done}/{len(items)}] {sid} -> FAIL {e}", flush=True)
            # checkpoint
            if done % 5 == 0 or done == len(items):
                out_path.write_text(
                    json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
                )

    # also stamp into _split.json for convenience
    for name, rows in (split.get("splits") or {}).items():
        for it in rows:
            z = cache.get(it["sample_id"]) or {}
            if z.get("prompt_zh"):
                it["prompt_zh"] = z["prompt_zh"]
    (hc / "_split.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"wrote": str(out_path), "ok": ok, "fail": fail, "n_cache": len(cache)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
