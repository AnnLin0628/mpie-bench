#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPIE-Bench Pilot case study:10 Complex multi-person interaction case × Multi-model drawing.

Depend on external OpenAI Compatible with raw image client (optional): Settings MPIE_LEGACY_CLIENT_PATH oriented
Contains `services.image_creation.gen_models` of Python Package root directory.
For formal evaluation, please use `code/eval/closedsource/run_closed.py`。

The result is written data/manifests/pilot_generation_results.json, picture placement
data/crops/pilot_case_study/. Resume running from breakpoint: Successful (case, model) jump over.

usage:
  export MPIE_LEGACY_CLIENT_PATH=/path/to/legacy_client_root
  python run_generation.py [--models m1,m2] [--case case_id]
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import requests

_legacy = (os.environ.get("MPIE_LEGACY_CLIENT_PATH") or "").strip()
if _legacy:
    sys.path.insert(0, _legacy)
try:
    from services.image_creation.gen_models import generate_image, SafetyRejected  # noqa: E402
except ImportError as e:
    raise SystemExit(
        "Unable to import external graphics client. Please export MPIE_LEGACY_CLIENT_PATH=... "
        "or use instead code/eval/closedsource/run_closed.py\n"
        f"ImportError: {e}"
    ) from e

BENCH = Path(".")
CASES_FILE = BENCH / "data/manifests/pilot_cases.json"
RESULTS_FILE = BENCH / "data/manifests/pilot_generation_results.json"
IMG_DIR = BENCH / "data/crops/pilot_case_study"
IMG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["gpt-image-2", "seedream-5"]


def load_results() -> dict:
    if RESULTS_FILE.exists():
        return json.load(open(RESULTS_FILE))
    return {"generated_at": "", "runs": {}}


def save_results(results: dict):
    results["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(results, open(RESULTS_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def download(url: str, dest: Path):
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--case", default="", help="Only run specified case_id")
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    cases = json.load(open(CASES_FILE))["cases"]
    if args.case:
        cases = [c for c in cases if c["case_id"] == args.case]

    results = load_results()
    runs = results["runs"]

    total = len(cases) * len(models)
    done = 0
    for case in cases:
        cid = case["case_id"]
        ref_urls = [r["url"] for r in case["refs"]]
        for model in models:
            done += 1
            key = f"{cid}::{model}"
            prev = runs.get(key)
            if prev and prev.get("ok"):
                print(f"[{done}/{total}] {key} SKIP (already ok)", flush=True)
                continue
            t0 = time.time()
            rec = {"case_id": cid, "model": model, "density": case["density"],
                   "n_person": case["n_person"], "ok": False}
            try:
                oss_url, meta = generate_image(model, case["prompt"], ref_urls=ref_urls)
                rec["image_url"] = oss_url
                rec["meta"] = {k: meta.get(k) for k in ("model", "size", "cost_usd", "latency_s")}
                raw = meta.get("raw") or {}
                rec["meta"]["actual_model"] = raw.get("actual_model")
                rec["meta"]["fallback_path"] = raw.get("fallback_path")
                fn = f"{cid}__{model}.png"
                download(oss_url, IMG_DIR / fn)
                rec["local_file"] = fn
                rec["ok"] = True
            except SafetyRejected as e:
                rec["error"] = f"SafetyRejected(by_prefilter={e.by_prefilter}): {e}"[:300]
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:300]
                rec["trace_tail"] = traceback.format_exc()[-500:]
            rec["secs"] = round(time.time() - t0, 1)
            runs[key] = rec
            save_results(results)
            status = "OK" if rec["ok"] else "FAIL " + rec.get("error", "")[:120]
            print(f"[{done}/{total}] {key} {rec['secs']}s {status}", flush=True)

    ok = sum(1 for r in runs.values() if r.get("ok"))
    print(f"\nDONE: {ok}/{len(runs)} ok -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
