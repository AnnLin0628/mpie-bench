#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPIE-Bench Pilot case:VLM Structured scoring (six-dimensional proxy indicators).

Generate a picture for each: [Generate graph, Reference pictureA, Reference pictureB(, Reference pictureC)] + Editing instructions → gemini-3.5-flash
structured JSON output Count/ID/Anat/Inter/Instr/Qual six dimensional points + Dimension-by-dimensional text basis.
output data/manifests/pilot_scores.json. Resume running from breakpoint.

Notice: This is a proxy indicator, It is not a formal indicator used in the paper.(ArcFace/DWPose/SMPL-X Leave it to formality pipeline)。
usage: python3 score_vlm.py
"""
import base64
import io
import json
import re
import time
from pathlib import Path

import requests
from PIL import Image

BENCH = Path(".")
CASES_FILE = BENCH / "data/manifests/pilot_cases.json"
RESULTS_FILE = BENCH / "data/manifests/pilot_generation_results.json"
SCORES_FILE = BENCH / "data/manifests/pilot_scores.json"
IMG_DIR = BENCH / "data/crops/pilot_case_study"
REF_CACHE = IMG_DIR / "_refs"
REF_CACHE.mkdir(parents=True, exist_ok=True)

# VLM Referee gateway:URL/Key Use environment variables (no hard coding)
import os
API_URL = os.environ.get("MPIE_VLM_URL", "")
API_KEY = os.environ.get("MPIE_VLM_KEY", "")
MODEL = os.environ.get("MPIE_VLM_MODEL", "gemini-3.5-flash")
MAX_SIDE = 1024

SESSION = requests.Session()
if API_KEY:
    SESSION.headers.update({"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"})

PROMPT_TMPL = """You are a strict image-editing evaluator for MULTI-PERSON INTERACTION scenes.

Image 1 is a GENERATED image. The remaining {n_refs} image(s) are character REFERENCE portraits: {ref_names}.
The generation instruction was: "{instruction}"
The expected number of people in the generated image: {n_person}.

Evaluate Image 1 on six dimensions. Be harsh and specific — anatomical and interaction errors in contact regions (hands, arms, interlocked limbs, occluded body parts) are the focus. Look carefully at every hand, every arm, every leg: count fingers, check limb ownership (does each limb connect to the right body?), check for body interpenetration, impossible joint angles, merged or duplicated limbs, floating body parts.

Return ONLY a JSON object (no markdown fence, no commentary):
{{
  "count_detected": <int, how many people are actually visible>,
  "count_correct": <bool>,
  "id_scores": [<int 1-5 per reference character, in the given order: does that character appear in the generated image and resemble the reference?>],
  "anat_score": <int 1-5, anatomical correctness: 5=flawless bodies, 1=severe extra/missing/merged limbs>,
  "anat_issues": ["<specific issue with location, empty list if none>"],
  "inter_score": <int 1-5, interaction plausibility: is the described interaction physically achieved? contact points correct? no interpenetration? 5=fully plausible, 1=interaction fails or bodies fused>,
  "inter_issues": ["<specific issue, empty list if none>"],
  "instr_score": <int 1-5, does the image follow the instruction (action, scene, framing)?>,
  "qual_score": <int 1-5, general image quality/sharpness/composition, ignore anatomy here>,
  "overall_notes": "<one sentence summary>"
}}"""


def img_b64(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def fetch_ref(url: str) -> Path:
    fn = REF_CACHE / (url.rstrip("/").split("/")[-1])
    if not fn.exists():
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        fn.write_bytes(r.content)
    return fn


def parse_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no json in: {text[:200]}")
    return json.loads(m.group(0))


def score_one(gen_img: Path, ref_paths: list, ref_names: list, instruction: str,
              n_person: int, retries: int = 6) -> dict:
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64(gen_img)}"}}]
    for rp in ref_paths:
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64(rp)}"}})
    prompt = PROMPT_TMPL.format(n_refs=len(ref_paths),
                                ref_names=", ".join(ref_names),
                                instruction=instruction, n_person=n_person)
    content.append({"type": "text", "text": prompt})
    payload = {"model": MODEL, "max_tokens": 8192, "temperature": 0.1,
               "messages": [{"role": "user", "content": content}]}
    last = ""
    for attempt in range(retries):
        try:
            r = SESSION.post(API_URL, json=payload, timeout=120)
            if r.status_code == 429:
                time.sleep(min(3 * 2 ** attempt, 60))
                continue
            r.raise_for_status()
            text = (r.json()["choices"][0]["message"].get("content") or "").strip()
            return parse_json(text)
        except Exception as e:
            last = f"{type(e).__name__}: {e}"[:200]
            if attempt < retries - 1:
                time.sleep(2 ** attempt + 1)
    return {"error": last}


def main():
    if not API_URL or not API_KEY:
        raise SystemExit(
            "Lack MPIE_VLM_URL / MPIE_VLM_KEY（OpenAI compatible chat/completions）"
        )
    cases = {c["case_id"]: c for c in json.load(open(CASES_FILE))["cases"]}
    runs = json.load(open(RESULTS_FILE))["runs"]

    scores = {}
    if SCORES_FILE.exists():
        scores = json.load(open(SCORES_FILE)).get("scores", {})

    todo = [(k, v) for k, v in sorted(runs.items())
            if v.get("ok") and (k not in scores or scores[k].get("error"))]
    print(f"to score: {len(todo)}")
    for i, (key, run) in enumerate(todo, 1):
        case = cases[run["case_id"]]
        gen_img = IMG_DIR / run["local_file"]
        ref_paths = [fetch_ref(r["url"]) for r in case["refs"]]
        ref_names = [f"Image {j+2} = character '{r['name']}'" for j, r in enumerate(case["refs"])]
        t0 = time.time()
        s = score_one(gen_img, ref_paths, ref_names, case["prompt"], case["n_person"])
        s["_meta"] = {"case_id": run["case_id"], "model": run["model"],
                      "density": case["density"], "n_person": case["n_person"],
                      "secs": round(time.time() - t0, 1)}
        scores[key] = s
        json.dump({"scored_at": time.strftime("%Y-%m-%d %H:%M:%S"), "judge_model": MODEL,
                   "scores": scores},
                  open(SCORES_FILE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        ok = "ERR " + s.get("error", "") if s.get("error") else \
            f"anat={s.get('anat_score')} inter={s.get('inter_score')} count={s.get('count_detected')}"
        print(f"[{i}/{len(todo)}] {key} {s['_meta']['secs']}s {ok}", flush=True)

    print(f"\nDONE -> {SCORES_FILE}")


if __name__ == "__main__":
    main()
