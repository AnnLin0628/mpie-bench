#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 7: VLM structured caption back-write.

Call VLM on selected keyframes → JSON (13 interaction types + C0-C3 + per-person actions + edit instruction),
write captions table. confidence=low or parse fail → needs_review=1 for human review.
Primary model via MPIE_VLM_URL (OpenAI-compatible chat/completions); backup via env vars.
Smoke test on any image dir without DB: --smoke <dir>

Usage:
  export MPIE_VLM_URL=https://<host>/v1/chat/completions
  export MPIE_VLM_KEY=...
  python caption_client.py --db ~/mpie_data/manifests/mpie.db [--workers 6]
"""
import argparse
import base64
import io
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.manifest import connect, upsert, rows  # noqa: E402

import os


def _require_vlm_url() -> str:
    url = (os.environ.get("MPIE_VLM_URL") or "").strip()
    if not url:
        raise RuntimeError("Missing MPIE_VLM_URL (OpenAI-compatible chat/completions endpoint)")
    return url


YS_URL = os.environ.get("MPIE_VLM_URL", "")
AI_GATEWAY_KEY = os.environ.get("MPIE_VLM_KEY", "")              # Required: export MPIE_VLM_KEY=...
YS_MODEL = os.environ.get("MPIE_VLM_MODEL", "gemini-3.5-flash")
MAX_SIDE = 1024

INTERACTION_VOCAB = ["hug", "dance", "fight_combat", "wrestle_grapple", "carry_lift",
                     "handshake", "high_five", "piggyback", "hand_hold",
                     "arm_around_shoulder", "face_to_face_talk", "restrain_pin", "other"]

PROMPT = """You are annotating a video frame for a multi-person interaction image-editing dataset.

Analyze the image and return ONLY a JSON object (no markdown fence):
{{
  "n_person": <int, people clearly visible>,
  "interaction_type": <one of {vocab}>,
  "contact_density_level": <"C0" no contact | "C1" hand-level (handshake/high-five) | "C2" sustained point/line or light torso (hand-hold/arm-around/dance) | "C3" high contact: weight-bearing OR full-body entanglement/combat (hug/lift/piggyback/wrestle/fight/pin)>,
  "per_person_role": [<one short phrase per person describing their action, e.g. "man lifting partner by the waist">],
  "contact_points": [<body-part pairs in contact, e.g. "A.hands-B.waist">],
  "scene_caption": "<one sentence: style, lighting, camera framing, background>",
  "edit_instruction": "<an imperative editing instruction in natural user tone that would produce this image from individual portraits of these people, in English>",
  "confidence": <"high"|"medium"|"low" — low if occlusion makes limb ownership ambiguous>,
  "flag_underage": <true if ANY person appears to be a minor or childlike, else false>
}}"""


def img_b64(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_vlm(img_path: Path, retries: int = 6) -> dict:
    content = [{"type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64(img_path)}"}},
               {"type": "text", "text": PROMPT.format(vocab=INTERACTION_VOCAB)}]
    payload = {"model": YS_MODEL, "max_tokens": 8192, "temperature": 0.1,
               "messages": [{"role": "user", "content": content}]}
    if not AI_GATEWAY_KEY:
        raise RuntimeError("Missing MPIE_VLM_KEY")
    url = YS_URL or _require_vlm_url()
    for attempt in range(retries):
        try:
            r = requests.post(url, json=payload, timeout=120,
                              headers={"Authorization": f"Bearer {AI_GATEWAY_KEY}"})
            if r.status_code == 429:
                time.sleep(min(3 * 2 ** attempt, 60))
                continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0))
        except Exception as e:
            if attempt == retries - 1:
                return {"error": f"{type(e).__name__}: {e}"[:200]}
            time.sleep(2 ** attempt + 1)
    return {"error": "exhausted"}


def caption_one(conn, kf_id: str, frame_path: str) -> str:
    c = call_vlm(Path(frame_path))
    bad = "error" in c or c.get("interaction_type") not in INTERACTION_VOCAB \
        or c.get("contact_density_level") not in ("C0", "C1", "C2", "C3")
    upsert(conn, "captions", {
        "kf_id": kf_id, "n_person": c.get("n_person"),
        "interaction_type": c.get("interaction_type"),
        "contact_density_level": c.get("contact_density_level"),
        "per_person_role": c.get("per_person_role"),
        "contact_points": c.get("contact_points"),
        "scene_caption": c.get("scene_caption"),
        "edit_instruction": c.get("edit_instruction"),
        "confidence": c.get("confidence"),
        "flag_underage": 1 if c.get("flag_underage") else 0,
        "raw_json": c,
        "needs_review": 1 if (bad or c.get("confidence") == "low") else 0})
    return "review" if (bad or c.get("confidence") == "low") else "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", default="", help="Skip DB; smoke-test on an image directory and print")
    args = ap.parse_args()

    if args.smoke:
        for p in sorted(Path(args.smoke).glob("*.jpg"))[:5] + sorted(Path(args.smoke).glob("*.png"))[:5]:
            print("==", p.name)
            print(json.dumps(call_vlm(p), ensure_ascii=False, indent=2)[:800])
        return

    conn = connect(args.db)
    todo = rows(conn, """SELECT k.kf_id, k.frame_path FROM keyframes k
                         LEFT JOIN captions c ON k.kf_id=c.kf_id
                         WHERE k.selected=1 AND c.kf_id IS NULL""")
    print(f"to caption: {len(todo)}")
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(call_vlm, Path(t["frame_path"])): t for t in todo}
        # Sequential DB writes (sqlite single writer)
        for fut in list(futs):
            t = futs[fut]
            c = fut.result()
            bad = "error" in c or c.get("interaction_type") not in INTERACTION_VOCAB
            upsert(conn, "captions", {
                "kf_id": t["kf_id"], "n_person": c.get("n_person"),
                "interaction_type": c.get("interaction_type"),
                "contact_density_level": c.get("contact_density_level"),
                "per_person_role": c.get("per_person_role"),
                "contact_points": c.get("contact_points"),
                "scene_caption": c.get("scene_caption"),
                "edit_instruction": c.get("edit_instruction"),
                "confidence": c.get("confidence"),
                "flag_underage": 1 if c.get("flag_underage") else 0,
                "raw_json": c,
                "needs_review": 1 if (bad or c.get("confidence") == "low") else 0})
            done += 1
            if done % 50 == 0:
                conn.commit()
                print(f"{done}/{len(todo)}", flush=True)
    conn.commit()
    nr = rows(conn, "SELECT COUNT(*) c FROM captions WHERE needs_review=1")[0]["c"]
    print(f"caption done: {done}, needs_review={nr}")


if __name__ == "__main__":
    main()
