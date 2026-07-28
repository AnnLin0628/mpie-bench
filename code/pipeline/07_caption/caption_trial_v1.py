#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 7 trial v1: test-set scene groups -> VLM layered prompts (10 groups, human review).

design: docs/02_pipeline_design/caption_prompt_v1.md
Input: board_state.json tset + scenes_export.json per category
Output: trial_v1.json + trial_v1.html (board /captiontrial)
Key/URL: MPIE_VLM_KEY / MPIE_VLM_URL env vars only.
Usage: python caption_trial_v1.py [--scenes 10]
"""
import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from PIL import Image

HOME = Path.home()
ROOT = HOME / "mpie_bench/data/cc0_review_full"
OUT_DIR = HOME / "mpie_bench/data/manifests/caption_trial"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YS_MODEL = os.environ.get("MPIE_VLM_MODEL", "gemini-3.5-flash")
MAX_SIDE = 1024
MAX_REFS, MAX_TGTS = 4, 3

# trial picks: (category, anchor or None=first valid test scene)
PICKS = [("harmony4d", "grappling"), ("harmony4d", "ballroom"), ("chi3d", None),
         ("hug", None), ("carry_lift", None), ("piggyback", None), ("dance_lift", None),
         ("wrestle_grapple", None), ("other_multi_person", None), ("face_to_face_talk", None)]

PROMPT = """You are writing evaluation prompts for MPIE-Bench, a multi-person interaction image-editing benchmark.

Task setup: an image-editing model will receive (a) one clean reference portrait per person, and (b) a text prompt, and must generate the TARGET image from only these inputs. Text is the ONLY control channel besides the portraits, so each prompt must carry ALL information needed to reproduce its target image.

You are given reference portraits labeled R1, R2, ... (one person each) and target frames labeled T1, T2, ... — all from the SAME scene/event. First understand the whole event using all frames together, then reverse-write one detailed prompt PER target frame.

PERSON REFERENCE RULES (critical):
- First mention of each referenced person in a prompt: appearance descriptor + tag, e.g. "the man in the gray hoodie (R1)". Afterwards use the short descriptor alone.
- The descriptor must describe the person's appearance IN THE TARGET FRAME (clothing there may differ from the portrait).
- Each subject entry must state: position in frame (left/right/center, foreground/background), body orientation and posture, the specific action, gaze/expression.
- Never write bare "person 1" / "the first person" without an appearance descriptor.
- If a referenced person is not visible in a target frame, omit them from subjects and list their tag in "absent".

BYSTANDER RULES:
- People visible in a target but NOT among the references are bystanders. Describe them in ONE short clause in the bystanders layer: count + coarse appearance + location + activity, e.g. "in the background, two blurred passersby in dark clothing walk past".
- Never give a bystander an (R#) tag, never describe their facial identity, never involve them in the main interaction.
- No bystanders -> null.

LAYERS per target (fill all, then assemble):
1. setting: location type, indoor/outdoor, time of day, ambience
2. camera: shot scale (close-up/medium/full/wide), camera height and angle, subject distance, depth-of-field feel
3. subjects: one entry per visible referenced person (rules above)
4. interaction: what they do TOGETHER — spatial relation (who is left/right/behind), precise contact points (e.g. right hand on partner's waist), direction of force/motion
5. bystanders: rules above
6. lighting_color: light source direction/quality, palette, shadows, mood
7. style: photographic realism descriptor (e.g. "candid documentary photograph", "cinematic still")

FINAL PROMPT assembly: one fluent English paragraph of 90-160 words, in this order: style+camera -> setting -> subjects & interaction (the core, most detailed) -> bystanders -> lighting/color. Present tense, declarative (describe the image; no imperative commands). Do not invent anything not visible. No personal names. No lens/f-stop numbers.

SAFETY: if any person appears to be a minor, set flag_underage=true and still complete the annotation.

Return ONLY a JSON object (no markdown fence):
{"scene_summary": "<2-3 sentences: the event across all frames>",
 "targets": [
  {"target": "T1",
   "layers": {"setting": "...", "camera": "...",
    "subjects": [{"ref": "R1", "descriptor": "...", "position": "...", "pose_action": "...", "gaze_expression": "..."}],
    "absent": [],
    "interaction": "...",
    "bystanders": null,
    "lighting_color": "...", "style": "..."},
   "prompt": "<assembled paragraph>",
   "confidence": "high|medium|low",
   "flag_underage": false}
 ]}"""


def load_key() -> str:
    k = os.environ.get("MPIE_VLM_KEY") or os.environ.get("AI_GATEWAY_KEY")
    if not k:
        raise RuntimeError("Missing MPIE_VLM_KEY(or AI_GATEWAY_KEY）")
    return k


def load_url() -> str:
    url = (os.environ.get("MPIE_VLM_URL") or "").strip()
    if not url:
        raise RuntimeError("Missing MPIE_VLM_URL (OpenAI-compatible chat/completions)")
    return url


def pick_scene(cat, anchor=None):
    """Return (cat, anchor, ref_fns, tgt_fns) or None. Test-set scenes; starred refs first."""
    st_p = ROOT / cat / "board_state.json"
    exp_p = ROOT / cat / "scenes_export.json"
    if not (st_p.exists() and exp_p.exists()):
        return None
    st = json.loads(st_p.read_text())
    tset = set(st.get("tset") or [])
    if not tset:
        return None
    dtgt, dref = set(st.get("dtgt") or []), set(st.get("del") or [])
    star = st.get("star") or {}
    into = {m[0]: m[1] for m in (st.get("smerge") or []) if len(m) == 2}

    def fin(a):
        seen = set()
        while a in into and a not in seen:
            seen.add(a)
            a = into[a]
        return a

    marked = {fin(a) for a in tset}
    for s in json.loads(exp_p.read_text()):
        root = fin(s["anchor"])
        if root not in marked or (anchor and root != anchor):
            continue
        tgts = [t for t in s["targets"] if t not in dtgt][:MAX_TGTS]
        refs = []
        for a in s["actors"]:
            live = [fn for fn in a["refs"] if fn not in dref]
            if live:
                chosen = star.get(a["id"])
                refs.append(chosen if chosen in live else live[0])
        if tgts and refs:
            return cat, root, refs[:MAX_REFS], tgts
    return None


def b64(cat, fn):
    im = Image.open(ROOT / cat / "flat" / fn).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def call_scene(cat, anchor, refs, tgts, retries=5):
    content = []
    for i, fn in enumerate(refs):
        content.append({"type": "text", "text": f"Reference portrait R{i+1}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(cat, fn)}"}})
    for i, fn in enumerate(tgts):
        content.append({"type": "text", "text": f"Target frame T{i+1}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64(cat, fn)}"}})
    content.append({"type": "text", "text": PROMPT})
    payload = {"model": YS_MODEL, "max_tokens": 8192, "temperature": 0.3,
               "messages": [{"role": "user", "content": content}]}
    for attempt in range(retries):
        try:
            r = requests.post(load_url(), json=payload, timeout=180,
                              headers={"Authorization": f"Bearer {load_key()}"})
            if r.status_code == 429:
                time.sleep(min(3 * 2 ** attempt, 60))
                continue
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0))
        except Exception as e:
            if attempt == retries - 1:
                return {"error": f"{type(e).__name__}: {e}"[:300]}
            time.sleep(2 ** attempt + 1)
    return {"error": "exhausted"}


def build_html(results):
    parts = ["""<!DOCTYPE html><html lang=zh><head><meta charset=utf-8><title>caption trial v1 review</title><style>
body{font-family:-apple-system,"PingFang SC",sans-serif;background:#f4f5f8;color:#1f2430;margin:0;padding:16px}
.sc{background:#fff;border:1px solid #e8eaee;border-radius:12px;margin:14px 0;padding:14px}
.sc h3{margin:0 0 8px} .imgs{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}
.imgs figure{margin:0;text-align:center} .imgs img{height:170px;border-radius:8px;background:#111}
.imgs figcaption{font-size:11px;color:#6b7280}
.tgt{border-top:1px dashed #e5e7eb;padding:10px 0;display:flex;gap:14px;align-items:flex-start}
.tgt img{height:230px;border-radius:8px;background:#111}
.pr{flex:1;font-size:14px;line-height:1.55;background:#fafbfc;border:1px solid #eef0f3;border-radius:8px;padding:10px}
.pr b.rtag{color:#8b5cf6} .meta{font-size:12px;color:#6b7280;margin-top:6px}
details{margin-top:6px;font-size:12px} pre{white-space:pre-wrap;background:#f6f7f9;padding:8px;border-radius:6px}
.low{color:#ef4444;font-weight:700} .summ{font-size:13px;color:#374151;background:#f0f5ff;border-radius:8px;padding:8px}
</style></head><body><h2>Stage 7 prompt trial v1 — human review</h2>
<p style="font-size:13px;color:#6b7280">Review: ① action/contact (C3/C4 limb ownership) ② (R#) bindings ③ bystanders ④ blind-imagine from prompt ⑤ no hallucination.</p>"""]
    for r in results:
        cat, anchor = r["cat"], r["anchor"]
        ref_html = "".join(
            f'<figure><img loading=lazy src="/cc0img/{cat}/{fn}"><figcaption>R{i+1}</figcaption></figure>'
            for i, fn in enumerate(r["refs"]))
        parts.append(f'<div class=sc><h3>{cat} · <code>{anchor}</code></h3><div class=imgs>{ref_html}</div>')
        if "error" in r["vlm"]:
            parts.append(f'<p class=low>Call failed: {r["vlm"]["error"]}</p></div>')
            continue
        parts.append(f'<div class=summ><b>scene_summary:</b> {r["vlm"].get("scene_summary","")}</div>')
        tmap = {f"T{i+1}": fn for i, fn in enumerate(r["tgts"])}
        for t in r["vlm"].get("targets", []):
            fn = tmap.get(t.get("target"), "")
            prompt = t.get("prompt", "")
            prompt = re.sub(r"\((R\d+)\)", r'<b class=rtag>(\1)</b>', prompt)
            conf = t.get("confidence", "?")
            conf_html = f'<span class={"low" if conf == "low" else ""}>{conf}</span>'
            layers = json.dumps(t.get("layers", {}), ensure_ascii=False, indent=1)
            parts.append(
                f'<div class=tgt><figure style="margin:0;text-align:center"><img loading=lazy src="/cc0img/{cat}/{fn}">'
                f'<figcaption style="font-size:11px;color:#6b7280">{t.get("target")}</figcaption></figure>'
                f'<div class=pr>{prompt}<div class=meta>confidence: {conf_html} · Number of poems≈{len(prompt.split())} · underage: {t.get("flag_underage")}</div>'
                f'<details><summary>Layer JSON</summary><pre>{layers}</pre></details></div></div>')
        parts.append("</div>")
    parts.append("</body></html>")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=10)
    args = ap.parse_args()

    picked = []
    for cat, anchor in PICKS[:args.scenes]:
        p = pick_scene(cat, anchor)
        if p:
            picked.append(p)
        else:
            print(f"skip {cat}({anchor}): no valid test-set scene")
    print(f"picked {len(picked)} groups: " + ", ".join(f"{c}:{a}" for c, a, _, _ in picked))

    results = []
    with ThreadPoolExecutor(4) as ex:
        futs = {ex.submit(call_scene, *p): p for p in picked}
        for fut, (cat, anchor, refs, tgts) in futs.items():
            v = fut.result()
            results.append({"cat": cat, "anchor": anchor, "refs": refs, "tgts": tgts, "vlm": v})
            tag = "ERR " + v["error"][:60] if "error" in v else f'{len(v.get("targets", []))} prompts'
            print(f"[{cat}:{anchor}] {tag}", flush=True)

    order = {(c, a): i for i, (c, a, _, _) in enumerate(picked)}
    results.sort(key=lambda r: order[(r["cat"], r["anchor"])])
    (OUT_DIR / "trial_v1.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    (OUT_DIR / "trial_v1.html").write_text(build_html(results))
    n_ok = sum(1 for r in results if "error" not in r["vlm"])
    print(f"done {n_ok}/{len(results)} groups -> {OUT_DIR}/trial_v1.html (/captiontrial)")


if __name__ == "__main__":
    main()
