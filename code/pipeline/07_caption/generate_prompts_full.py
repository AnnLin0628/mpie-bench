#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 7 full reverse-caption prompts (evolved from caption_trial_v1.py).

Changes:
- Drop hardcoded PICKS; auto-scan test-set scenes under all categories
- Lift image-count caps (truncate via --max_refs/--max_tgts)
- Support resume (--resume) and threaded concurrency (--workers)
- Add --limit to process only the first N scenes for debugging
- Emit per-category JSON chunks plus a review HTML

Usage:
    python generate_prompts_full.py --mode test --limit 10 --workers 4
    python generate_prompts_full.py --mode test --max_refs 6 --max_tgts 4 --limit 20 --resume
"""
import argparse
import base64
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

import requests
from PIL import Image

# ---------- path config ----------
HOME = Path.home()
ROOT = HOME / "mpie_bench/data/cc0_review_full"      # data root
OUT_DIR = HOME / "mpie_bench/data/manifests/prompts_full"  # output dir
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- API config (env vars only) ----------
YS_MODEL = os.environ.get("MPIE_VLM_MODEL", "gemini-3.5-flash")
MAX_IMAGE_SIDE = 1024


def load_key() -> str:
    k = os.environ.get("MPIE_VLM_KEY") or os.environ.get("AI_GATEWAY_KEY")
    if not k:
        raise RuntimeError("missing MPIE_VLM_KEY (or AI_GATEWAY_KEY)")
    return k


def load_url() -> str:
    url = (os.environ.get("MPIE_VLM_URL") or "").strip()
    if not url:
        raise RuntimeError("missing MPIE_VLM_URL (OpenAI-compatible chat/completions)")
    return url

# ---------- Prompt (same as prior version) ----------
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

# ---------- helpers ----------
def img_b64(cat, filename, max_side=MAX_IMAGE_SIDE, quality=90):
    """Encode image as base64 (auto-resize); may downscale to bypass PROHIBITED_CONTENT."""
    path = ROOT / cat / "flat" / filename
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()

def get_test_scenes(cat):
    """
    Read scenes_export.json and board_state.json under category cat.
    Return test-set (tset) scenes as (anchor, refs, tgts).

    Semantics aligned with the review frontend test split:
    - resolve smerge chains to the final root
    - merge target images across multiple exports under the same root
    - drop dtgt; drop deleted refs; prefer starred refs
    Corpus total should be 2500 target images.
    """
    board_path = ROOT / cat / "board_state.json"
    export_path = ROOT / cat / "scenes_export.json"
    if not (board_path.exists() and export_path.exists()):
        return []

    board = json.loads(board_path.read_text())
    export = json.loads(export_path.read_text())

    tset = set(board.get("tset") or [])
    dtgt = set(board.get("dtgt") or [])
    dref = set(board.get("del") or [])
    star = board.get("star") or {}
    into = {m[0]: m[1] for m in (board.get("smerge") or []) if len(m) == 2}

    def resolve(a):
        seen = set()
        while a in into and a not in seen:
            seen.add(a)
            a = into[a]
        return a

    test_roots = {resolve(a) for a in tset}
    # root -> {"tgts": [...], "refs": [...]}  ordered dedupe/merge
    picked = {}
    for s in export:
        root = resolve(s["anchor"])
        if root not in test_roots:
            continue
        g = picked.setdefault(root, {"tgts": [], "refs": [], "_tgt_set": set(), "_ref_set": set()})
        for t in s["targets"]:
            if t in dtgt or t in g["_tgt_set"]:
                continue
            g["_tgt_set"].add(t)
            g["tgts"].append(t)
        for actor in s["actors"]:
            live = [fn for fn in actor["refs"] if fn not in dref]
            if not live:
                continue
            chosen = star.get(actor["id"])
            ref = chosen if chosen in live else live[0]
            if ref in g["_ref_set"]:
                continue
            g["_ref_set"].add(ref)
            g["refs"].append(ref)

    scenes = []
    for root, g in picked.items():
        if g["tgts"] and g["refs"]:
            scenes.append((root, g["refs"], g["tgts"]))
    return scenes

def _call_vlm_chunk(cat, refs, tgts, t_offset=0, retries=8, max_side=MAX_IMAGE_SIDE, quality=90):
    """One VLM call; target labels start at T{t_offset+1}.

    503/429: exponential backoff retry;
    400 payload: return payload_too_large for upstream chunk shrink;
    400 PROHIBITED_CONTENT: return prohibited_content for downscale retry.
    """
    content = []
    for i, fn in enumerate(refs):
        content.append({"type": "text", "text": f"Reference portrait R{i+1}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64(cat, fn, max_side, quality)}"}})
    for i, fn in enumerate(tgts):
        content.append({"type": "text", "text": f"Target frame T{t_offset + i + 1}:"})
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64(cat, fn, max_side, quality)}"}})
    content.append({"type": "text", "text": PROMPT})

    payload = {
        "model": YS_MODEL,
        "max_tokens": 8192,
        "temperature": 0.3,
        "messages": [{"role": "user", "content": content}],
    }

    for attempt in range(retries):
        try:
            resp = requests.post(
                load_url(), json=payload, timeout=300,
                headers={"Authorization": f"Bearer {load_key()}"},
            )
            if resp.status_code in (429, 503, 502, 504):
                wait = min(5 * (2 ** attempt), 120)
                print(f"  transient HTTP {resp.status_code}, sleep {wait}s (attempt {attempt+1}/{retries})", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code == 400:
                body = (resp.text or "")[:300]
                if "PROHIBITED_CONTENT" in body or "prompt_blocked" in body:
                    return {"error": f"HTTPError: 400 prohibited_content refs={len(refs)} tgts={len(tgts)} side={max_side} q={quality}"}
                return {"error": f"HTTPError: 400 payload_too_large refs={len(refs)} tgts={len(tgts)}"}
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", text, re.S)
            if not m:
                raise ValueError(f"no JSON object in VLM response: {text[:200]!r}")
            return json.loads(m.group(0))
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if attempt == retries - 1:
                return {"error": msg[:300]}
            wait = min(2 ** attempt + 1, 60)
            print(f"  retry after {msg[:80]} sleep {wait}s", flush=True)
            time.sleep(wait)
    return {"error": "exhausted"}


# Stepwise downscale on PROHIBITED_CONTENT (512/q50 bypasses some false blocks)
_SAFE_ENCODE_LADDER = [(768, 70), (512, 50), (384, 40), (256, 30)]


def _call_chunk_adaptive(cat, refs, tgts, t_offset, retries, max_refs_fallback=4):
    """On 400 payload: halve target chunk then truncate refs; on prohibited: downscale retry."""
    result = _call_vlm_chunk(cat, refs, tgts, t_offset=t_offset, retries=retries)
    if "error" not in result:
        return result
    err = result["error"]

    if "prohibited_content" in err:
        for side, q in _SAFE_ENCODE_LADDER:
            print(f"  400 prohibited→encode side={side} q={q} refs={len(refs)} tgts={len(tgts)}", flush=True)
            result = _call_vlm_chunk(
                cat, refs, tgts, t_offset=t_offset, retries=retries, max_side=side, quality=q,
            )
            if "error" not in result:
                return result
            if "prohibited_content" not in result.get("error", ""):
                err = result["error"]
                break
            err = result["error"]
        else:
            return result

    if "payload_too_large" not in err and "prohibited_content" not in err:
        return result

    if "payload_too_large" in err:
        if len(tgts) > 1:
            mid = max(1, len(tgts) // 2)
            print(f"  400→split chunk {len(tgts)}→{mid}+{len(tgts)-mid}", flush=True)
            left = _call_chunk_adaptive(cat, refs, tgts[:mid], t_offset, retries, max_refs_fallback)
            if "error" in left:
                return left
            right = _call_chunk_adaptive(cat, refs, tgts[mid:], t_offset + mid, retries, max_refs_fallback)
            if "error" in right:
                return right
            merged = (left.get("targets") or []) + (right.get("targets") or [])
            return {"scene_summary": left.get("scene_summary") or right.get("scene_summary") or "", "targets": merged}
        # Single-target still payload-400: cut reference images
        if len(refs) > max_refs_fallback:
            print(f"  400→trim refs {len(refs)}→{max_refs_fallback}", flush=True)
            return _call_vlm_chunk(cat, refs[:max_refs_fallback], tgts, t_offset=t_offset, retries=retries)
        if len(refs) > 1:
            print(f"  400→trim refs {len(refs)}→1", flush=True)
            return _call_vlm_chunk(cat, refs[:1], tgts, t_offset=t_offset, retries=retries)
    return result


def call_vlm_scene(cat, anchor, refs, tgts, retries=8, chunk_tgts=4):
    """Call VLM; chunk targets by chunk_tgts and auto-shrink on HTTP 400."""
    if not tgts:
        return {"error": "empty tgts"}
    size = chunk_tgts if chunk_tgts and chunk_tgts > 0 else len(tgts)
    summaries = []
    merged_targets = []
    for start in range(0, len(tgts), size):
        chunk = tgts[start:start + size]
        result = _call_chunk_adaptive(cat, refs, chunk, t_offset=start, retries=retries)
        if "error" in result:
            return result
        if result.get("scene_summary"):
            summaries.append(result["scene_summary"])
        for t in result.get("targets") or []:
            merged_targets.append(t)
    def _tkey(t):
        m = re.match(r"T(\d+)", str(t.get("target", "")))
        return int(m.group(1)) if m else 10**9
    merged_targets.sort(key=_tkey)
    return {
        "scene_summary": summaries[0] if summaries else "",
        "targets": merged_targets,
    }

def build_html(results_by_cat):
    """Build a review HTML, similar to the prior version but grouped by category."""
    parts = ["""<!DOCTYPE html><html lang=zh><head><meta charset=utf-8>
    <title>Full Prompt Reverse-Caption Review</title>
    <style>
    body{font-family:-apple-system,"PingFang SC",sans-serif;background:#f4f5f8;color:#1f2430;margin:0;padding:16px}
    .cat{background:#fff;border:1px solid #e8eaee;border-radius:12px;margin:14px 0;padding:14px}
    .cat h2{margin:0 0 8px;color:#2c3e50}
    .sc{background:#fafbfc;border-left:4px solid #8b5cf6;margin:8px 0;padding:8px 12px}
    .sc h3{margin:0 0 6px}
    .imgs{display:flex;gap:8px;flex-wrap:wrap;margin:6px 0}
    .imgs figure{margin:0;text-align:center}
    .imgs img{height:150px;border-radius:8px;background:#111}
    .imgs figcaption{font-size:11px;color:#6b7280}
    .tgt{border-top:1px dashed #e5e7eb;padding:10px 0;display:flex;gap:14px;align-items:flex-start}
    .tgt img{height:200px;border-radius:8px;background:#111}
    .pr{flex:1;font-size:14px;line-height:1.55;background:#fafbfc;border:1px solid #eef0f3;border-radius:8px;padding:10px}
    .pr b.rtag{color:#8b5cf6}
    .meta{font-size:12px;color:#6b7280;margin-top:6px}
    details{margin-top:6px;font-size:12px}
    pre{white-space:pre-wrap;background:#f6f7f9;padding:8px;border-radius:6px}
    .low{color:#ef4444;font-weight:700}
    .summ{font-size:13px;color:#374151;background:#f0f5ff;border-radius:8px;padding:8px}
    </style></head><body>
    <h1>Full Prompt Reverse-Caption Results — Human Review</h1>
    <p style="font-size:13px;color:#6b7280">Review checklist: (1) actions/contact match the image (2) (R#) bindings point to the right people (3) bystander generalization (4) sufficient detail (5) no fabrication.</p>
    """]

    for cat, items in results_by_cat.items():
        parts.append(f'<div class=cat><h2>{cat}</h2>')
        for item in items:
            refs, tgts, vlm = item["refs"], item["tgts"], item["vlm"]
            parts.append(f'<div class=sc><h3>Anchor: {item["anchor"]}</h3>')
            # show reference images
            ref_html = "".join(
                f'<figure><img loading=lazy src="/cc0img/{cat}/{fn}"><figcaption>R{i+1}</figcaption></figure>'
                for i, fn in enumerate(refs)
            )
            parts.append(f'<div class=imgs>{ref_html}</div>')
            if "error" in vlm:
                parts.append(f'<p class=low>call failed: {vlm["error"]}</p></div>')
                continue
            parts.append(f'<div class=summ><b>scene_summary:</b> {vlm.get("scene_summary","")}</div>')
            tmap = {f"T{i+1}": fn for i, fn in enumerate(tgts)}
            for t in vlm.get("targets", []):
                fn = tmap.get(t.get("target"), "")
                prompt = t.get("prompt", "")
                prompt = re.sub(r"\((R\d+)\)", r'<b class=rtag>(\1)</b>', prompt)
                conf = t.get("confidence", "?")
                conf_html = f'<span class={"low" if conf=="low" else ""}>{conf}</span>'
                layers = json.dumps(t.get("layers", {}), ensure_ascii=False, indent=1)
                parts.append(
                    f'<div class=tgt><figure style="margin:0;text-align:center"><img loading=lazy src="/cc0img/{cat}/{fn}">'
                    f'<figcaption style="font-size:11px;color:#6b7280">{t.get("target")}</figcaption></figure>'
                    f'<div class=pr>{prompt}<div class=meta>confidence: {conf_html} · words≈{len(prompt.split())} · underage: {t.get("flag_underage")}</div>'
                    f'<details><summary>layered JSON</summary><pre>{layers}</pre></details></div></div>')
            parts.append('</div>')
        parts.append('</div>')
    parts.append("</body></html>")
    return "".join(parts)

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Full reverse-caption prompts")
    parser.add_argument("--mode", choices=["test", "all"], default="test",
                        help="only 'test' (test set) is implemented; 'all' is not yet supported")
    parser.add_argument("--workers", type=int, default=2,
                        help="worker concurrency (2 recommended; higher risks HTTP 503)")
    parser.add_argument("--max_refs", type=int, default=0,
                        help="max reference images per scene (0=unlimited)")
    parser.add_argument("--max_tgts", type=int, default=0,
                        help="max target images per scene (0=unlimited; keep 0 for full ~2500 targets)")
    parser.add_argument("--chunk_tgts", type=int, default=4,
                        help="max targets per VLM request; chunk beyond this; auto-split further on HTTP 400")
    parser.add_argument("--resume", action="store_true",
                        help="resume: skip scenes that already have complete successful targets; rerun truncated/failed")
    parser.add_argument("--until-complete", action="store_true",
                        help="outer loop: repeat --resume until all scenes complete or two rounds make no progress")
    parser.add_argument("--limit", type=int, default=0,
                        help="limit number of scenes (0=unlimited); useful to run the first N for testing")
    args = parser.parse_args()

    if args.mode != "test":
        print("warning: only --mode test is supported; full mode not implemented; switched to test")
        args.mode = "test"

    def scene_ok(item):
        return isinstance(item.get("vlm"), dict) and "error" not in item["vlm"]

    def collect_all_scenes():
        cats = sorted(d.name for d in ROOT.iterdir() if d.is_dir())
        print(f"categories found: {cats}")
        all_scenes = []
        for cat in cats:
            scenes = get_test_scenes(cat)
            if not scenes:
                print(f"category {cat} has no test-set scenes")
                continue
            if args.max_refs > 0:
                scenes = [(anchor, refs[:args.max_refs], tgts) for anchor, refs, tgts in scenes]
            if args.max_tgts > 0:
                scenes = [(anchor, refs, tgts[:args.max_tgts]) for anchor, refs, tgts in scenes]
            for anchor, refs, tgts in scenes:
                all_scenes.append((cat, anchor, refs, tgts))
        print(f"found {len(all_scenes)} scenes / {sum(len(s[3]) for s in all_scenes)} targets")
        return all_scenes

    def load_done_ok(desired_tgts):
        done_ok = set()
        if not args.resume and not args.until_complete:
            return done_ok
        for f in OUT_DIR.glob("prompts_*.json"):
            for item in json.loads(f.read_text()):
                key = (item["cat"], item["anchor"])
                if key not in desired_tgts or not scene_ok(item):
                    continue
                if set(item.get("tgts") or []) == set(desired_tgts[key]):
                    n_vlm = len((item.get("vlm") or {}).get("targets") or [])
                    if n_vlm >= len(desired_tgts[key]):
                        done_ok.add(key)
        return done_ok

    def run_one_pass(all_scenes, desired_tgts, done_ok):
        scoped = all_scenes[:args.limit] if args.limit > 0 else all_scenes
        if args.limit > 0:
            print(f"limited by --limit={args.limit} to {len(scoped)} scenes")
        todo = [s for s in scoped if (s[0], s[1]) not in done_ok]
        print(f"todo {len(todo)} scenes / {sum(len(s[3]) for s in todo)} targets")
        if not todo:
            return 0
        results_by_cat = defaultdict(list)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_scene = {
                executor.submit(
                    call_vlm_scene, cat, anchor, refs, tgts,
                    8, args.chunk_tgts,
                ): (cat, anchor, refs, tgts)
                for cat, anchor, refs, tgts in todo
            }
            for future in as_completed(future_to_scene):
                cat, anchor, refs, tgts = future_to_scene[future]
                result = future.result()
                results_by_cat[cat].append({
                    "cat": cat, "anchor": anchor, "refs": refs, "tgts": tgts, "vlm": result
                })
                if "error" in result:
                    print(f"[{cat}:{anchor}] ERR {result['error'][:100]}", flush=True)
                else:
                    print(f"[{cat}:{anchor}] {len(result.get('targets', []))} prompts", flush=True)
        for cat, data in results_by_cat.items():
            out_path = OUT_DIR / f"prompts_{cat}.json"
            by_anchor = {}
            if out_path.exists():
                for item in json.loads(out_path.read_text()):
                    by_anchor[item["anchor"]] = item
            for item in data:
                by_anchor[item["anchor"]] = item
            out_path.write_text(json.dumps(list(by_anchor.values()), ensure_ascii=False, indent=2))
            print(f"saved {cat} results to {out_path} ({len(by_anchor)} entries)")
        return len(todo)

    def refresh_html():
        full_results = defaultdict(list)
        for f in sorted(OUT_DIR.glob("prompts_*.json")):
            for item in json.loads(f.read_text()):
                full_results[item["cat"]].append(item)
        (OUT_DIR / "index.html").write_text(build_html(full_results))
        desired_n = sum(len(v) for v in desired_tgts.values())
        n_ok_sc = 0
        n_prompts = 0
        for item in (x for v in full_results.values() for x in v):
            key = (item["cat"], item["anchor"])
            if key not in desired_tgts or not scene_ok(item):
                continue
            if set(item.get("tgts") or []) == set(desired_tgts[key]) and \
               len((item.get("vlm") or {}).get("targets") or []) >= len(desired_tgts[key]):
                n_ok_sc += 1
                n_prompts += len(item["vlm"]["targets"])
        print(f"HTML updated. complete scenes {n_ok_sc}/{len(desired_tgts)}, full prompts {n_prompts}/{desired_n}")
        root_index = HOME / "mpie_bench/data/index.html"
        root_index.write_text(
            '<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=/manifests/prompts_full/index.html"></head><body></body></html>'
        )
        return n_ok_sc, n_prompts, desired_n

    all_scenes = collect_all_scenes()
    desired_tgts = {(cat, anchor): list(tgts) for cat, anchor, refs, tgts in all_scenes}
    # until-complete implies resume
    if args.until_complete:
        args.resume = True

    round_i = 0
    prev_done = -1
    stall = 0
    while True:
        round_i += 1
        done_ok = load_done_ok(desired_tgts)
        print(f"=== round {round_i}: complete {len(done_ok)}/{len(desired_tgts)} ===", flush=True)
        if len(done_ok) == len(desired_tgts):
            print("all scenes complete; exiting")
            break
        if len(done_ok) == prev_done:
            stall += 1
            if stall >= 2 and args.until_complete:
                print("no progress for two rounds; pause (retry later with --until-complete)")
                break
            if not args.until_complete:
                break
        else:
            stall = 0
        prev_done = len(done_ok)
        n_todo = run_one_pass(all_scenes, desired_tgts, done_ok)
        if n_todo == 0:
            break
        if not args.until_complete:
            break
        time.sleep(15)

    refresh_html()
    print(f"results saved under {OUT_DIR}")

if __name__ == "__main__":
    main()