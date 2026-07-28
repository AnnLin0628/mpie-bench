#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPIE VLM Judge v1 — gateway API Criticize referees in parallel (can resume running from breakpoint).

protocol: docs/eval_protocol_v3.md (Count axis)
layout: $PACK/outputs/<model>/ + manifest → $PACK/judgments/vlm_judge_v1/<model>/

Design points (the terminal will not lose progress even if it is interrupted):
  - Each sample is written independently judgments/.../<sample_id>.json(First .tmp Again os.replace）
  - Skip existing and passed on startup schema Verified JSON（--resume On by default)
  - parallel ThreadPoolExecutor; When killing a process, only the "in progress" items will be lost. Completed processes will not be affected.
  - recommend nohup / systemd Run; if you run the same command again, it will continue.

usage:
  # smoke100 comment gemini-3-pro-image(Original nano-banana-pro），gpt-5.5 trial,8 concurrent
  python score_vlm_v1.py \\
    --pack "$MPIE_TEST_PACK" \\
    --model-id gemini-3-pro-image \\
    --judge-model gpt-5.5 \\
    --workers 8

  # Backstage
  bash run_score_vlm_v1.sh gemini-3-pro-image
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from pack_io import load_manifest, pack_root  # noqa: E402

# Reuse closed source gateway key / URL(Environment variables are used, no hard-coded intranet address)
try:
    from closedsource.gateway import gateway_base, gateway_key  # noqa: E402
except Exception:  # pragma: no cover
    gateway_base = None  # type: ignore
    gateway_key = None  # type: ignore

DEFAULT_MODEL = "gpt-5.5"
MAX_SIDE = 1024
ANAT_VOCAB = {
    "extra_limb",
    "missing_limb",
    "merged_body",
    "limb_ownership_error",
    "bad_hand",
    "impossible_joint",
    "floating_part",
    "face_melt",
}
YNPU = {"yes", "no", "partial", "uncertain"}
REQUIRED_TOP = {
    "count_detected",
    "count_expected",
    "count_pass",
    "id",
    "anat_errors",
    "anat_pass",
    "inter",
    "instr_qa",
    "qual_pass",
    "confidence",
}

_print_lock = threading.Lock()
_err_lock = threading.Lock()

JUDGE_PROMPT = """You are a strict evaluator for MULTI-PERSON INTERACTION image generation/editing (MPIE-Bench).

Image 1 is the GENERATED image to score.
The remaining images are character REFERENCE portraits in order: {ref_names}.

Generation / edit instruction:
\"\"\"{instruction}\"\"\"

Expected main characters (references): {n_person}
Declared bystanders (if any, from the instruction): {n_bystander}
Interaction type (hint): {interaction_type}
Contact density hint: {contact_density}
Expected contact points hint: {contact_points}

Evaluate with CLOSED CHECKLISTS — no 1-5 Likert scores.
Deep legitimate contact (wrestling / lifts / C3) is NOT pathological interpenetration.
Qual must IGNORE anatomy failures (do not double-count).

Return ONLY one JSON object (no markdown fence, no commentary):
{{
  "count_detected": <int, clearly visible people including declared bystanders>,
  "count_expected": <int, main refs + declared bystanders>,
  "count_pass": <bool, detected == expected>,
  "id": [
    {{"ref": "R1", "match": "yes|no|uncertain", "evidence": "<one short clause>"}},
    ...
  ],
  "anat_errors": [
    {{"type": "extra_limb|missing_limb|merged_body|limb_ownership_error|bad_hand|impossible_joint|floating_part|face_melt",
      "where": "<e.g. R1 right arm / contact region>"}}
  ],
  "anat_pass": <bool, true iff anat_errors is empty>,
  "inter": {{
    "semantic": "yes|partial|no",
    "contact_points": "yes|partial|no",
    "no_pathological_penetration": "yes|partial|no"
  }},
  "instr_qa": [
    {{"q": "<atomic yes/no question about action / who-does-what / scene>", "a": "yes|partial|no"}},
    {{"q": "...", "a": "yes|partial|no"}}
  ],
  "qual_pass": <bool, sharpness/composition OK ignoring anatomy>,
  "confidence": "high|medium|low",
  "overall_notes": "<one sentence>"
}}

Rules:
- id[] must cover every reference R1..RN in order.
- Use match=uncertain only for heavy occlusion / back view / extreme profile.
- anat_errors types MUST be from the closed vocabulary; empty list if none.
- instr_qa: 2-4 atomic questions derived from the instruction.
- Focus on contact-region limb ownership, fingers, fused bodies.
"""


def _api_url() -> str:
    env = os.environ.get("MPIE_VLM_URL") or os.environ.get("AI_GATEWAY_URL")
    if env:
        env = env.rstrip("/")
        if env.endswith("/chat/completions"):
            return env
        return env + "/chat/completions"
    if gateway_base is not None:
        return gateway_base().rstrip("/") + "/chat/completions"
    raise RuntimeError(
        "Missing gateway URL：export MPIE_VLM_URL=... or AI_GATEWAY_URL=https://<host>/v1"
    )


def _api_key() -> str:
    key = (
        os.environ.get("MPIE_VLM_KEY")
        or os.environ.get("AI_GATEWAY_KEY")
        or os.environ.get("AI_GATEWAY_KEY")
    )
    if key:
        return key
    if gateway_key is not None:
        return gateway_key()
    raise RuntimeError("Missing gateway Key：export AI_GATEWAY_KEY=... or MPIE_VLM_KEY=...")


def judgment_dir(root: Path, model_id: str) -> Path:
    d = root / "judgments" / "vlm_judge_v1" / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def judgment_path(root: Path, model_id: str, sample_id: str) -> Path:
    return judgment_dir(root, model_id) / f"{sample_id}.json"


def find_gen_image(root: Path, model_id: str, sample_id: str) -> Path | None:
    d = root / "outputs" / model_id
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = d / f"{sample_id}{ext}"
        if p.is_file() and p.stat().st_size > 1000:
            return p
    return None


def img_b64_jpeg(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getbuffer()).decode("ascii")


def parse_json_obj(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError(f"no JSON object in model output: {text[:240]}")
    return json.loads(m.group(0))


def schema_ok(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    if not REQUIRED_TOP.issubset(obj.keys()):
        return False
    if not isinstance(obj.get("id"), list) or not obj["id"]:
        return False
    if not isinstance(obj.get("anat_errors"), list):
        return False
    inter = obj.get("inter")
    if not isinstance(inter, dict):
        return False
    for k in ("semantic", "contact_points", "no_pathological_penetration"):
        if inter.get(k) not in ("yes", "partial", "no"):
            return False
    if obj.get("confidence") not in ("high", "medium", "low"):
        return False
    return True


def already_judged(path: Path, require_judge: str | None = None) -> bool:
    """legitimate JSON It's complete. like require_judge Given, it is also required _meta.judge_model consistent
    (avoid Agent Old results block gateway gpt-5.5 Re-evaluation; continuous running with the same model will still skip). """
    if not path.is_file() or path.stat().st_size < 20:
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    body = obj.get("judgment") if isinstance(obj.get("judgment"), dict) else obj
    if not schema_ok({k: v for k, v in body.items() if k != "_meta"}):
        return False
    if require_judge:
        meta = obj.get("_meta") if isinstance(obj.get("_meta"), dict) else {}
        if not meta and isinstance(body.get("_meta"), dict):
            meta = body["_meta"]
        return (meta or {}).get("judge_model") == require_judge
    return True


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, ensure_ascii=False, indent=2)
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


def append_error(root: Path, model_id: str, sample_id: str, err: str) -> None:
    p = judgment_dir(root, model_id) / "_errors.jsonl"
    line = json.dumps(
        {
            "sample_id": sample_id,
            "error": err[:500],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        ensure_ascii=False,
    )
    with _err_lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def infer_n_person(row: dict) -> int:
    if row.get("n_person"):
        return int(row["n_person"])
    return max(1, len(row.get("ref_relpaths") or []))


def infer_n_bystander(row: dict) -> int:
    if row.get("n_bystander") is not None:
        return int(row["n_bystander"])
    # Coarse heuristic:prompt inside bystander/passersby
    p = (row.get("prompt") or "").lower()
    if "bystander" in p or "passersby" in p or "passerby" in p:
        return 1
    return 0


def call_judge(
    *,
    gen: Path,
    refs: list[Path],
    instruction: str,
    n_person: int,
    n_bystander: int,
    interaction_type: str,
    contact_density: str,
    contact_points: str,
    judge_model: str,
    timeout: int,
    retries: int,
) -> dict:
    content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64_jpeg(gen)}"},
        }
    ]
    ref_names = []
    for i, rp in enumerate(refs):
        ref_names.append(f"R{i+1}")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64_jpeg(rp)}"},
            }
        )
    prompt = JUDGE_PROMPT.format(
        ref_names=", ".join(ref_names) if ref_names else "(none)",
        instruction=instruction,
        n_person=n_person,
        n_bystander=n_bystander,
        interaction_type=interaction_type or "unknown",
        contact_density=contact_density or "unknown",
        contact_points=contact_points or "unknown",
    )
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 4096,
        "temperature": 0.1,
    }
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    url = _api_url()
    last = ""
    for attempt in range(retries + 1):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if r.status_code in (429, 502, 503):
                last = f"HTTP {r.status_code}: {r.text[:200]}"
                time.sleep(min(3 * 2**attempt, 60))
                continue
            r.raise_for_status()
            body = r.json()
            if body.get("error"):
                raise RuntimeError(str(body["error"])[:400])
            msg = (body.get("choices") or [{}])[0].get("message") or {}
            text = msg.get("content") or ""
            obj = parse_json_obj(text)
            if not schema_ok(obj):
                raise ValueError(f"schema validation failed: keys={list(obj.keys())[:20]}")
            # Light cleaning anat out-of-vocabulary types
            cleaned = []
            for e in obj.get("anat_errors") or []:
                if isinstance(e, dict) and e.get("type") in ANAT_VOCAB:
                    cleaned.append(e)
            obj["anat_errors"] = cleaned
            obj["anat_pass"] = len(cleaned) == 0
            return obj
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"[:300]
            if attempt < retries:
                time.sleep(2**attempt + 1)
    raise RuntimeError(last or "judge failed")


def score_one(row: dict, *, root: Path, model_id: str, args) -> str:
    sid = row["sample_id"]
    out_p = judgment_path(root, model_id, sid)
    require = args.judge_model if args.resume else None
    # --resume Press "Same Referee Model" to skip;--no-resume Force full rewrite
    if args.resume and already_judged(out_p, require_judge=require):
        return "skip"

    gen = find_gen_image(root, model_id, sid)
    if gen is None:
        append_error(root, model_id, sid, "missing_gen")
        return "missing_gen"

    refs: list[Path] = []
    for rel in row.get("ref_relpaths") or []:
        p = root / rel
        if not p.exists():
            append_error(root, model_id, sid, f"missing_ref:{rel}")
            return "missing_ref"
        refs.append(p)

    n_person = infer_n_person(row)
    n_bystander = infer_n_bystander(row)
    t0 = time.time()
    try:
        judgment = call_judge(
            gen=gen,
            refs=refs,
            instruction=row.get("prompt") or "",
            n_person=n_person,
            n_bystander=n_bystander,
            interaction_type=row.get("interaction_type") or row.get("cat") or "",
            contact_density=str(row.get("contact_density") or ""),
            contact_points=str(row.get("contact_points") or ""),
            judge_model=args.judge_model,
            timeout=args.timeout,
            retries=args.retries,
        )
    except Exception as e:  # noqa: BLE001
        append_error(root, model_id, sid, str(e))
        return "fail"

    payload = {
        **judgment,
        "_meta": {
            "sample_id": sid,
            "model_id": model_id,
            "judge_model": args.judge_model,
            "judge_endpoint": _api_url(),
            "seconds": round(time.time() - t0, 2),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "n_refs": len(refs),
            "gen_relpath": str(gen.relative_to(root)),
        },
    }
    # schema_ok neglect _meta;Confirm that the subject is still legal before writing
    body_check = {k: v for k, v in payload.items() if k != "_meta"}
    if not schema_ok(body_check):
        append_error(root, model_id, sid, "postcheck_schema_fail")
        return "fail"
    atomic_write_json(out_p, payload)
    return "ok"


def iter_judge_todo(
    root: Path, model_id: str, limit: int, resume: bool, judge_model: str
):
    n = 0
    require = judge_model if resume else None
    for row in load_manifest(root):
        sid = row["sample_id"]
        if find_gen_image(root, model_id, sid) is None:
            continue
        if resume and already_judged(
            judgment_path(root, model_id, sid), require_judge=require
        ):
            continue
        yield row
        n += 1
        if limit and n >= limit:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="MPIE VLM Judge v1 API batch scorer")
    ap.add_argument("--pack", default="", help="pack root directory")
    ap.add_argument(
        "--model-id",
        required=True,
        help="The directory name of the evaluated generated model, such as gemini-3-pro-image / gpt-image-2",
    )
    ap.add_argument(
        "--judge-model",
        default=os.environ.get("MPIE_JUDGE_MODEL", DEFAULT_MODEL),
        help=f"Referee model (default {DEFAULT_MODEL},Walk AI_GATEWAY_URL / MPIE_VLM_URL）",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="It is legal to skip JSON(On by default;--no-resume Forced re-evaluation)",
    )
    args = ap.parse_args()

    # alias
    model_id = args.model_id
    if model_id == "nano-banana-pro":
        model_id = "gemini-3-pro-image"

    root = pack_root(args.pack or None)
    out_dir = judgment_dir(root, model_id)
    todo = list(
        iter_judge_todo(root, model_id, args.limit, args.resume, args.judge_model)
    )

    print(
        json.dumps(
            {
                "pack": str(root),
                "model_id": model_id,
                "judge_model": args.judge_model,
                "endpoint": _api_url(),
                "workers": args.workers,
                "todo": len(todo),
                "out_dir": str(out_dir),
                "resume": args.resume,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not todo:
        print("nothing to do", flush=True)
        return

    counts = {"ok": 0, "skip": 0, "fail": 0, "missing_gen": 0, "missing_ref": 0}
    t0 = time.time()

    def _job(row: dict) -> tuple[str, str]:
        sid = row["sample_id"]
        status = score_one(row, root=root, model_id=model_id, args=args)
        return sid, status

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_job, row): row["sample_id"] for row in todo}
        done_n = 0
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                _, status = fut.result()
            except Exception as e:  # noqa: BLE001
                status = "fail"
                append_error(root, model_id, sid, f"worker_crash:{e}")
            counts[status] = counts.get(status, 0) + 1
            done_n += 1
            with _print_lock:
                print(
                    f"[{done_n}/{len(todo)}] {sid} -> {status}",
                    flush=True,
                )

    summary = {
        "pack": str(root),
        "model_id": model_id,
        "judge_model": args.judge_model,
        "counts": counts,
        "elapsed_sec": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
