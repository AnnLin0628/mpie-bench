#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage B: answer frozen Instr QA v2 on a generated image only.

Does NOT invent questions. Does NOT use reference portraits for Instr
(identity is ArcFace's job). Questions come from $PACK/instr_qa_v2/.

Output: $PACK/judgments/instr_v2/<model_id>/<sample_id>.json

Usage:
  python score_instr_v2.py --pack "$MPIE_TEST_PACK" --model-id gpt-image-2 --limit 5
  bash run_score_instr_v2.sh gpt-image-2 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from instr_qa_common import (  # noqa: E402
    DEFAULT_MODEL,
    PROTOCOL,
    REVISION,
    api_url,
    atomic_write_json,
    chat_json,
    find_gen_image,
    frozen_qa_ok,
    img_b64_jpeg,
    judgment_ok,
    judgment_path,
    normalize_answer,
    qa_path,
    score_answers,
)
from pack_io import load_manifest, pack_root  # noqa: E402

_print_lock = threading.Lock()
_err_lock = threading.Lock()

ANSWER_PROMPT = """You are answering a FROZEN instruction-following VQA checklist for MPIE-Bench.
Image 1 is the GENERATED image.
You must answer EACH question below with exactly one of: yes | partial | no.
Do NOT rewrite, skip, or add questions.
Do NOT judge face identity similarity to reference photos (not provided).
Do NOT invent anatomy scores.
If the asked detail is not visible, answer no (not partial), unless it is clearly partially true.

Questions:
{questions_block}

Return ONLY JSON:
{{
  "answers": [
    {{"id": "q1", "a": "yes|partial|no", "evidence": "one short clause"}}
  ],
  "confidence": "high|medium|low"
}}
"""


def append_error(root: Path, model_id: str, sample_id: str, err: str) -> None:
    p = root / "judgments" / "instr_v2" / model_id / "_errors.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
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


def load_frozen(root: Path, sample_id: str) -> dict | None:
    p = qa_path(root, sample_id)
    if not p.is_file():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not frozen_qa_ok(obj):
        return None
    meta = obj.get("_meta") if isinstance(obj.get("_meta"), dict) else {}
    rev = obj.get("revision") or meta.get("revision")
    if rev != REVISION:
        return None
    return obj


def already_scored(path: Path, require_judge: str | None) -> bool:
    if not path.is_file() or path.stat().st_size < 40:
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not judgment_ok(obj, require_judge=require_judge):
        return False
    meta = obj.get("_meta") if isinstance(obj.get("_meta"), dict) else {}
    rev = obj.get("revision") or meta.get("revision")
    return rev == REVISION


def answer_image(
    *,
    gen: Path,
    questions: list[dict],
    judge_model: str,
    timeout: int,
    retries: int,
) -> dict:
    block = "\n".join(f'- {q["id"]} [{q["bucket"]}]: {q["q"]}' for q in questions)
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64_jpeg(gen)}"},
        },
        {"type": "text", "text": ANSWER_PROMPT.format(questions_block=block)},
    ]
    obj = chat_json(
        messages=[{"role": "user", "content": content}],
        model=judge_model,
        timeout=timeout,
        retries=retries,
        temperature=0.0,
        max_tokens=2000,
    )
    if not isinstance(obj, dict):
        raise ValueError("answer model did not return JSON object")
    ans_list = obj.get("answers")
    if not isinstance(ans_list, list) or not ans_list:
        raise ValueError("missing answers[]")
    by_id: dict[str, str] = {}
    evidence: dict[str, str] = {}
    for a in ans_list:
        if not isinstance(a, dict):
            continue
        qid = str(a.get("id") or "")
        na = normalize_answer(a.get("a"))
        if qid and na:
            by_id[qid] = na
            evidence[qid] = str(a.get("evidence") or "")[:160]
    conf = str(obj.get("confidence") or "medium").lower()
    if conf not in ("high", "medium", "low"):
        conf = "medium"
    return {"answers": by_id, "evidence": evidence, "confidence": conf}


def score_one(row: dict, *, root: Path, model_id: str, args) -> str:
    sid = row["sample_id"]
    out_p = judgment_path(root, model_id, sid)
    require = args.judge_model if args.resume else None
    if args.resume and already_scored(out_p, require_judge=require):
        return "skip"

    frozen = load_frozen(root, sid)
    if frozen is None:
        append_error(root, model_id, sid, "missing_frozen_qa")
        return "missing_qa"

    gen = find_gen_image(root, model_id, sid)
    if gen is None:
        append_error(root, model_id, sid, "missing_gen")
        return "missing_gen"

    questions = frozen["questions"]
    t0 = time.time()
    try:
        ans = answer_image(
            gen=gen,
            questions=questions,
            judge_model=args.judge_model,
            timeout=args.timeout,
            retries=args.retries,
        )
    except Exception as e:  # noqa: BLE001
        append_error(root, model_id, sid, str(e))
        return "fail"

    scored = score_answers(questions, ans["answers"])
    # attach evidence
    for d in scored["detail"]:
        d["evidence"] = ans["evidence"].get(d["id"], "")

    payload = {
        "ok": True,
        "protocol": PROTOCOL,
        "revision": REVISION,
        "sample_id": sid,
        "model_id": model_id,
        "S_instr": scored["S_instr"],
        "S_instr_role": scored["S_instr_role"],
        "S_instr_asymm": scored["S_instr_asymm"],
        "S_instr_prop": scored["S_instr_prop"],
        "S_instr_scene": scored["S_instr_scene"],
        "S_instr_role_duty": scored.get("S_instr_role_duty"),
        "S_instr_prop_object": scored.get("S_instr_prop_object"),
        "n_main": scored["n_main"],
        "n_all": scored["n_all"],
        "confidence": ans["confidence"],
        "detail": scored["detail"],
        "weights": scored.get("weights"),
        "_meta": {
            "protocol": PROTOCOL,
            "revision": REVISION,
            "sample_id": sid,
            "model_id": model_id,
            "judge_model": args.judge_model,
            "qg_model": (frozen.get("_meta") or {}).get("qg_model"),
            "endpoint": api_url(),
            "seconds": round(time.time() - t0, 2),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gen_relpath": str(gen.relative_to(root)),
            "qa_relpath": str(qa_path(root, sid).relative_to(root)),
            "n_questions": len(questions),
        },
    }
    atomic_write_json(out_p, payload)
    return "ok"


def iter_todo(root: Path, model_id: str, limit: int, resume: bool, judge_model: str):
    n = 0
    require = judge_model if resume else None
    for row in load_manifest(root):
        sid = row["sample_id"]
        if find_gen_image(root, model_id, sid) is None:
            continue
        if not qa_path(root, sid).is_file():
            continue
        if resume and already_scored(
            judgment_path(root, model_id, sid), require_judge=require
        ):
            continue
        yield row
        n += 1
        if limit and n >= limit:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Score Instr QA v2 (frozen questions)")
    ap.add_argument("--pack", default="")
    ap.add_argument("--model-id", required=True)
    ap.add_argument(
        "--judge-model",
        default=os.environ.get("MPIE_JUDGE_MODEL", DEFAULT_MODEL),
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = ap.parse_args()

    model_id = args.model_id
    if model_id == "nano-banana-pro":
        model_id = "gemini-3-pro-image"

    root = pack_root(args.pack or None)
    out_dir = root / "judgments" / "instr_v2" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = list(
        iter_todo(root, model_id, args.limit, args.resume, args.judge_model)
    )

    print(
        json.dumps(
            {
                "pack": str(root),
                "model_id": model_id,
                "judge_model": args.judge_model,
                "endpoint": api_url(),
                "workers": args.workers,
                "todo": len(todo),
                "out_dir": str(out_dir),
                "note": "requires instr_qa_v2/ frozen bank",
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not todo:
        print("nothing to do (need gen images + frozen QA)", flush=True)
        return

    counts: dict[str, int] = {
        "ok": 0,
        "skip": 0,
        "fail": 0,
        "missing_gen": 0,
        "missing_qa": 0,
    }
    t0 = time.time()

    def _job(row: dict) -> tuple[str, str]:
        return row["sample_id"], score_one(row, root=root, model_id=model_id, args=args)

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
                print(f"[{done_n}/{len(todo)}] {sid} -> {status}", flush=True)

    summary = {
        "pack": str(root),
        "model_id": model_id,
        "judge_model": args.judge_model,
        "counts": counts,
        "elapsed_sec": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": PROTOCOL,
    }
    (out_dir / "_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
