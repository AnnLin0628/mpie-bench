#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage A: text-only Instr QA bank (TIFA / MultiHuman style).

NO generated image. NO reference portraits.
Extract atomic claims from the edit instruction → freeze yes/no questions
with gold_a=yes, then text-filter bad questions.

Output: $PACK/instr_qa_v2/<sample_id>.json

Usage:
  python build_instr_qa_v2.py --pack "$MPIE_TEST_PACK" --limit 5
  bash run_build_instr_qa_v2.sh 8
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
    ALL_BUCKETS,
    DEFAULT_MODEL,
    PROTOCOL,
    REVISION,
    api_url,
    atomic_write_json,
    chat_json,
    clean_questions,
    enforce_quota,
    frozen_qa_ok,
    infer_n_bystander,
    infer_n_person,
    merge_with_fallback,
    qa_path,
    question_looks_forbidden,
    ref_labels,
    synthesize_fallback_questions,
)
from pack_io import load_manifest, pack_root  # noqa: E402

_print_lock = threading.Lock()
_err_lock = threading.Lock()

QG_SYSTEM = """You build a frozen atomic VQA bank for MPIE-Bench.
MPIE-Bench evaluates MULTI-PERSON INTERACTION EDITING: named people (R1,R2,…) must keep
asymmetric contact roles (who supports / who is carried / who holds the mitt / who leads).

You see ONLY the edit instruction text — never an image.
Follow TIFA practice: extract entailed visual claims, then write yes/no questions with gold_a=yes.

PRIORITY buckets (this is what separates strong vs weak editors on MPIE):
1) asymm (REQUIRED, write 2–3): asymmetric duty that FAILS if R1↔R2 are swapped.
   Good: "Is R1 supporting R2's weight from below, rather than R2 supporting R1?"
   Good: "Does R1 hold the focus mitt that absorbs R2's punch?"
   Bad:  "Are they hugging?" / "Is R1 on the left?"
2) role (write 1–2): contact role / body configuration / who-does-what with R# labels.
   Good: "Is R2 on R1's back while R1 supports her?"
   Bad:  pure left/right standing ("Is R1 on the left?") — FORBIDDEN
3) prop (write 1–2): contact-relevant OBJECTS only (mitt, bag, phone, boxes, bouquet…).
   Bad:  clothing color alone ("Is R1 wearing a white dress?") — FORBIDDEN
4) scene: at most ONE concrete forced setting detail, optional.

HARD FORBIDDEN:
- left/right-only layout questions
- clothing-color-only questions
- "are they hugging/shaking hands?" (Inter axis)
- person count (Count) / face identity (ID) / anatomy-fusion (Anat)

Rules:
- Every asymm/role question MUST use R1/R2/… labels.
- set swap_sensitive=true for every asymm and role-duty question.
- 5–7 questions total; when n_person>=2 you MUST include >=2 asymm.
- Near-symmetric contacts (high-five / handshake): still write asymm about
  (a) who actively raises/initiates, (b) who is onlooker vs contact pair when n>=3,
  (c) reciprocal raise — NEVER skip asymm just because the contact looks mutual.
- Answerable as YES from the TEXT alone.
- Return ONLY JSON.
"""

QG_USER = """Edit instruction:
\"\"\"{instruction}\"\"\"

n_person (main refs): {n_person}
ref labels: {ref_names}
n_bystander (declared): {n_bystander}
interaction_type hint: {interaction_type}
contact_density hint: {contact_density}

Return JSON:
{{
  "elements": ["asymmetric duty 1", "contact role 2", "contact object 3", "..."],
  "questions": [
    {{
      "id": "q1",
      "bucket": "asymm",
      "q": "Is R1 … rather than R2 …?",
      "gold_a": "yes",
      "swap_sensitive": true,
      "element": "asymmetric support/contact duty"
    }}
  ]
}}

Checklist before you answer: >=2 asymm, >=1 role (duty not left/right), >=1 object prop if any object is mentioned; zero clothing-only; zero left/right-only.
"""

FILTER_USER = """You are a text-only QA filter for MPIE-Bench hard questions.
Given ONLY the instruction, answer each question with yes/no/partial/unanswerable.

Keep (answer yes/partial) only if the instruction entails the claim AND the question tests
asymmetric contact role, who-does-what, or a contact-relevant object.

Mark unanswerable if:
- left/right layout only, or clothing color only
- face identity / person count / generic "are they hugging?"
- anatomy / fusion
- claim not entailed by the instruction

Instruction:
\"\"\"{instruction}\"\"\"

Questions:
{questions_block}

Return JSON:
{{
  "answers": [
    {{"id": "q1", "a": "yes|no|partial|unanswerable", "reason": "short"}}
  ]
}}
"""


def append_error(root: Path, sample_id: str, err: str) -> None:
    p = root / "instr_qa_v2" / "_errors.jsonl"
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


def already_built(path: Path, require_model: str | None) -> bool:
    if not path.is_file() or path.stat().st_size < 40:
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not frozen_qa_ok(obj):
        return False
    meta = obj.get("_meta") or {}
    if meta.get("revision") != REVISION:
        return False
    if require_model and meta.get("qg_model") != require_model:
        return False
    return True


def generate_raw(
    *,
    instruction: str,
    n_person: int,
    n_bystander: int,
    interaction_type: str,
    contact_density: str,
    model: str,
    timeout: int,
    retries: int,
) -> dict:
    user = QG_USER.format(
        instruction=instruction,
        n_person=n_person,
        ref_names=", ".join(ref_labels(n_person)),
        n_bystander=n_bystander,
        interaction_type=interaction_type or "unknown",
        contact_density=contact_density or "unknown",
    )
    obj = chat_json(
        messages=[
            {"role": "system", "content": QG_SYSTEM},
            {"role": "user", "content": user},
        ],
        model=model,
        timeout=timeout,
        retries=retries,
        temperature=0.2,
        max_tokens=3000,
    )
    if not isinstance(obj, dict):
        raise ValueError("QG did not return a JSON object")
    return obj


def text_filter(
    *,
    instruction: str,
    questions: list[dict],
    model: str,
    timeout: int,
    retries: int,
) -> list[dict]:
    if not questions:
        return []
    block = "\n".join(f'- {q["id"]}: {q["q"]}' for q in questions)
    obj = chat_json(
        messages=[{"role": "user", "content": FILTER_USER.format(
            instruction=instruction, questions_block=block
        )}],
        model=model,
        timeout=timeout,
        retries=retries,
        temperature=0.0,
        max_tokens=2000,
    )
    ans_list = obj.get("answers") if isinstance(obj, dict) else None
    by_id = {}
    if isinstance(ans_list, list):
        for a in ans_list:
            if isinstance(a, dict) and a.get("id"):
                by_id[str(a["id"])] = str(a.get("a") or "").strip().lower()
    kept = []
    for q in questions:
        a = by_id.get(q["id"], "yes")  # if filter fails open, keep heuristic-clean ones
        if a in ("yes", "partial"):
            # partial from text is weak but keep if not forbidden
            if not question_looks_forbidden(q["q"]):
                kept.append(q)
        # no / unanswerable → drop
    return kept


def build_one(row: dict, *, root: Path, args) -> str:
    sid = row["sample_id"]
    out_p = qa_path(root, sid)
    require = args.qg_model if args.resume else None
    if args.resume and already_built(out_p, require_model=require):
        return "skip"

    instruction = (row.get("prompt") or "").strip()
    if len(instruction) < 20:
        append_error(root, sid, "empty_prompt")
        return "fail"

    n_person = infer_n_person(row)
    n_bystander = infer_n_bystander(row)
    t0 = time.time()
    raw_used: dict = {}
    try:

        def _once() -> tuple[dict, list[dict], list[str]]:
            raw = generate_raw(
                instruction=instruction,
                n_person=n_person,
                n_bystander=n_bystander,
                interaction_type=row.get("interaction_type") or row.get("cat") or "",
                contact_density=str(row.get("contact_density") or ""),
                model=args.qg_model,
                timeout=args.timeout,
                retries=args.retries,
            )
            cleaned = clean_questions(raw.get("questions") or [], n_person=n_person)
            if args.filter:
                cleaned = text_filter(
                    instruction=instruction,
                    questions=cleaned,
                    model=args.filter_model or args.qg_model,
                    timeout=args.timeout,
                    retries=args.retries,
                )
            kept_i, warnings_i = enforce_quota(cleaned, n_person=n_person)
            return raw, kept_i, warnings_i

        raw_used, kept, warnings = _once()
        need_retry = (
            "asymm_missing" in warnings
            or "role_duty_missing" in warnings
            or "main_lt_3" in warnings
            or len([q for q in kept if q.get("main")]) < 3
        )
        if need_retry:
            raw2, kept2, warnings2 = _once()

            def _hard_score(qs: list[dict], ws: list[str]) -> int:
                return (
                    sum(1 for q in qs if q.get("subtype") == "asymm") * 10
                    + sum(1 for q in qs if q.get("subtype") == "role_duty") * 3
                    + sum(1 for q in qs if q.get("main"))
                    - (5 if "asymm_missing" in ws else 0)
                )

            if _hard_score(kept2, warnings2) > _hard_score(kept, warnings):
                raw_used, kept, warnings = raw2, kept2, warnings2

        # Near-symmetric / thin banks: pad with deterministic hard templates
        draft = {
            "questions": kept,
            "n_person": n_person,
        }
        if not frozen_qa_ok(draft) or "asymm_missing" in warnings or "main_lt_3" in warnings:
            fb = synthesize_fallback_questions(
                instruction=instruction,
                n_person=n_person,
                cat=str(row.get("cat") or row.get("interaction_type") or ""),
            )
            kept, warnings = merge_with_fallback(kept, fb, n_person=n_person)
            warnings = list(dict.fromkeys(warnings + ["used_fallback"]))
    except Exception as e:  # noqa: BLE001
        append_error(root, sid, str(e))
        return "fail"

    payload = {
        "protocol": PROTOCOL,
        "revision": REVISION,
        "sample_id": sid,
        "instruction": instruction,
        "n_person": n_person,
        "n_bystander": n_bystander,
        "ref_labels": ref_labels(n_person),
        "elements": (
            raw_used.get("elements")
            if isinstance(raw_used.get("elements"), list)
            else []
        ),
        "questions": kept,
        "bucket_counts": {
            b: sum(1 for q in kept if q["bucket"] == b) for b in ALL_BUCKETS
        },
        "warnings": warnings,
        "_meta": {
            "revision": REVISION,
            "qg_model": args.qg_model,
            "filter": bool(args.filter),
            "filter_model": (args.filter_model or args.qg_model) if args.filter else None,
            "endpoint": api_url(),
            "seconds": round(time.time() - t0, 2),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cat": row.get("cat"),
            "used_fallback": "used_fallback" in warnings,
        },
    }
    if not frozen_qa_ok(payload):
        append_error(root, sid, f"quota_fail warnings={warnings}")
        # still write for debugging under .bad suffix? skip write → fail
        atomic_write_json(
            out_p.with_suffix(".json.bad"),
            payload,
        )
        return "fail"

    atomic_write_json(out_p, payload)
    return "ok"


def iter_todo(root: Path, limit: int, resume: bool, qg_model: str):
    n = 0
    require = qg_model if resume else None
    for row in load_manifest(root):
        sid = row["sample_id"]
        if resume and already_built(qa_path(root, sid), require_model=require):
            continue
        yield row
        n += 1
        if limit and n >= limit:
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Build frozen Instr QA v2 (text-only)")
    ap.add_argument("--pack", default="")
    ap.add_argument(
        "--qg-model",
        default=os.environ.get("MPIE_JUDGE_MODEL", DEFAULT_MODEL),
    )
    ap.add_argument("--filter-model", default="", help="defaults to --qg-model")
    ap.add_argument(
        "--filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="text-only filter pass (default on)",
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

    root = pack_root(args.pack or None)
    out_dir = root / "instr_qa_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        endpoint = api_url()
    except RuntimeError as e:
        print(f"ERROR: {e}", flush=True)
        print(
            "Hint: export AI_GATEWAY_URL=https://<host>/v1 and AI_GATEWAY_KEY=... "
            "(same as score_vlm_v1). Optional: put them in ~/.mpie_env",
            flush=True,
        )
        raise SystemExit(2)
    todo = list(iter_todo(root, args.limit, args.resume, args.qg_model))

    print(
        json.dumps(
            {
                "pack": str(root),
                "qg_model": args.qg_model,
                "filter": args.filter,
                "endpoint": endpoint,
                "workers": args.workers,
                "todo": len(todo),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not todo:
        print("nothing to do", flush=True)
        return

    counts: dict[str, int] = {"ok": 0, "skip": 0, "fail": 0}
    t0 = time.time()

    def _job(row: dict) -> tuple[str, str]:
        return row["sample_id"], build_one(row, root=root, args=args)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_job, row): row["sample_id"] for row in todo}
        done_n = 0
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                _, status = fut.result()
            except Exception as e:  # noqa: BLE001
                status = "fail"
                append_error(root, sid, f"worker_crash:{e}")
            counts[status] = counts.get(status, 0) + 1
            done_n += 1
            with _print_lock:
                print(f"[{done_n}/{len(todo)}] {sid} -> {status}", flush=True)

    summary = {
        "pack": str(root),
        "qg_model": args.qg_model,
        "filter": args.filter,
        "counts": counts,
        "elapsed_sec": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": PROTOCOL,
    }
    (out_dir / "_build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
