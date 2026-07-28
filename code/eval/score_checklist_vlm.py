#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anat/Inter Checklist_V — VLM Walk the same way as humans 0/1 Sample table (not the old six-axis score_vlm_v1）。

protocol: docs/02_pipeline_design/eval_human_consistency_anat_inter.md §5
Place the order: $PACK/judgments/human_consistency/checklist_vlm/<judge>/<model_id>/<sample_id>.json

By default it only runs _split.json Entries in (with H/M same set).

usage:
  # and VQA(score_vlm_v1) Same gateway:gpt-5.5 + AI_GATEWAY_URL / AI_GATEWAY_KEY
  set -a; source your env file; set +a
  python score_checklist_vlm.py --pack "$MPIE_TEST_PACK" --split pilot --limit 2
  bash run_score_checklist_vlm.sh pilot
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import (  # noqa: E402
    ANAT_ITEMS,
    INTER_ITEMS,
    OVERALL_ITEMS,
    PROTOCOL_ID,
    SCHEME_VERSION,
    anat_pass_from_items,
    apply_inter_dependencies,
    atomic_write_json,
    construct_scores,
    inter_pass_from_items,
    n_expected_from_prompt,
    normalize_anat_item,
    normalize_inter_item,
    normalize_overall,
    pair_key,
    validate_checklist_payload,
)
from instr_qa_common import api_key, api_url, chat_json  # noqa: E402  # and VQA/Instr Same origin gateway
from mesh_metrics import prompt_contact_intent  # noqa: E402
from pack_io import pack_root  # noqa: E402

DEFAULT_JUDGE = "gpt-5.5"
MAX_SIDE = 1024
_print_lock = threading.Lock()
_err_lock = threading.Lock()

SYSTEM_SKELETON = """You are a STRICT rater for multi-person interaction QUALITY (construct validity).
Protocol: checklist_anat_inter_v5 / analysis v5.0. Prefer FAIL / lower severity when borderline — do NOT inflate.
Do NOT give full marks (Q=5 or severity=2) unless clearly deserved. Use the FULL scale; many images are mediocre.

Encoding:
  Overall Q_*: Likert 1=very bad … 5=very good (MUST always fill; no U). Anchors:
    Q_inter: 1=fuse/total interaction fail; 3=partial with clear fuse/contact issues; 5=close contact, no pathological fuse, region OK.
    Q_anat: 1=extra limbs/severe deformity; 3=medium structure issues; 5=clean ownership + proportions.
  Binary items (I0,I3,Ir,A4,A5): 1 = NORMAL/OK, 0 = PROBLEM, "U" = unreadable.
  ORDINAL SEVERITY (I1,A1,A2,A3): 2=clean, 1=mild issue, 0=severe. DO NOT mark 2 if mild defects exist.
  Ic ORDINAL contact: 0=no contact, 1=contact but not close, 2=close contact established.

Decision tree:
  - Unseparable fused blob → I1=0; Anat items U (do not double-count).
  - Separable + clear limb through torso/head → I1=0; mild stickiness/shallow penetrate → I1=1; clean → I1=2.
  - Separable + extra/floating parts → grade A1 0/1/2 (not automatic I1=0).
  - Separable + only hands mushy → I1=2 (or U), A5=0; NEVER fail I1 for hands alone.
  - Ic = closeness only; Ir = body region only. Do NOT set Ir=0 just because not close.
  - If Ic=0, Ir=null. Person count ONLY on I0.

Legitimate hug/wrestle contact is NOT pathological fusion.
Ignore clothing folds for A1. Ignore unnamed bystanders for I0.

System-guessed contact intent (HINT ONLY, may be wrong — judge from the edit instruction yourself): {intent}
Edit instruction:
\"\"\"
{prompt}
\"\"\"
Expected main persons (R# count): {n_expected}

IMPORTANT (protocol v5.1): Answer ALL Inter items I0,I1,Ic,I3 always.
Ir is null ONLY when Ic=0. Do NOT leave Ic/I3 null because of the system-guessed intent.

Return ONLY JSON:
{{
  "intent_used": "required|forbidden|unspecified",
  "overall": {{
    "Q_inter": 1|2|3|4|5,
    "Q_anat": 1|2|3|4|5
  }},
  "inter": {{
    "I0": 0|1|"U",
    "I1": 0|1|2|"U",
    "Ic": 0|1|2|"U",
    "I3": 0|1|"U",
    "Ir": 0|1|"U"|null
  }},
  "anat": {{
    "A1": 0|1|2|"U",
    "A2": 0|1|2|"U",
    "A3": 0|1|2|"U",
    "A4": 0|1|"U",
    "A5": 0|1|"U"
  }},
  "notes": "<one short sentence>"
}}

Items:
intent_used: YOUR judgment of contact intent from the edit instruction (not the system hint)
Q_inter overall interaction quality 1–5 (use full range)
Q_anat overall anatomy quality 1–5 (use full range)
I0 count OK vs instruction (binary) — ALWAYS
I1 fuse/penetration SEVERITY 2/1/0 — ALWAYS
Ic ordinal contact quality 0/1/2 — ALWAYS
I3 NO unwanted/extra cling (binary) — ALWAYS
Ir contact region matches instruction — null ONLY if Ic=0; else ALWAYS
A1 extra-structure SEVERITY 2/1/0
A2 body-shape SEVERITY 2/1/0
A3 ownership SEVERITY 2/1/0
A4 relative scale OK (binary)
A5 hands roughly recognizable (binary; IN construct)
"""


def img_b64_jpeg(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getbuffer()).decode("ascii")


def find_gen_image(root: Path, model_id: str, sample_id: str) -> Optional[Path]:
    dirs = [
        root / "outputs" / model_id,
        root / "judgments" / "human_consistency" / "media" / model_id,
    ]
    for d in dirs:
        if not d.is_dir():
            continue
        for ext in (".png", ".jpg", ".jpeg", ".webp"):
            p = d / f"{sample_id}{ext}"
            if p.is_file() and p.stat().st_size > 1000:
                return p
    return None


def out_path(root: Path, judge: str, model_id: str, sample_id: str) -> Path:
    safe = judge.replace("/", "_")
    return (
        root
        / "judgments"
        / "human_consistency"
        / "checklist_vlm"
        / safe
        / model_id
        / f"{sample_id}.json"
    )


def load_split_items(split_path: Path, names: List[str]) -> List[dict]:
    data = json.loads(split_path.read_text(encoding="utf-8"))
    want = set(names)
    items: List[dict] = []
    for name, rows in (data.get("splits") or {}).items():
        if name not in want:
            continue
        for r in rows:
            d = dict(r)
            d["_split"] = name
            items.append(d)
    return items


def normalize_judgment(obj: dict, *, intent: str) -> dict:
    """Normalized fields;v5.1: Not according to the system intent Clear Ic/I3/Ir；S use intent_used。"""
    inter_in = obj.get("inter") or {}
    anat_in = obj.get("anat") or {}
    overall_in = obj.get("overall") or {}
    # Compatible with top Q_*
    if "Q_inter" in obj and "Q_inter" not in overall_in:
        overall_in = {**overall_in, "Q_inter": obj.get("Q_inter")}
    if "Q_anat" in obj and "Q_anat" not in overall_in:
        overall_in = {**overall_in, "Q_anat": obj.get("Q_anat")}
    intent_used = str(obj.get("intent_used") or intent or "unspecified").lower()
    if intent_used not in ("required", "forbidden", "unspecified"):
        intent_used = str(intent or "unspecified").lower()
    # Same as humans:gate_by_intent=False, answer all Inter
    inter = {
        k: normalize_inter_item(
            k, inter_in.get(k), intent=intent_used, gate_by_intent=False
        )
        for k in INTER_ITEMS
    }
    inter = apply_inter_dependencies(inter, intent_used)
    anat = {k: normalize_anat_item(k, anat_in.get(k)) for k in ANAT_ITEMS}
    overall = {k: normalize_overall(overall_in.get(k)) for k in OVERALL_ITEMS}
    scores = construct_scores(inter, anat, intent=intent_used)
    return {
        "protocol": PROTOCOL_ID,
        "scheme": SCHEME_VERSION,
        "intent_system": intent,
        "intent_used": intent_used,
        "overall": overall,
        "Q_inter": overall["Q_inter"],
        "Q_anat": overall["Q_anat"],
        "inter": inter,
        "anat": anat,
        "notes": (obj.get("notes") or "")[:300],
        **scores,
        "S_inter_V": scores["S_inter_H"],
        "S_anat_V": scores["S_anat_H"],
        "Inter_pass": inter_pass_from_items(inter, intent_used),
        "Anat_pass": anat_pass_from_items(anat),
        "human_all_inter_items": True,
    }


def schema_ok_checklist(obj: dict, *, intent: str) -> bool:
    try:
        norm = normalize_judgment(obj, intent=intent)
    except Exception:
        return False
    errs = validate_checklist_payload(norm, allow_u=True)
    if errs:
        return False
    for k in OVERALL_ITEMS:
        if norm.get(k) not in (1, 2, 3, 4, 5):
            return False
    # v5.1：Inter Core items must not be gated and cleared
    inter = norm.get("inter") or {}
    for k in ("I0", "I1", "Ic", "I3"):
        if inter.get(k) is None:
            return False
    ic = inter.get("Ic")
    if isinstance(ic, int) and ic >= 1 and inter.get("Ir") is None:
        return False
    return True


def already_done(path: Path, require_judge: Optional[str]) -> bool:
    if not path.is_file() or path.stat().st_size < 20:
        return False
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    body = obj.get("judgment") if isinstance(obj.get("judgment"), dict) else obj
    meta = obj.get("_meta") if isinstance(obj.get("_meta"), dict) else {}
    intent = meta.get("intent") or body.get("intent_used") or "unspecified"
    if not schema_ok_checklist(body, intent=str(intent)):
        return False
    if require_judge and meta.get("judge_model") != require_judge:
        return False
    return True


def append_error(root: Path, judge: str, model_id: str, sample_id: str, err: str) -> None:
    p = (
        root
        / "judgments"
        / "human_consistency"
        / "checklist_vlm"
        / judge.replace("/", "_")
        / model_id
        / "_errors.jsonl"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "sample_id": sample_id,
            "model_id": model_id,
            "error": err[:500],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        ensure_ascii=False,
    )
    with _err_lock:
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def call_judge(
    *,
    gen: Path,
    prompt: str,
    intent: str,
    n_expected: int,
    judge_model: str,
    timeout: int,
    retries: int,
) -> dict:
    """Go with score_vlm_v1 / instr_qa same OpenAI compatible chat/completions gateway. """
    text = SYSTEM_SKELETON.format(
        intent=intent, prompt=prompt, n_expected=n_expected
    )
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64_jpeg(gen)}"},
        },
        {"type": "text", "text": text},
    ]
    obj = chat_json(
        messages=[{"role": "user", "content": content}],
        model=judge_model,
        timeout=timeout,
        retries=retries,
        temperature=0.0,
        max_tokens=2048,
    )
    if not isinstance(obj, dict):
        raise ValueError(f"judge returned non-object: {type(obj)}")
    norm = normalize_judgment(obj, intent=intent)
    errs = validate_checklist_payload(norm, allow_u=True)
    if errs:
        raise ValueError(f"schema: {errs[:5]}")
    return norm


def score_one(item: dict, *, root: Path, args) -> str:
    sid = item["sample_id"]
    mid = item["model_id"]
    out_p = out_path(root, args.judge_model, mid, sid)
    require = args.judge_model if args.resume else None
    if args.resume and already_done(out_p, require_judge=require):
        return "skip"

    gen = find_gen_image(root, mid, sid)
    if gen is None:
        append_error(root, args.judge_model, mid, sid, "missing_gen")
        return "missing_gen"

    intent = item.get("intent") or prompt_contact_intent(item.get("prompt"))
    n_exp = int(item.get("n_expected") or n_expected_from_prompt(item))
    prompt = item.get("prompt") or ""
    t0 = time.time()
    try:
        judgment = call_judge(
            gen=gen,
            prompt=prompt,
            intent=str(intent),
            n_expected=n_exp,
            judge_model=args.judge_model,
            timeout=args.timeout,
            retries=args.retries,
        )
    except Exception as e:  # noqa: BLE001
        append_error(root, args.judge_model, mid, sid, str(e))
        return "fail"

    payload = {
        "protocol": PROTOCOL_ID,
        "sample_id": sid,
        "model_id": mid,
        "key": pair_key(sid, mid),
        **judgment,
        "_meta": {
            "sample_id": sid,
            "model_id": mid,
            "judge_model": args.judge_model,
            "judge_endpoint": api_url(),
            "intent": intent,
            "n_expected": n_exp,
            "split": item.get("_split"),
            "seconds": round(time.time() - t0, 2),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gen_relpath": str(gen.relative_to(root)),
            "no_reference_images": True,
        },
    }
    atomic_write_json(out_p, payload)
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="")
    ap.add_argument("--split", default="pilot", help="Comma separated:guide,pilot,holdout,main")
    ap.add_argument("--judge-model", default=os.environ.get("MPIE_JUDGE_MODEL", DEFAULT_JUDGE))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--dry-run", action="store_true", help="Only list todo, not tuned API")
    args = ap.parse_args()

    root = pack_root(args.pack or None)
    split_path = root / "judgments" / "human_consistency" / "_split.json"
    if not split_path.is_file():
        raise SystemExit(f"missing {split_path}; run select_consistency_split.py first")

    names = [x.strip() for x in args.split.split(",") if x.strip()]
    items = load_split_items(split_path, names)
    require = args.judge_model if args.resume else None
    todo = []
    for it in items:
        p = out_path(root, args.judge_model, it["model_id"], it["sample_id"])
        if args.resume and already_done(p, require_judge=require):
            continue
        if find_gen_image(root, it["model_id"], it["sample_id"]) is None:
            continue
        todo.append(it)
        if args.limit and len(todo) >= args.limit:
            break

    print(
        json.dumps(
            {
                "pack": str(root),
                "judge_model": args.judge_model,
                "endpoint": (api_url() if not args.dry_run else "(dry-run)"),
                "splits": names,
                "n_split": len(items),
                "todo": len(todo),
                "workers": args.workers,
                "resume": args.resume,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.dry_run or not todo:
        if not todo:
            print("nothing to do", flush=True)
        return

    counts = {"ok": 0, "skip": 0, "fail": 0, "missing_gen": 0}
    t0 = time.time()

    def _job(it: dict) -> Tuple[str, str]:
        st = score_one(it, root=root, args=args)
        return pair_key(it["sample_id"], it["model_id"]), st

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_job, it): it for it in todo}
        done_n = 0
        for fut in as_completed(futs):
            it = futs[fut]
            key = pair_key(it["sample_id"], it["model_id"])
            try:
                _, status = fut.result()
            except Exception as e:  # noqa: BLE001
                status = "fail"
                append_error(root, args.judge_model, it["model_id"], it["sample_id"], f"worker:{e}")
            counts[status] = counts.get(status, 0) + 1
            done_n += 1
            with _print_lock:
                print(f"[{done_n}/{len(todo)}] {key} -> {status}", flush=True)

    summary = {
        "pack": str(root),
        "judge_model": args.judge_model,
        "splits": names,
        "counts": counts,
        "elapsed_sec": round(time.time() - t0, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_root = root / "judgments" / "human_consistency" / "checklist_vlm" / args.judge_model.replace("/", "_")
    out_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_root / "_run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
