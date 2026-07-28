#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instr QA v2 shared helpers (TIFA / MultiHuman-style).

Protocol: docs/02_pipeline_design/eval_instr_vqa_redesign.md

Layout:
  $PACK/instr_qa_v2/<sample_id>.json          # frozen questions (Stage A)
  $PACK/judgments/instr_v2/<model_id>/<sid>.json  # answers (Stage B)
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable

import requests
from PIL import Image

try:
    from closedsource.gateway import gateway_base, gateway_key
except Exception:  # pragma: no cover
    gateway_base = None  # type: ignore
    gateway_key = None  # type: ignore

DEFAULT_MODEL = "gpt-5.5"
MAX_SIDE = 1024
PROTOCOL = "instr_qa_v2"
REVISION = "mpie_hard_v2.1"
MAIN_BUCKETS = ("role", "asymm", "prop")
ALL_BUCKETS = ("role", "asymm", "prop", "scene")
# Subtypes that enter primary S_instr (MPIE-discriminative)
MAIN_SUBTYPES = ("asymm", "role_duty", "prop_object")
# Weights over available subtype means (renormalized if a subtype is missing)
SUBTYPE_WEIGHTS = {"asymm": 0.50, "role_duty": 0.35, "prop_object": 0.15}
YN = {"yes": 1.0, "partial": 0.5, "no": 0.0}
ANSWER_VOCAB = set(YN)  # yes|partial|no

# Heuristic forbidden patterns for Instr (belong to other axes)
_FORBIDDEN_Q = re.compile(
    r"(?i)\b("
    r"how many|number of people|count of|"
    r"hugging\b|handshake|shaking hands|high[- ]five|"
    r"are (they|the (two )?people) (hugging|kissing|fighting|wrestling|dancing|carrying)|"
    r"is there (an? )?(hug|handshake|fight|dance)|"
    r"look like|resemble|same (person|face|identity)|identity|"
    r"extra (arm|leg|limb)|missing (arm|leg|limb)|fused|merged bod|penetrat|"
    r"anatom|limb ownership|floating (arm|leg|limb)"
    r")\b"
)

# Easy / non-MPIE patterns
_LEFT_RIGHT = re.compile(
    r"(?i)\b("
    r"on the left|on the right|left side|right side|"
    r"positioned on the left|positioned on the right|"
    r"standing on the left|standing on the right|"
    r"seated on the left|seated on the right"
    r")\b"
)
_DUTY = re.compile(
    r"(?i)\b("
    r"support(?:ing|ed|s)?|carry(?:ing|ied|ies)?|lift(?:ing|ed|s)?|"
    r"hold(?:ing|s)?|press(?:ing|es|ed)?|wrap(?:ping|s|ped)?|"
    r"grip(?:ping|s|ped)?|lean(?:ing|s)?|rest(?:ing|s)?|"
    r"drape(?:d|s)?|absorb(?:s|ing)?|punch(?:ing|es)?|"
    r"trainer|boxer|mitt|from behind|from below|from above|"
    r"rather than|instead of|who is|the one (who|with)|"
    r"tying|tie[sd]?|piggyback|on (?:his|her|their) back"
    r")\b"
)
_CLOTHING_ONLY = re.compile(
    r"(?i)\b(wearing|shirt|dress|hoodie|jacket|shorts|pants|skirt|polo|"
    r"crop top|long-sleeved|sleeve|gi\b|sneakers|shoes)\b"
)
_PROP_OBJECT = re.compile(
    r"(?i)\b("
    r"mitt|glove|bag|phone|smartphone|bouquet|bottle|box(?:es)?|"
    r"rope|mat|peon(?:y|ies)|target|focus mitt|song book|umbrella|"
    r"holding a|jointly holding|stack of"
    r")\b"
)
_SUBJECTIVE_FINE = re.compile(
    r"(?i)\b("
    r"slightly|gently|directly above|eyes cast|head tilted|"
    r"looking slightly|smiles? gently"
    r")\b"
)


def api_url() -> str:
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


def api_key() -> str:
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


def parse_json_obj(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # object or array
    m = re.search(r"[\{\[].*[\}\]]", text, re.S)
    if not m:
        raise ValueError(f"no JSON in model output: {text[:240]}")
    return json.loads(m.group(0))


def chat_json(
    *,
    messages: list[dict],
    model: str,
    timeout: int = 180,
    retries: int = 3,
    temperature: float = 0.1,
    max_tokens: int = 4096,
) -> Any:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
    }
    url = api_url()
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
            return parse_json_obj(msg.get("content") or "")
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"[:300]
            if attempt < retries:
                time.sleep(2**attempt + 1)
    raise RuntimeError(last or "chat_json failed")


def img_b64_jpeg(path: Path) -> str:
    im = Image.open(path).convert("RGB")
    if max(im.size) > MAX_SIDE:
        im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getbuffer()).decode("ascii")


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def infer_n_person(row: dict) -> int:
    if row.get("n_person"):
        return int(row["n_person"])
    return max(1, len(row.get("ref_relpaths") or []))


def infer_n_bystander(row: dict) -> int:
    if row.get("n_bystander") is not None:
        return int(row["n_bystander"])
    p = (row.get("prompt") or "").lower()
    if "bystander" in p or "passersby" in p or "passerby" in p:
        return 1
    return 0


def ref_labels(n_person: int) -> list[str]:
    return [f"R{i}" for i in range(1, n_person + 1)]


def qa_path(root: Path, sample_id: str) -> Path:
    return root / "instr_qa_v2" / f"{sample_id}.json"


def judgment_path(root: Path, model_id: str, sample_id: str) -> Path:
    return root / "judgments" / "instr_v2" / model_id / f"{sample_id}.json"


def find_gen_image(root: Path, model_id: str, sample_id: str) -> Path | None:
    d = root / "outputs" / model_id
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        p = d / f"{sample_id}{ext}"
        if p.is_file() and p.stat().st_size > 1000:
            return p
    return None


def normalize_answer(a: Any) -> str | None:
    if a is None:
        return None
    s = str(a).strip().lower()
    if s in ANSWER_VOCAB:
        return s
    if s in ("y", "true", "1"):
        return "yes"
    if s in ("n", "false", "0"):
        return "no"
    if s in ("p", "half", "somewhat"):
        return "partial"
    return None


def map_answer(a: Any) -> float | None:
    n = normalize_answer(a)
    return YN[n] if n is not None else None


def question_looks_forbidden(q: str) -> bool:
    return bool(_FORBIDDEN_Q.search(q or ""))


def classify_subtype(bucket: str, q: str, *, swap_sensitive: bool = False) -> str:
    """Map (bucket, question text) → fine subtype for MPIE-hard scoring."""
    bucket = (bucket or "").lower()
    qq = q or ""
    if bucket == "asymm":
        return "asymm"
    if bucket == "scene":
        return "scene"
    if bucket == "prop":
        if _PROP_OBJECT.search(qq):
            return "prop_object"
        return "prop_clothing"
    # role
    has_lr = bool(_LEFT_RIGHT.search(qq))
    has_duty = bool(_DUTY.search(qq))
    if has_duty:
        return "role_duty"
    if has_lr or re.search(r"(?i)\b(looking|facing|smiling|gaze)\b", qq):
        return "role_spatial"
    return "role_duty" if swap_sensitive else "role_spatial"


def is_main_subtype(subtype: str) -> bool:
    return subtype in MAIN_SUBTYPES


def clean_questions(raw: Iterable[dict], *, n_person: int) -> list[dict]:
    """Normalize / drop bad QAs from Stage A LLM output."""
    out: list[dict] = []
    seen = set()
    refs = set(ref_labels(n_person))
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        q = " ".join(str(item.get("q") or "").split()).strip()
        if len(q) < 12 or len(q) > 220:
            continue
        bucket = str(item.get("bucket") or "").strip().lower()
        if bucket not in ALL_BUCKETS:
            continue
        gold = normalize_answer(item.get("gold_a") or item.get("gold") or "yes")
        if gold != "yes":
            continue
        if question_looks_forbidden(q):
            continue
        # Drop overly subjective micro-details (VLM nitpicks, low signal)
        if _SUBJECTIVE_FINE.search(q) and not _DUTY.search(q):
            continue
        mentioned = {r for r in refs if re.search(rf"\b{r}\b", q, re.I)}
        if bucket in ("role", "asymm") and n_person >= 1 and not mentioned:
            if not _DUTY.search(q):
                continue
        swap_sensitive = bool(item.get("swap_sensitive"))
        # Force swap_sensitive for asymm / duty questions with ≥2 refs
        if bucket == "asymm" or (bucket == "role" and _DUTY.search(q)):
            swap_sensitive = True
        subtype = classify_subtype(bucket, q, swap_sensitive=swap_sensitive)
        # Drop easy spatial / clothing from the candidate pool entirely
        if subtype in ("role_spatial", "prop_clothing"):
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "id": str(item.get("id") or f"q{len(out)+1}"),
                "bucket": bucket,
                "subtype": subtype,
                "main": is_main_subtype(subtype),
                "q": q,
                "gold_a": "yes",
                "swap_sensitive": swap_sensitive,
                "element": str(item.get("element") or "")[:120],
            }
        )
    for i, item in enumerate(out, 1):
        item["id"] = f"q{i}"
    return out


def synthesize_fallback_questions(
    *,
    instruction: str,
    n_person: int,
    cat: str = "",
) -> list[dict]:
    """Deterministic hard QAs when LLM bank is too thin (near-symmetric contacts)."""
    refs = ref_labels(n_person)
    if len(refs) < 2:
        return []
    r1, r2 = refs[0], refs[1]
    cat_l = (cat or "").lower()
    instr_l = (instruction or "").lower()
    out: list[dict] = []

    def _add(bucket: str, q: str, subtype: str, element: str) -> None:
        out.append(
            {
                "id": f"fb{len(out)+1}",
                "bucket": bucket,
                "subtype": subtype,
                "main": is_main_subtype(subtype),
                "q": q,
                "gold_a": "yes",
                "swap_sensitive": True,
                "element": element,
            }
        )

    if n_person >= 3:
        r3 = refs[2]
        _add(
            "asymm",
            f"Is {r2} an onlooker / non-contact person relative to the main hand contact "
            f"between {r1} and {r3}, rather than {r2} being one of the two making contact?",
            "asymm",
            "onlooker vs contact-pair duty",
        )
        _add(
            "role",
            f"Are {r1} and {r3} the two people whose hands meet in the contact, "
            f"rather than {r2} joining that hand contact?",
            "role_duty",
            "contact-pair membership",
        )
    elif any(k in cat_l or k in instr_l for k in ("high_five", "high-five", "handshake")):
        verb = "high-five" if ("high_five" in cat_l or "high-five" in instr_l) else "handshake"
        _add(
            "asymm",
            f"Is {r1} raising a hand to complete a {verb} with {r2}, "
            f"rather than {r1} remaining fully passive with both hands down?",
            "asymm",
            f"{verb} initiator/active hand",
        )
        _add(
            "role",
            f"Are {r1} and {r2} the two people whose hands meet in the {verb}, "
            f"rather than interacting with someone else?",
            "role_duty",
            f"{verb} contact pair",
        )
        _add(
            "role",
            f"Is {r2} also raising a hand toward {r1} for the {verb}, "
            f"rather than keeping both hands down?",
            "role_duty",
            f"{verb} reciprocal raise",
        )
    else:
        _add(
            "asymm",
            f"Does {r1} take the supporting / leading contact role relative to {r2}, "
            f"rather than {r2} taking that role toward {r1}?",
            "asymm",
            "asymmetric contact duty",
        )
        _add(
            "role",
            f"Are {r1} and {r2} the two people executing the instructed contact interaction?",
            "role_duty",
            "contact pair",
        )

    return out


def merge_with_fallback(
    questions: list[dict],
    fallback: list[dict],
    *,
    n_person: int,
) -> tuple[list[dict], list[str]]:
    """Fill missing hard subtypes from deterministic fallback, then re-enforce quota."""
    have = {(q.get("subtype"), (q.get("q") or "").lower()) for q in questions}
    have_sub = {q.get("subtype") for q in questions}
    merged = list(questions)
    for fb in fallback:
        key = (fb.get("subtype"), (fb.get("q") or "").lower())
        if key in have:
            continue
        if fb.get("subtype") in have_sub and fb.get("subtype") != "asymm":
            # still allow second role_duty if we have <2 of that subtype
            n_same = sum(1 for q in merged if q.get("subtype") == fb.get("subtype"))
            if fb.get("subtype") == "role_duty" and n_same >= 2:
                continue
            if fb.get("subtype") == "asymm" and n_same >= 2:
                continue
            if fb.get("subtype") not in ("role_duty", "asymm") and n_same >= 1:
                continue
        if fb.get("subtype") == "asymm" and "asymm" in have_sub:
            n_a = sum(1 for q in merged if q.get("subtype") == "asymm")
            if n_a >= 2:
                continue
        merged.append(fb)
        have.add(key)
        have_sub.add(fb.get("subtype"))
    kept, warnings = enforce_quota(merged, n_person=n_person)
    if any(q.get("id", "").startswith("fb") or "fallback" in str(q.get("element", "")) for q in kept):
        warnings.append("used_fallback")
    # mark fallback provenance
    for q in kept:
        if str(q.get("id", "")).startswith("fb") or q.get("element", "").endswith("(fallback)"):
            q["from_fallback"] = True
    return kept, warnings


def enforce_quota(questions: list[dict], *, n_person: int) -> tuple[list[dict], list[str]]:
    """Prefer MPIE-hard subtypes; return (kept, warnings)."""
    warnings: list[str] = []
    by_sub: dict[str, list[dict]] = {s: [] for s in (
        "asymm", "role_duty", "prop_object", "scene", "role_spatial", "prop_clothing"
    )}
    for q in questions:
        by_sub.setdefault(q.get("subtype") or "role_spatial", []).append(q)

    kept: list[dict] = []
    # Primary discriminative budget
    kept.extend(by_sub["asymm"][:3])          # up to 3
    kept.extend(by_sub["role_duty"][:2])      # up to 2
    kept.extend(by_sub["prop_object"][:2])    # up to 2
    # scene optional diagnostic (not in S_instr)
    if by_sub["scene"]:
        sc = dict(by_sub["scene"][0])
        sc["main"] = False
        sc["subtype"] = "scene"
        kept.append(sc)

    if n_person >= 2 and len(by_sub["asymm"]) < 1:
        warnings.append("asymm_missing")
    if n_person >= 2 and len(by_sub["asymm"]) < 2:
        warnings.append("asymm_lt_2")
    if len(by_sub["role_duty"]) < 1 and n_person >= 2:
        warnings.append("role_duty_missing")
    main_kept = [q for q in kept if q.get("main")]
    if len(main_kept) < 3:
        warnings.append("main_lt_3")

    for i, item in enumerate(kept, 1):
        item["id"] = f"q{i}"
        item["main"] = bool(item.get("main")) and is_main_subtype(
            item.get("subtype") or ""
        )
    return kept, warnings


def score_answers(
    questions: list[dict], answers: dict[str, str]
) -> dict[str, Any]:
    """Compute weighted S_instr over MPIE-hard subtypes (+ diagnostics)."""
    per_bucket: dict[str, list[float]] = {b: [] for b in ALL_BUCKETS}
    per_sub: dict[str, list[float]] = {s: [] for s in MAIN_SUBTYPES}
    per_sub["scene"] = []
    per_sub["role_spatial"] = []
    per_sub["prop_clothing"] = []
    detail = []

    for q in questions:
        qid = q["id"]
        a = normalize_answer(answers.get(qid))
        val = map_answer(a)
        bucket = q.get("bucket") or "role"
        subtype = q.get("subtype") or classify_subtype(
            bucket, q.get("q") or "", swap_sensitive=bool(q.get("swap_sensitive"))
        )
        main = bool(q.get("main")) if "main" in q else is_main_subtype(subtype)
        rec = {
            "id": qid,
            "bucket": bucket,
            "subtype": subtype,
            "main": main,
            "q": q["q"],
            "a": a,
            "score": val,
            "gold_a": q.get("gold_a", "yes"),
        }
        detail.append(rec)
        if val is None:
            continue
        per_bucket.setdefault(bucket, []).append(val)
        per_sub.setdefault(subtype, []).append(val)

    def _m(xs: list[float]) -> float | None:
        return sum(xs) / len(xs) if xs else None

    # Weighted mean over available main subtype means
    w_sum = 0.0
    s_sum = 0.0
    for st, w in SUBTYPE_WEIGHTS.items():
        m = _m(per_sub.get(st) or [])
        if m is None:
            continue
        w_sum += w
        s_sum += w * m
    s_instr = (s_sum / w_sum) if w_sum > 0 else None

    # Flat mean of main-flagged items (diagnostic)
    main_vals = [d["score"] for d in detail if d.get("main") and d["score"] is not None]

    return {
        "S_instr": s_instr,
        "S_instr_flat_main": _m(main_vals),
        "S_instr_scene": _m(per_sub.get("scene") or []),
        "S_instr_role": _m(per_bucket.get("role") or []),
        "S_instr_asymm": _m(per_sub.get("asymm") or []),
        "S_instr_prop": _m(per_bucket.get("prop") or []),
        "S_instr_role_duty": _m(per_sub.get("role_duty") or []),
        "S_instr_prop_object": _m(per_sub.get("prop_object") or []),
        "n_main": len(main_vals),
        "n_all": sum(1 for d in detail if d["score"] is not None),
        "detail": detail,
        "weights": {k: v for k, v in SUBTYPE_WEIGHTS.items()},
        "revision": REVISION,
    }


def frozen_qa_ok(obj: dict) -> bool:
    if not isinstance(obj, dict):
        return False
    qs = obj.get("questions")
    if not isinstance(qs, list) or len(qs) < 2:
        return False
    n_person = int(obj.get("n_person") or 2)
    main = [
        q
        for q in qs
        if isinstance(q, dict)
        and (
            q.get("main")
            or is_main_subtype(q.get("subtype") or "")
        )
    ]
    if len(main) < 2:
        return False
    asymm_n = sum(
        1
        for q in qs
        if isinstance(q, dict)
        and (q.get("subtype") == "asymm" or q.get("bucket") == "asymm")
    )
    role_duty_n = sum(
        1 for q in qs if isinstance(q, dict) and q.get("subtype") == "role_duty"
    )
    # MPIE-hard: prefer asymm; allow role_duty-heavy banks for near-symmetric contacts
    if n_person >= 2 and asymm_n < 1 and role_duty_n < 2:
        return False
    for q in qs:
        if not isinstance(q, dict):
            return False
        if q.get("bucket") not in ALL_BUCKETS:
            return False
        if not (q.get("q") and q.get("id")):
            return False
        if normalize_answer(q.get("gold_a")) != "yes":
            return False
    # Reject banks that are still dominated by easy spatial (legacy)
    spatial = sum(1 for q in qs if q.get("subtype") == "role_spatial")
    if spatial > 0 and spatial >= len(main):
        return False
    return True


def judgment_ok(obj: dict, *, require_judge: str | None = None) -> bool:
    if not isinstance(obj, dict):
        return False
    if obj.get("ok") is False:
        return False
    if obj.get("S_instr") is None and obj.get("n_main", 0) == 0:
        if not obj.get("detail"):
            return False
    meta = obj.get("_meta") if isinstance(obj.get("_meta"), dict) else {}
    if require_judge and meta.get("judge_model") != require_judge:
        return False
    if meta.get("protocol") not in (None, PROTOCOL):
        if obj.get("S_instr") is None:
            return False
    return True
