#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Anat/Inter checklist helpers: coding, construct scores, summaries, majority vote, tau, mesh mapping.

See:
  docs/02_pipeline_design/eval_construct_validity_principle.md
  docs/02_pipeline_design/eval_human_consistency_anat_inter.md
  docs/02_pipeline_design/eval_human_consistency_analysis_protocol.md
Current: checklist_anat_inter_v4 + analysis rules v4.1
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

PROTOCOL_ID = "checklist_anat_inter_v4"
SCHEME_VERSION = "v4.1"
ANALYSIS_PROTOCOL_ID = "anat_inter_analysis_v4.1"
MESH_BIN_PROTOCOL = "mesh_bin_map_v4"
THRESHOLDS_MAP_VERSION = "map_v4"

# S formulas (analysis protocol §1; components in [0,1]; nulls excluded from means)
# S_inter_req    = mean(I0, I1, Ic/2, [Ir if Ic>=1])
# S_inter_forb   = mean(I0, I1, I3)
# S_inter_unspec = mean(I0, I1)
# S_anat         = mean(A1..A5)
# Any core item marked U → sample-level S = None

# Inter: I0 count, I1 penetration, Ic contact ordinal (0/1/2), I3 forbidden contact, Ir region
INTER_ITEMS = ("I0", "I1", "Ic", "I3", "Ir")
# Anat: person-count lives in I0 only; A5 (hands) is in the main construct; no A6
ANAT_ITEMS = ("A1", "A2", "A3", "A4", "A5")
ANAT_PASS_ITEMS = ("A1", "A2", "A3", "A4", "A5")

BinVal = Union[int, str, None]

DEFAULT_THRESHOLDS: Dict[str, Any] = {
    "version": THRESHOLDS_MAP_VERSION,
    "protocol": PROTOCOL_ID,
    "tau_fuse": 0.30,
    "tau_miss_touch": 0.35,  # Ic>=1: any contact
    "tau_miss_close": 0.18,  # Ic==2: close contact
    "tau_dist_close": 0.08,
    "tau_unw": 0.40,
    "tau_extra": 0.30,
    "tau_resid": 0.18,
    "orphan_ok": 0.03,
    "leftover_ok": 0.68,
    "tau_own": 0.80,
    "tau_scale": 0.85,
    "tau_S_inter": 0.70,
    "tau_S_anat": 0.82,
    "status": "default_unfrozen",
    "note": "v4 construct mapping; freeze after pilot correlation/kappa lock",
}

DEFAULT_MODELS = [
    "flux1-kontext-dev",
    "dreamo",
    "gpt-image-2",
    "gemini-3-pro-image",
    "seedream-5-pro",
]

# Human overall preference anchors (1–5); parallel to item-level scores
OVERALL_ITEMS = ("Q_inter", "Q_anat")
OVERALL_QUESTIONS = [
    ("Q_inter", "Overall interaction quality: how well is the requested interaction realized? (1=poor … 5=excellent)"),
    ("Q_anat", "Overall anatomy quality: are human bodies clean and plausible? (1=poor … 5=excellent)"),
]

# Default model mix for consistency split (weaker open models dominate; dreamo replaces qwen)
DEFAULT_MODEL_MIX = {
    "flux1-kontext-dev": 0.35,
    "dreamo": 0.30,
    "gpt-image-2": 0.12,
    "gemini-3-pro-image": 0.12,
    "seedream-5-pro": 0.11,
}

# Question stems: binary items use 1=OK; Ic is ordinal 0/1/2
INTER_QUESTIONS = [
    (
        "I0",
        "Count OK: only instruction-named roles (R#) count; ignore unnamed bystanders/staff/onlookers",
        "always",
        "bin",
    ),
    (
        "I1",
        "Penetration OK: no limb through torso/head and no inseparable blob. "
        "Hand mush while people are separable is not an I1 fail (score A5). Touch/overlap perspective is OK",
        "always",
        "bin",
    ),
    (
        "Ic",
        "Contact quality (ordinal): 0=no contact (approach/eye-contact/gap); "
        "1=contact intended but not close enough; 2=close contact established. Score closeness only, not body region",
        "intent=required only",
        "ordinal3",
    ),
    (
        "I3",
        "Forbidden contact OK: no unintended touch/entanglement/embrace",
        "intent=forbidden only",
        "bin",
    ),
    (
        "Ir",
        "Region correct: judge whether the contact region roughly matches the instruction "
        "(e.g. hug requested but handshake → 0). Do not fail for insufficient closeness (that is Ic). Skip when Ic=0",
        "intent=required only; enters S_inter_req",
        "bin",
    ),
]
ANAT_QUESTIONS = [
    (
        "A1",
        "Structure clean: no extra/floating limbs or body fragments (clothing folds do not count)",
        "bin",
    ),
    (
        "A2",
        "Shape OK: no deformity/elongation/folded misalignment/collapse/impossible breaks. "
        "Exaggerated pose still readable as a bend → 1; clear break/misalign → 0; unsure → U",
        "bin",
    ),
    ("A3", "Ownership OK: no limb attached to the wrong person", "bin"),
    ("A4", "Scale OK: relative body sizes of the two people are not clearly absurd", "bin"),
    (
        "A5",
        "Hands OK: hands roughly recognizable (severe mush/extra-finger pile → 0). "
        "If people are separable and only hands are mushy: I1 may stay 1; score only this item 0",
        "bin",
    ),
]

DECISION_TREE = [
    "Inseparable blob → set I1=0 only; mark Anat as U; do not double-count the same blob on A1/A3/A5.",
    "Separable people + limb through torso/head → I1=0; do not auto-set A1=0.",
    "Separable people + extra limbs/fragments → A1=0; do not auto-set I1=0.",
    "Separable people + hands mush only → I1=1 (or U), A5=0; do not fail I1 for hand mush.",
    "Ic scores closeness only; Ir scores region only. Ir must not fail for lack of closeness. Ic=0 → Ir=null.",
    "I0: only instruction-named roles; ignore bystanders. Person count is scored only via I0.",
]


def inter_family(intent: str) -> str:
    intent = (intent or "unspecified").lower()
    if intent == "required":
        return "req"
    if intent == "forbidden":
        return "forb"
    return "unspec"


def atomic_write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def n_expected_from_prompt(row: dict) -> int:
    for k in ("n_person", "n_people", "num_person", "n_expected"):
        if k in row and row[k] is not None:
            return int(row[k])
    prompt = row.get("prompt") or ""
    ids = set(re.findall(r"\bR(\d+)\b", prompt))
    if ids:
        return int(len(ids))
    refs = row.get("ref_relpaths") or []
    ref_ids = set()
    for p in refs:
        m = re.search(r"(?:^|/|\\)R(\d+)_", str(p))
        if m:
            ref_ids.add(m.group(1))
    if ref_ids:
        return int(len(ref_ids))
    return max(2, len(refs)) if refs else 2


def normalize_overall(v: Any) -> BinVal:
    """Overall score 1–5; U/null forbidden (must rate)."""
    if v is None:
        raise ValueError("overall score required (1-5)")
    if isinstance(v, (int, float)) and v == int(v):
        iv = int(v)
        if 1 <= iv <= 5:
            return iv
        raise ValueError(f"overall must be 1-5, got {v!r}")
    s = str(v).strip()
    if s.isdigit():
        iv = int(s)
        if 1 <= iv <= 5:
            return iv
    raise ValueError(f"overall must be 1-5, got {v!r}")


def normalize_code(
    v: Any,
    *,
    allow_u: bool = True,
    allow_null: bool = True,
    max_int: int = 1,
) -> BinVal:
    if v is None:
        return None if allow_null else "U"
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)) and v == int(v):
        iv = int(v)
        if 0 <= iv <= max_int:
            return iv
        raise ValueError(f"invalid checklist code: {v!r} (max_int={max_int})")
    s = str(v).strip()
    if s.isdigit():
        iv = int(s)
        if 0 <= iv <= max_int:
            return iv
        raise ValueError(f"invalid checklist code: {v!r} (max_int={max_int})")
    if s.upper() == "U":
        return "U" if allow_u else None
    if s.lower() in ("null", "none", "na", "n/a"):
        return None if allow_null else "U"
    raise ValueError(f"invalid checklist code: {v!r}")


def normalize_inter_item(k: str, v: Any, *, intent: str) -> BinVal:
    intent = (intent or "unspecified").lower()
    if k == "Ic":
        if intent != "required":
            return None
        return normalize_code(v, allow_u=True, allow_null=True, max_int=2)
    if k == "Ir":
        if intent != "required":
            return None
        return normalize_code(v, allow_u=True, allow_null=True, max_int=1)
    if k == "I3":
        if intent != "forbidden":
            return None
        return normalize_code(v, allow_u=True, allow_null=True, max_int=1)
    return normalize_code(v, allow_u=True, allow_null=True, max_int=1)


def apply_inter_dependencies(inter: dict, intent: str) -> dict:
    """Ic=0 → force Ir=null; non-applicable intent fields cleared in normalize."""
    out = dict(inter)
    intent = (intent or "unspecified").lower()
    if intent == "required":
        ic = out.get("Ic")
        if ic in (0, "0"):
            out["Ir"] = None
    return out


def inter_construct_score(inter: dict, intent: str) -> Optional[float]:
    """S_inter_* in [0,1] (analysis protocol §1.1). Core item U/missing → None.

    Ic_norm = Ic/2; Ir enters the mean only when Ic>=1; nulls excluded from denominator.
    """
    intent = (intent or "unspecified").lower()
    inter = apply_inter_dependencies(inter, intent)
    parts: List[float] = []

    for k in ("I0", "I1"):
        v = normalize_inter_item(k, inter.get(k), intent=intent)
        if v is None or v == "U":
            return None
        parts.append(float(int(v)))

    if intent == "required":
        ic = normalize_inter_item("Ic", inter.get("Ic"), intent=intent)
        if ic is None or ic == "U":
            return None
        parts.append(float(int(ic)) / 2.0)  # Ic_norm
        if int(ic) >= 1:
            ir = normalize_inter_item("Ir", inter.get("Ir"), intent=intent)
            if ir is None or ir == "U":
                return None
            parts.append(float(int(ir)))
    elif intent == "forbidden":
        v = normalize_inter_item("I3", inter.get("I3"), intent=intent)
        if v is None or v == "U":
            return None
        parts.append(float(int(v)))

    return float(sum(parts) / len(parts)) if parts else None


def anat_construct_score(anat: dict) -> Optional[float]:
    """S_anat = mean(A1..A5); any core item U/missing → None."""
    parts: List[float] = []
    for k in ANAT_PASS_ITEMS:
        v = normalize_code(anat.get(k), allow_u=True, allow_null=True, max_int=1)
        if v is None or v == "U":
            return None
        parts.append(float(int(v)))
    return float(sum(parts) / len(parts)) if parts else None


def construct_scores(inter: dict, anat: dict, *, intent: str) -> dict:
    """Compute human-side construct fields once (for persistence)."""
    fam = inter_family(intent)
    return {
        "analysis_protocol": ANALYSIS_PROTOCOL_ID,
        "S_inter_family": fam,
        "S_inter_H": inter_construct_score(inter, intent),
        "S_anat_H": anat_construct_score(anat),
        "S_inter_H_geom": _inter_construct_geom_only(inter, intent),
        "S_anat_H_geom": _anat_construct_geom_only(anat),
    }


def inter_pass_from_items(inter: dict, intent: str) -> Optional[int]:
    """Binary appendix summary: construct score==1 counts as pass (required needs Ic==2 and Ir==1)."""
    intent = (intent or "unspecified").lower()
    inter = apply_inter_dependencies(inter, intent)
    for k in ("I0", "I1"):
        v = normalize_inter_item(k, inter.get(k), intent=intent)
        if v is None or v == "U":
            return None
        if int(v) != 1:
            return 0
    if intent == "required":
        ic = normalize_inter_item("Ic", inter.get("Ic"), intent=intent)
        if ic is None or ic == "U":
            return None
        if int(ic) < 2:
            return 0
        ir = normalize_inter_item("Ir", inter.get("Ir"), intent=intent)
        if ir is None or ir == "U":
            return None
        return int(int(ir) == 1)
    if intent == "forbidden":
        v = normalize_inter_item("I3", inter.get("I3"), intent=intent)
        if v is None or v == "U":
            return None
        return int(int(v) == 1)
    return 1


def anat_pass_from_items(anat: dict) -> Optional[int]:
    vals = []
    for k in ANAT_PASS_ITEMS:
        v = normalize_code(anat.get(k), allow_u=True, allow_null=True, max_int=1)
        if v is None or v == "U":
            return None
        vals.append(int(v))
    return int(all(vals))


def majority_vote(codes: Sequence[Any], *, max_int: int = 1) -> Tuple[BinVal, str]:
    normed = [
        normalize_code(c, allow_u=True, allow_null=True, max_int=max_int) for c in codes
    ]
    if all(v is None for v in normed):
        return None, "all_null"
    hard = [v for v in normed if isinstance(v, int)]
    us = [v for v in normed if v == "U"]
    if len(hard) >= 2:
        c = Counter(hard)
        top, n = c.most_common(1)[0]
        if n >= 2:
            return int(top), "ok"
        if len(us) >= 1:
            return None, "drop_conflict_with_u"
        return None, "drop_no_majority"
    if len(hard) == 1 and len(us) >= 1:
        return None, "drop_no_majority"
    if len(us) >= 2 and len(hard) == 0:
        return None, "drop_no_majority"
    return None, "drop_no_majority"


def consensus_checklist(anns: Sequence[dict], *, intent: str) -> dict:
    inter_c: Dict[str, Any] = {}
    anat_c: Dict[str, Any] = {}
    drop: Dict[str, str] = {}
    intent = (intent or "unspecified").lower()

    for k in INTER_ITEMS:
        if k == "Ic" and intent != "required":
            inter_c[k] = None
            continue
        if k == "Ir" and intent != "required":
            inter_c[k] = None
            continue
        if k == "I3" and intent != "forbidden":
            inter_c[k] = None
            continue
        vals = [(a.get("inter") or {}).get(k) for a in anns]
        max_int = 2 if k == "Ic" else 1
        if all(
            normalize_code(v, allow_null=True, max_int=max_int) is None for v in vals
        ):
            inter_c[k] = None
            continue
        v, st = majority_vote(vals, max_int=max_int)
        if st != "ok":
            inter_c[k] = None
            drop[f"inter.{k}"] = st
        else:
            inter_c[k] = v

    inter_c = apply_inter_dependencies(inter_c, intent)

    for k in ANAT_ITEMS:
        vals = [(a.get("anat") or {}).get(k) for a in anns]
        v, st = majority_vote(vals, max_int=1)
        if st != "ok":
            anat_c[k] = None
            drop[f"anat.{k}"] = st
        else:
            anat_c[k] = v

    scores = construct_scores(inter_c, anat_c, intent=intent)
    return {
        "protocol": PROTOCOL_ID,
        "scheme": SCHEME_VERSION,
        "intent": intent,
        "inter": inter_c,
        "anat": anat_c,
        **scores,
        "Inter_pass": inter_pass_from_items(inter_c, intent),
        "Anat_pass": anat_pass_from_items(anat_c),
        "dropped_items": drop,
        "n_annotators": len(anns),
        "gold_rule": "item_majority_then_construct_S",
    }


def load_thresholds(path: Optional[Path] = None) -> Dict[str, Any]:
    thr = dict(DEFAULT_THRESHOLDS)
    if path and Path(path).is_file():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        thr.update(raw)
    return thr


def pair_key(sample_id: str, model_id: str) -> str:
    return f"{sample_id}__{model_id}"


def img_relpath(model_id: str, sample_id: str) -> str:
    return f"outputs/{model_id}/{sample_id}.png"


def mesh_rec_path(pack: Path, model_id: str, sample_id: str) -> Path:
    return pack / "judgments" / "mesh_v3" / model_id / f"{sample_id}.json"


def _f(rec: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def hard_score_from_mesh(rec: Optional[dict]) -> Tuple[float, List[str]]:
    if not rec or not rec.get("ok"):
        return 0.0, []
    tags: List[str] = []
    score = 0.0
    p_fuse = _f(rec, "P_fuse")
    if p_fuse is not None and float(p_fuse) >= 0.30:
        score += float(p_fuse)
        tags.append("high_P_fuse")
    p_extra = _f(rec, "P_anat_extra", "P_extra")
    if p_extra is not None and float(p_extra) >= 0.30:
        score += float(p_extra)
        tags.append("high_P_extra")
    p_resid = _f(rec, "P_anat_resid")
    if p_resid is not None and float(p_resid) >= 0.18:
        score += float(p_resid)
        tags.append("high_P_resid")
    s_inter = _f(rec, "S_inter_mesh", "S_inter")
    if s_inter is not None and float(s_inter) <= 0.70:
        score += 1.0 - float(s_inter)
        tags.append("low_S_inter")
    s_anat = _f(rec, "S_anat_mesh", "S_anat")
    if s_anat is not None and float(s_anat) <= 0.82:
        score += 0.5 * (1.0 - float(s_anat))
        tags.append("low_S_anat")
    return float(score), tags


def map_mesh_to_checklist(
    rec: dict,
    *,
    thresholds: Optional[dict] = None,
    calib: Optional[dict] = None,
) -> dict:
    """mesh_v3 → Checklist_M (v4 construct mapping). Ir/A5 geometry blind spots → null."""
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    cal = calib or {}
    vol_ok = float(cal.get("vol_ok", thr.get("vol_ok", 0.05)))
    d_good = float(cal.get("d_good", thr.get("d_good", 0.05)))

    intent = rec.get("contact_intent")
    if not intent:
        try:
            from mesh_metrics import prompt_contact_intent

            intent = prompt_contact_intent(rec.get("prompt"))
        except Exception:
            intent = "unspecified"
    intent = str(intent)

    under = bool(rec.get("under_detect"))
    n_hum = rec.get("n_humans")
    n_exp = rec.get("n_expected")
    if n_exp is None:
        n_exp = n_expected_from_prompt(rec)

    if rec.get("recon_fail") or not rec.get("ok", True):
        i0 = 0
    else:
        i0 = int((not under) and (n_hum is not None) and (int(n_hum) == int(n_exp)))

    p_fuse = _f(rec, "P_fuse")
    pen_vol = _f(rec, "pen_volume_m3")
    if p_fuse is None and pen_vol is None:
        i1 = 0 if (rec.get("recon_fail") or not rec.get("ok", True)) else 1
    else:
        ok_fuse = (p_fuse is not None and float(p_fuse) < float(thr["tau_fuse"])) or (
            pen_vol is not None and float(pen_vol) <= vol_ok
        )
        i1 = int(bool(ok_fuse))

    p_miss = _f(rec, "P_miss")
    min_dist = _f(rec, "min_surf_dist")
    ic = i3 = ir = None
    if intent == "required":
        touch = p_miss is not None and float(p_miss) < float(thr["tau_miss_touch"])
        close_by_miss = p_miss is not None and float(p_miss) < float(thr["tau_miss_close"])
        dist_lim = float(thr.get("tau_dist_close", max(0.08, d_good * 1.6)))
        close_by_dist = (
            min_dist is not None
            and float(min_dist) == float(min_dist)
            and float(min_dist) <= dist_lim
        )
        if close_by_miss or close_by_dist:
            ic = 2
        elif touch:
            ic = 1
        else:
            ic = 0
        ir = None  # geometry blind spot: do not map region
    elif intent == "forbidden":
        p_unw = _f(rec, "P_unwanted")
        i3 = int(p_unw is not None and float(p_unw) < float(thr["tau_unw"]))

    inter = {"I0": i0, "I1": i1, "Ic": ic, "I3": i3, "Ir": ir}
    inter = apply_inter_dependencies(inter, intent)

    p_extra = _f(rec, "P_anat_extra", "P_extra")
    orphan = _f(rec, "anat_orphan_frac", "orphan_frac")
    leftover = _f(rec, "anat_leftover_frac", "leftover_frac")
    if p_extra is None and orphan is None and leftover is None:
        a1 = 1
    else:
        by_tau = p_extra is not None and float(p_extra) < float(thr["tau_extra"])
        by_raw = (
            orphan is not None
            and leftover is not None
            and float(orphan) < float(thr["orphan_ok"])
            and float(leftover) < float(thr["leftover_ok"])
        )
        a1 = int(by_tau or by_raw)

    p_resid = _f(rec, "P_anat_resid")
    s_anat = _f(rec, "S_anat_mesh", "S_anat")
    if p_resid is not None:
        a2 = int(float(p_resid) < float(thr["tau_resid"]))
    elif s_anat is not None:
        a2 = int(float(s_anat) >= float(thr["tau_S_anat"]))
    else:
        a2 = 1

    s_own = _f(rec, "S_anat_ownership")
    a3 = 1 if s_own is None else int(float(s_own) >= float(thr["tau_own"]))
    s_scale = _f(rec, "S_anat_scale")
    a4 = 1 if s_scale is None else int(float(s_scale) >= float(thr["tau_scale"]))

    anat = {
        "A1": a1,
        "A2": a2,
        "A3": a3,
        "A4": a4,
        "A5": None,  # hands: geometry blind spot
    }

    s_inter = _f(rec, "S_inter_mesh", "S_inter")
    # M-side construct: Ir/A5 are geometry blind spots; use mappable subset
    s_inter_m_geom = _inter_construct_geom_only(inter, intent)
    s_anat_m_geom = _anat_construct_geom_only(anat)

    return {
        "sample_id": rec.get("sample_id"),
        "model_id": rec.get("model_id"),
        "protocol": MESH_BIN_PROTOCOL,
        "checklist_protocol": PROTOCOL_ID,
        "thresholds_ref": "_thresholds.json",
        "thresholds_version": thr.get("version", THRESHOLDS_MAP_VERSION),
        "intent": intent,
        "inter": inter,
        "anat": anat,
        "analysis_protocol": ANALYSIS_PROTOCOL_ID,
        "scheme": SCHEME_VERSION,
        "S_inter_family": inter_family(intent),
        "S_inter_M": s_inter_m_geom,
        "S_anat_M": s_anat_m_geom,
        "S_inter_mesh_raw": s_inter,
        "S_anat_mesh_raw": s_anat,
        "Inter_pass": _inter_pass_geom(inter, intent),
        "Anat_pass_geom": anat_pass_from_items({**anat, "A5": 1}),
        "blind_items": ["Ir", "A5"],
        "report_blocks_required": ["geom_H_M", "full_construct_IAA", "blind_H_only"],
        "raw_refs": {
            "P_fuse": p_fuse,
            "P_miss": p_miss,
            "P_unwanted": _f(rec, "P_unwanted"),
            "P_anat_extra": p_extra,
            "P_anat_resid": p_resid,
            "anat_orphan_frac": orphan,
            "anat_leftover_frac": leftover,
            "S_anat_ownership": s_own,
            "S_anat_scale": s_scale,
            "S_inter_mesh": s_inter,
            "S_anat_mesh": s_anat,
            "pen_volume_m3": pen_vol,
            "min_surf_dist": min_dist,
            "n_humans": n_hum,
            "n_expected": n_exp,
            "under_detect": under,
            "vol_ok": vol_ok,
            "d_good": d_good,
        },
    }


def _inter_construct_geom_only(inter: dict, intent: str) -> Optional[float]:
    """Mappable M subset: required uses I0/I1/Ic (excludes Ir)."""
    intent = (intent or "unspecified").lower()
    parts: List[float] = []
    for k in ("I0", "I1"):
        v = normalize_inter_item(k, inter.get(k), intent=intent)
        if v is None or v == "U":
            return None
        parts.append(float(int(v)))
    if intent == "required":
        ic = normalize_inter_item("Ic", inter.get("Ic"), intent=intent)
        if ic is None or ic == "U":
            return None
        parts.append(float(int(ic)) / 2.0)
    elif intent == "forbidden":
        v = normalize_inter_item("I3", inter.get("I3"), intent=intent)
        if v is None or v == "U":
            return None
        parts.append(float(int(v)))
    return float(sum(parts) / len(parts)) if parts else None


def _anat_construct_geom_only(anat: dict) -> Optional[float]:
    parts: List[float] = []
    for k in ("A1", "A2", "A3", "A4"):
        v = normalize_code(anat.get(k), allow_u=True, allow_null=True, max_int=1)
        if v is None or v == "U":
            return None
        parts.append(float(int(v)))
    return float(sum(parts) / len(parts)) if parts else None


def _inter_pass_geom(inter: dict, intent: str) -> Optional[int]:
    """When required and Ir is blind: I0 ∧ I1 ∧ Ic==2."""
    intent = (intent or "unspecified").lower()
    for k in ("I0", "I1"):
        v = normalize_inter_item(k, inter.get(k), intent=intent)
        if v is None or v == "U":
            return None
        if int(v) != 1:
            return 0
    if intent == "required":
        ic = normalize_inter_item("Ic", inter.get("Ic"), intent=intent)
        if ic is None or ic == "U":
            return None
        return int(int(ic) >= 2)
    if intent == "forbidden":
        v = normalize_inter_item("I3", inter.get("I3"), intent=intent)
        if v is None or v == "U":
            return None
        return int(int(v) == 1)
    return 1


def validate_checklist_payload(obj: dict, *, allow_u: bool = True) -> List[str]:
    errs: List[str] = []
    inter = obj.get("inter") if isinstance(obj.get("inter"), dict) else {}
    anat = obj.get("anat") if isinstance(obj.get("anat"), dict) else {}
    intent = str(obj.get("intent_used") or obj.get("intent") or "unspecified")
    if not isinstance(obj.get("inter"), dict):
        errs.append("missing inter")
    if not isinstance(obj.get("anat"), dict):
        errs.append("missing anat")
    for k in INTER_ITEMS:
        if k not in inter:
            errs.append(f"missing inter.{k}")
            continue
        try:
            normalize_inter_item(k, inter[k], intent=intent)
        except ValueError as e:
            errs.append(f"inter.{k}: {e}")
    for k in ANAT_ITEMS:
        if k not in anat:
            errs.append(f"missing anat.{k}")
            continue
        try:
            normalize_code(anat[k], allow_u=allow_u, allow_null=True, max_int=1)
        except ValueError as e:
            errs.append(f"anat.{k}: {e}")
    return errs
