#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stratified sampling into human_consistency/_split.json (discrimination-first, split v2).

Protocol: eval_human_consistency_analysis_protocol.md §9b
Unit = (sample_id, model_id); requires outputs/<model>/<sample>.png to exist.

Hard constraints (defaults):
  - hard_frac ≥ 0.50 (prefer mesh hard_score; else VLM-v1 failure score)
  - weaker open models (flux+qwen) ≥ 60% combined (via --model-mix)
  - any single closed-source model ≤ ~15%

Usage:
  python select_consistency_split.py --pack "$MPIE_TEST_PACK" --force
  python select_consistency_split.py --pack ... --hard-frac 0.55 --guide-n 20 --pilot-n 100 --holdout-n 50
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import (  # noqa: E402
    DEFAULT_MODEL_MIX,
    DEFAULT_MODELS,
    PROTOCOL_ID,
    SCHEME_VERSION,
    atomic_write_json,
    hard_score_from_mesh,
    img_relpath,
    mesh_rec_path,
    n_expected_from_prompt,
    pair_key,
)
from mesh_metrics import prompt_contact_intent  # noqa: E402
from pack_io import load_manifest, pack_root  # noqa: E402

INTENT_ORDER = ("required", "forbidden", "unspecified")
WEAK_MODELS = ("flux1-kontext-dev", "dreamo")


def _parse_models(s: str) -> List[str]:
    if not s.strip():
        return list(DEFAULT_MODELS)
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_model_mix(s: str) -> Dict[str, float]:
    if not s.strip():
        return dict(DEFAULT_MODEL_MIX)
    out: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(f"bad --model-mix entry {part!r}; want model:weight")
        mid, w = part.split(":", 1)
        out[mid.strip()] = float(w)
    tot = sum(out.values())
    if tot <= 0:
        raise SystemExit("model-mix weights sum to 0")
    return {k: v / tot for k, v in out.items()}


def _discover_models(pack: Path, preferred: Sequence[str]) -> List[str]:
    out = pack / "outputs"
    if not out.is_dir():
        return list(preferred)
    have = {p.name for p in out.iterdir() if p.is_dir() and not p.name.startswith("_")}
    chosen = [m for m in preferred if m in have]
    if not chosen:
        chosen = sorted(have)
    return chosen


def hard_score_from_vlm(obj: Optional[dict]) -> Tuple[float, List[str]]:
    """Legacy six-axis VLM failure signal → hardness score (when mesh is absent)."""
    if not obj:
        return 0.0, []
    score = 0.0
    tags: List[str] = []
    if obj.get("anat_pass") is False:
        errs = obj.get("anat_errors") or []
        score += 1.0 + 0.25 * min(4, len(errs) if isinstance(errs, list) else 0)
        tags.append("vlm_anat_fail")
        for e in (errs if isinstance(errs, list) else [])[:3]:
            if isinstance(e, dict) and e.get("type"):
                tags.append(f"err:{e['type']}")
    inter = obj.get("inter") or {}
    if isinstance(inter, dict):
        pen = str(inter.get("no_pathological_penetration") or "").lower()
        if pen in ("no", "false", "0", "fail"):
            score += 1.2
            tags.append("vlm_pen_fail")
        contact = str(inter.get("contact_points") or inter.get("contact") or "").lower()
        if contact in ("no", "fail", "missing", "none"):
            score += 0.8
            tags.append("vlm_contact_fail")
        elif contact in ("partial",):
            score += 0.4
            tags.append("vlm_contact_partial")
        sem = str(inter.get("semantic") or "").lower()
        if sem in ("no", "fail"):
            score += 0.5
            tags.append("vlm_sem_fail")
        elif sem in ("partial",):
            score += 0.25
            tags.append("vlm_sem_partial")
    if obj.get("count_pass") is False:
        score += 0.5
        tags.append("vlm_count_fail")
    if obj.get("qual_pass") is False:
        score += 0.3
        tags.append("vlm_qual_fail")
    return float(score), tags


def build_candidates(
    pack: Path,
    rows: List[dict],
    models: Sequence[str],
    *,
    use_mesh: bool,
    use_vlm_hard: bool,
) -> Tuple[List[dict], int, int]:
    cands: List[dict] = []
    mesh_hits = 0
    vlm_hits = 0
    for row in rows:
        sid = row["sample_id"]
        intent_prompt = prompt_contact_intent(row.get("prompt"))
        n_exp = n_expected_from_prompt(row)
        for mid in models:
            rel = img_relpath(mid, sid)
            if not (pack / rel).is_file():
                # try jpg
                alt = f"outputs/{mid}/{sid}.jpg"
                if (pack / alt).is_file():
                    rel = alt
                else:
                    continue
            rec = None
            hard_s, hard_tags = 0.0, []
            intent_m = intent_prompt
            has_mesh = False
            if use_mesh:
                mp = mesh_rec_path(pack, mid, sid)
                if mp.is_file():
                    try:
                        rec = json.loads(mp.read_text(encoding="utf-8"))
                        mesh_hits += 1
                        has_mesh = True
                        hard_s, hard_tags = hard_score_from_mesh(rec)
                        if rec.get("contact_intent"):
                            intent_m = rec["contact_intent"]
                    except Exception:
                        rec = None
            if hard_s <= 0 and use_vlm_hard:
                vp = pack / "judgments" / "vlm_judge_v1" / mid / f"{sid}.json"
                if vp.is_file():
                    try:
                        vobj = json.loads(vp.read_text(encoding="utf-8"))
                        vlm_hits += 1
                        vs, vt = hard_score_from_vlm(vobj)
                        if vs > hard_s:
                            hard_s, hard_tags = vs, vt
                    except Exception:
                        pass
            cands.append(
                {
                    "sample_id": sid,
                    "model_id": mid,
                    "key": pair_key(sid, mid),
                    "cat": row.get("cat"),
                    "intent": intent_m,
                    "n_expected": n_exp,
                    "prompt": row.get("prompt") or "",
                    "img_relpath": rel,
                    "hard_score": hard_s,
                    "hard_tags": hard_tags,
                    "has_mesh": has_mesh,
                }
            )
    return cands, mesh_hits, vlm_hits


def _renorm_intent_weights(pool: Sequence[dict], weights: Dict[str, float]) -> Dict[str, float]:
    present = {c["intent"] for c in pool}
    w = {k: float(weights.get(k, 0.0)) for k in INTENT_ORDER}
    for k in INTENT_ORDER:
        if k not in present:
            w[k] = 0.0
    s = sum(w.values())
    if s <= 0:
        n = max(1, len(present))
        return {k: (1.0 / n if k in present else 0.0) for k in INTENT_ORDER}
    return {k: v / s for k, v in w.items()}


def _quota(n: int, weights: Dict[str, float], keys: Sequence[str]) -> Dict[str, int]:
    raw = {k: weights.get(k, 0.0) * n for k in keys}
    base = {k: int(raw[k]) for k in keys}
    rem = n - sum(base.values())
    frac = sorted(keys, key=lambda k: -(raw[k] - base[k]))
    for k in frac:
        if rem <= 0:
            break
        base[k] += 1
        rem -= 1
    return base


def _take_intent(
    src: List[dict], want: int, iq: Dict[str, int], rng: random.Random
) -> List[dict]:
    picked: List[dict] = []
    by_i: Dict[str, List[dict]] = defaultdict(list)
    for c in src:
        by_i[c["intent"]].append(c)
    for k in INTENT_ORDER:
        rng.shuffle(by_i[k])
    for k in INTENT_ORDER:
        need = min(iq.get(k, 0), want - len(picked), len(by_i[k]))
        for _ in range(need):
            picked.append(by_i[k].pop())
    if len(picked) < want:
        rest: List[dict] = []
        for k in INTENT_ORDER:
            rest.extend(by_i[k])
        rng.shuffle(rest)
        for c in rest:
            if len(picked) >= want:
                break
            picked.append(c)
    return picked[:want]


def stratified_take_model_mix(
    pool: List[dict],
    n: int,
    rng: random.Random,
    *,
    intent_w: Dict[str, float],
    hard_frac: float,
    model_mix: Dict[str, float],
) -> List[dict]:
    """Sample n units by model mix; within each model, hard_frac hard cases + intent strata."""
    if n <= 0 or not pool:
        return []
    n = min(n, len(pool))
    by_m: Dict[str, List[dict]] = defaultdict(list)
    for c in pool:
        by_m[c["model_id"]].append(c)

    # Keep only models present in the pool, then renormalize
    mix = {m: w for m, w in model_mix.items() if m in by_m and by_m[m]}
    if not mix:
        # fallback: equal weight over available models
        ms = list(by_m.keys())
        mix = {m: 1.0 / len(ms) for m in ms}
    else:
        s = sum(mix.values())
        mix = {m: w / s for m, w in mix.items()}

    mq = _quota(n, mix, list(mix.keys()))
    # Redistribute when a model cannot fill its quota
    for m in list(mq.keys()):
        mq[m] = min(mq[m], len(by_m[m]))
    short = n - sum(mq.values())
    if short > 0:
        rich = sorted(by_m.keys(), key=lambda m: len(by_m[m]) - mq.get(m, 0), reverse=True)
        for m in rich:
            if short <= 0:
                break
            room = len(by_m[m]) - mq.get(m, 0)
            add = min(room, short)
            mq[m] = mq.get(m, 0) + add
            short -= add

    out: List[dict] = []
    for mid, want in mq.items():
        if want <= 0:
            continue
        src = list(by_m[mid])
        w = _renorm_intent_weights(src, intent_w)
        hard_pool = [c for c in src if float(c.get("hard_score") or 0) > 0]
        n_hard = min(len(hard_pool), int(round(want * hard_frac))) if hard_frac > 0 else 0
        n_rand = want - n_hard
        hq = _quota(n_hard, w, INTENT_ORDER) if n_hard else {k: 0 for k in INTENT_ORDER}
        rq = _quota(n_rand, w, INTENT_ORDER) if n_rand else {k: 0 for k in INTENT_ORDER}
        picked_hard = _take_intent(hard_pool, n_hard, hq, rng) if n_hard else []
        hard_keys = {c["key"] for c in picked_hard}
        remain = [c for c in src if c["key"] not in hard_keys]
        soft = [c for c in remain if float(c.get("hard_score") or 0) <= 0]
        if len(soft) < n_rand:
            soft = remain
        picked_rand = _take_intent(soft, n_rand, rq, rng) if n_rand else []
        for c in picked_hard:
            d = dict(c)
            d["is_hard_slot"] = True
            out.append(d)
        for c in picked_rand:
            d = dict(c)
            d["is_hard_slot"] = False
            out.append(d)

    if len(out) < n:
        have = {c["key"] for c in out}
        rest = [c for c in pool if c["key"] not in have]
        rng.shuffle(rest)
        for c in rest[: n - len(out)]:
            d = dict(c)
            d["is_hard_slot"] = float(d.get("hard_score") or 0) > 0
            out.append(d)

    drop = {c["key"] for c in out}
    pool[:] = [c for c in pool if c["key"] not in drop]
    rng.shuffle(out)
    return out[:n]


def diverse_guide(
    pool: List[dict],
    n: int,
    rng: random.Random,
    *,
    hard_frac: float = 0.55,
    model_mix: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """Guide split: model mix + hard oversampling + category/intent diversity."""
    if n <= 0 or not pool:
        return []
    # First sample a large pool by mix+hardness, then diversify by buckets
    picked = stratified_take_model_mix(
        pool,
        n,
        rng,
        intent_w={"required": 0.7, "forbidden": 0.1, "unspecified": 0.2},
        hard_frac=hard_frac,
        model_mix=model_mix or DEFAULT_MODEL_MIX,
    )
    # If swaps remain, prefer a different category
    return picked


def _slim(c: dict) -> dict:
    return {
        "sample_id": c["sample_id"],
        "model_id": c["model_id"],
        "key": c["key"],
        "cat": c.get("cat"),
        "intent": c["intent"],
        "n_expected": c.get("n_expected"),
        "img_relpath": c["img_relpath"],
        "hard_score": round(float(c.get("hard_score") or 0.0), 4),
        "hard_tags": list(c.get("hard_tags") or []),
        "is_hard_slot": bool(c.get("is_hard_slot")),
        "has_mesh": bool(c.get("has_mesh")),
        "prompt": c.get("prompt") or "",
    }


def _stats(items: Sequence[dict]) -> dict:
    by_m = dict(Counter(c["model_id"] for c in items))
    n = max(1, len(items))
    weak = sum(by_m.get(m, 0) for m in WEAK_MODELS)
    return {
        "n": len(items),
        "by_intent": dict(Counter(c["intent"] for c in items)),
        "by_model": by_m,
        "by_cat": dict(Counter(c.get("cat") or "?" for c in items)),
        "n_hard_slot": sum(1 for c in items if c.get("is_hard_slot")),
        "hard_frac_realized": round(
            sum(1 for c in items if c.get("is_hard_slot")) / n, 3
        ),
        "weak_model_frac": round(weak / n, 3),
        "n_has_mesh": sum(1 for c in items if c.get("has_mesh")),
        "mesh_frac": round(sum(1 for c in items if c.get("has_mesh")) / n, 3),
    }


def write_annot_csv(path: Path, items: Sequence[dict], *, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "split",
        "sample_id",
        "model_id",
        "img_relpath",
        "intent",
        "n_expected",
        "cat",
        "prompt",
        "Q_inter",
        "Q_anat",
        "I0",
        "I1",
        "Ic",
        "I3",
        "Ir",
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "seconds",
        "annotator_id",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for c in items:
            row = {k: "" for k in cols}
            row.update(
                {
                    "split": split,
                    "sample_id": c["sample_id"],
                    "model_id": c["model_id"],
                    "img_relpath": c["img_relpath"],
                    "intent": c["intent"],
                    "n_expected": c.get("n_expected"),
                    "cat": c.get("cat"),
                    "prompt": c.get("prompt") or "",
                }
            )
            if c["intent"] != "required":
                row["Ic"] = "null"
                row["Ir"] = "null"
            if c["intent"] != "forbidden":
                row["I3"] = "null"
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack", default="", help="pack root directory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--models",
        default="",
        help="comma-separated; default flux,qwen, plus 3 closed-source models",
    )
    ap.add_argument(
        "--model-mix",
        default="",
        help="model:weight,... default weaker-open-model mix DEFAULT_MODEL_MIX",
    )
    ap.add_argument("--guide-n", type=int, default=20)
    ap.add_argument("--pilot-n", type=int, default=100)
    ap.add_argument("--holdout-n", type=int, default=50)
    ap.add_argument("--main-n", type=int, default=0, help="main-study unit count; 0=skip this round")
    ap.add_argument("--hard-frac", type=float, default=0.55)
    ap.add_argument("--no-mesh-hard", action="store_true")
    ap.add_argument("--no-vlm-hard", action="store_true", help="do not use vlm_judge_v1 hardness signal")
    ap.add_argument(
        "--intent-weights",
        default="0.70,0.10,0.20",
        help="required,forbidden,unspecified (lower target when forbidden pool is small)",
    )
    ap.add_argument("--no-annot-csv", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite existing _split.json")
    ap.add_argument("--archive-old", action="store_true", help="archive previous _split.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    pack = pack_root(args.pack) if args.pack else pack_root()
    rows = load_manifest(pack)
    model_mix = _parse_model_mix(args.model_mix)
    preferred = _parse_models(args.models) or list(model_mix.keys()) or list(DEFAULT_MODELS)
    models = _discover_models(pack, preferred)
    # Keep mix entries only for models that actually exist
    model_mix = {m: model_mix[m] for m in models if m in model_mix}
    if not model_mix:
        model_mix = {m: 1.0 / len(models) for m in models}
    else:
        s = sum(model_mix.values())
        model_mix = {m: w / s for m, w in model_mix.items()}

    mesh_root = pack / "judgments" / "mesh_v3"
    use_mesh = (not args.no_mesh_hard) and mesh_root.is_dir()
    use_vlm = not args.no_vlm_hard
    parts = [float(x) for x in args.intent_weights.split(",")]
    if len(parts) != 3:
        raise SystemExit("--intent-weights need 3 floats")
    intent_w = {
        "required": parts[0],
        "forbidden": parts[1],
        "unspecified": parts[2],
    }

    cands, mesh_hits, vlm_hits = build_candidates(
        pack, rows, models, use_mesh=use_mesh, use_vlm_hard=use_vlm
    )
    prefer_hard = args.hard_frac > 0 and (
        (use_mesh and mesh_hits > 0) or (use_vlm and vlm_hits > 0)
    )
    if not prefer_hard:
        print("[split] WARN: no hardness signal; hard_frac will be ineffective", flush=True)

    pool = list(cands)
    rng.shuffle(pool)

    hc_root = pack / "judgments" / "human_consistency"
    split_path = hc_root / "_split.json"
    if split_path.is_file() and not args.force:
        raise SystemExit(f"exists {split_path}; pass --force to overwrite")
    if split_path.is_file() and args.archive_old:
        ts = time.strftime("%Y%m%d_%H%M%S")
        arch = hc_root / f"_split_archive_{ts}.json"
        arch.write_text(split_path.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[split] archived old -> {arch}", flush=True)

    hard_frac = args.hard_frac if prefer_hard else 0.0
    guide = diverse_guide(
        pool, args.guide_n, rng, hard_frac=hard_frac, model_mix=model_mix
    )
    pilot = stratified_take_model_mix(
        pool,
        args.pilot_n,
        rng,
        intent_w=intent_w,
        hard_frac=hard_frac,
        model_mix=model_mix,
    )
    holdout = stratified_take_model_mix(
        pool,
        args.holdout_n,
        rng,
        intent_w=intent_w,
        hard_frac=hard_frac,
        model_mix=model_mix,
    )
    main_set = stratified_take_model_mix(
        pool,
        args.main_n,
        rng,
        intent_w=intent_w,
        hard_frac=hard_frac,
        model_mix=model_mix,
    )

    splits = {
        "guide": [_slim(c) for c in guide],
        "pilot": [_slim(c) for c in pilot],
        "holdout": [_slim(c) for c in holdout],
        "main": [_slim(c) for c in main_set],
    }

    seen: Set[str] = set()
    for items in splits.values():
        for c in items:
            if c["key"] in seen:
                raise RuntimeError(f"duplicate key across splits: {c['key']}")
            seen.add(c["key"])

    created = time.strftime("%Y-%m-%dT%H:%M:%S")
    all_items = [c for v in splits.values() for c in v]
    payload = {
        "protocol": PROTOCOL_ID,
        "scheme": SCHEME_VERSION,
        "split_design": "v2_discrimination",
        "seed": args.seed,
        "pack": str(pack),
        "created_at": created,
        "models": models,
        "model_mix": model_mix,
        "intent_source": "prompt_keywords",
        "intent_note": "For the formal study, overwrite contact_intent with human gold labels and freeze here",
        "intent_targets": intent_w,
        "hard_frac": hard_frac,
        "hard_source": "mesh_v3+vlm_judge_v1" if use_mesh and use_vlm else (
            "mesh_v3" if use_mesh else ("vlm_judge_v1" if use_vlm else "none")
        ),
        "mesh_records_hit": mesh_hits,
        "vlm_hard_hits": vlm_hits,
        "n_candidates": len(cands),
        "n_remaining_pool": len(pool),
        "discrimination_constraints": {
            "hard_frac_target": 0.50,
            "weak_model_frac_target": 0.60,
            "max_single_closed_frac": 0.15,
            "hm_only_on_has_mesh": True,
        },
        "sizes": {k: len(v) for k, v in splits.items()},
        "stats": {k: _stats(v) for k, v in splits.items()},
        "stats_all": _stats(all_items),
        "splits": splits,
    }
    atomic_write_json(split_path, payload)

    # Do not overwrite a frozen v4.1 _protocol.json; only write a split pointer note
    note_path = hc_root / "_split_design_note.json"
    atomic_write_json(
        note_path,
        {
            "split_ref": "_split.json",
            "split_design": "v2_discrimination",
            "analysis_protocol": "anat_inter_analysis_v4.1 §9b",
            "created_at": created,
            "stats_all": payload["stats_all"],
        },
    )

    if not args.no_annot_csv:
        for name in ("guide", "pilot", "holdout", "main"):
            items = splits[name]
            if not items:
                continue
            write_annot_csv(
                hc_root / "annot_templates" / f"{name}.csv", items, split=name
            )

    sa = payload["stats_all"]
    print(
        json.dumps(
            {
                "wrote": str(split_path),
                "models": models,
                "model_mix": model_mix,
                "n_candidates": len(cands),
                "mesh_hits": mesh_hits,
                "vlm_hard_hits": vlm_hits,
                "hard_source": payload["hard_source"],
                "sizes": payload["sizes"],
                "all_hard_frac": sa["hard_frac_realized"],
                "all_weak_frac": sa["weak_model_frac"],
                "all_mesh_frac": sa["mesh_frac"],
                "all_by_model": sa["by_model"],
                "pilot_models": payload["stats"]["pilot"]["by_model"],
                "pilot_hard_frac": payload["stats"]["pilot"]["hard_frac_realized"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
