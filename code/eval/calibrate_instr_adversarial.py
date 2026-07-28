#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Instr Go/No-Go adversarial calibration (mpie_hard_v2.1).

Acceptance (docs/eval_instr_vqa_redesign.md §4.5):
  Role-swap / drop-prop adversarial should drop the score by ≥ DROP_MIN (default 0.3).

Modes (no new image synthesis required for bank probes):
  1) bank_swap_probe   — perfect yes vs flip all swap_sensitive → no
  2) bank_drop_prop    — perfect yes vs flip prop_object → no
  3) bank_asymm_flip   — perfect yes vs flip asymm subtype → no
  4) judgment_asymm_flip — on real model judgments, zero-out asymm answers → Δ
  5) cross_model_gap   — closed vs weakest open on S_instr_asymm (same samples)

Usage:
  python calibrate_instr_adversarial.py --pack "$MPIE_TEST_PACK"
  python calibrate_instr_adversarial.py --pack ... --out .../instr_adv_calibration.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from instr_qa_common import REVISION, MAIN_SUBTYPES, score_answers
from pack_io import pack_root

DROP_MIN = 0.30
PERFECT_THR = 0.999


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def load_frozen_bank(pack: Path) -> dict[str, dict]:
    d = pack / "instr_qa_v2"
    out = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = j.get("_meta") if isinstance(j.get("_meta"), dict) else {}
        rev = j.get("revision") or meta.get("revision")
        if rev and rev != REVISION:
            continue
        qs = j.get("questions") or []
        if len(qs) < 2:
            continue
        out[j.get("sample_id") or p.stem] = j
    return out


def load_judgments(pack: Path, model_id: str) -> dict[str, dict]:
    d = pack / "judgments" / "instr_v2" / model_id
    out = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if j.get("ok") is False or j.get("S_instr") is None:
            continue
        if j.get("confidence") == "low":
            continue
        out[j.get("sample_id") or p.stem] = j
    return out


def _perfect_answers(questions: list[dict]) -> dict[str, str]:
    return {q["id"]: "yes" for q in questions if q.get("id")}


def _flip(
    questions: list[dict],
    answers: dict[str, str],
    *,
    pred,
) -> dict[str, str]:
    out = dict(answers)
    for q in questions:
        qid = q.get("id")
        if not qid:
            continue
        if pred(q):
            out[qid] = "no"
    return out


def probe_bank(bank: dict[str, dict], *, drop_min: float = DROP_MIN) -> dict[str, Any]:
    """Text-only probes: perfect vs adversarial answer flips."""
    rows = []
    for sid, fr in bank.items():
        qs = fr.get("questions") or []
        perfect = _perfect_answers(qs)
        s0 = score_answers(qs, perfect)
        s_swap = score_answers(
            qs,
            _flip(qs, perfect, pred=lambda q: bool(q.get("swap_sensitive"))),
        )
        s_asymm = score_answers(
            qs,
            _flip(
                qs,
                perfect,
                pred=lambda q: (q.get("subtype") == "asymm" or q.get("bucket") == "asymm"),
            ),
        )
        s_prop = score_answers(
            qs,
            _flip(qs, perfect, pred=lambda q: q.get("subtype") == "prop_object"),
        )
        n_swap = sum(1 for q in qs if q.get("swap_sensitive"))
        n_asymm = sum(
            1 for q in qs if q.get("subtype") == "asymm" or q.get("bucket") == "asymm"
        )
        n_prop = sum(1 for q in qs if q.get("subtype") == "prop_object")
        row = {
            "sample_id": sid,
            "n_q": len(qs),
            "n_swap_sensitive": n_swap,
            "n_asymm": n_asymm,
            "n_prop_object": n_prop,
            "S_perfect": s0["S_instr"],
            "S_asymm_perfect": s0["S_instr_asymm"],
            "S_after_swap_flip": s_swap["S_instr"],
            "S_asymm_after_swap_flip": s_swap["S_instr_asymm"],
            "S_after_asymm_flip": s_asymm["S_instr"],
            "S_asymm_after_asymm_flip": s_asymm["S_instr_asymm"],
            "S_after_prop_flip": s_prop["S_instr"],
            "drop_swap": _drop(s0["S_instr"], s_swap["S_instr"]),
            "drop_asymm": _drop(s0["S_instr"], s_asymm["S_instr"]),
            "drop_asymm_on_asymm": _drop(s0["S_instr_asymm"], s_asymm["S_instr_asymm"]),
            "drop_prop": _drop(s0["S_instr"], s_prop["S_instr"]),
        }
        rows.append(row)

    def _agg(key: str) -> float | None:
        return _mean([r[key] for r in rows if r.get(key) is not None])

    n = len(rows)
    go_swap = [
        r for r in rows if (r.get("drop_swap") is not None and r["drop_swap"] >= drop_min)
    ]
    go_asymm = [
        r
        for r in rows
        if (r.get("drop_asymm_on_asymm") is not None and r["drop_asymm_on_asymm"] >= drop_min)
    ]
    # prop only meaningful when prop_object exists
    prop_rows = [r for r in rows if r["n_prop_object"] > 0]
    go_prop = [
        r
        for r in prop_rows
        if (r.get("drop_prop") is not None and r["drop_prop"] >= drop_min)
    ]

    return {
        "n_samples": n,
        "drop_min": drop_min,
        "mean_drop_swap": _agg("drop_swap"),
        "mean_drop_asymm": _agg("drop_asymm"),
        "mean_drop_asymm_on_asymm": _agg("drop_asymm_on_asymm"),
        "mean_drop_prop": _mean([r["drop_prop"] for r in prop_rows if r.get("drop_prop") is not None]),
        "frac_pass_swap": len(go_swap) / n if n else None,
        "frac_pass_asymm_flip": len(go_asymm) / n if n else None,
        "frac_pass_prop": (len(go_prop) / len(prop_rows)) if prop_rows else None,
        "n_with_prop_object": len(prop_rows),
        "go_swap": (len(go_swap) / n >= 0.90) if n else False,
        "go_asymm": (len(go_asymm) / n >= 0.90) if n else False,
        "go_prop": (len(go_prop) / len(prop_rows) >= 0.80) if prop_rows else None,
        "fail_swap_examples": [
            r["sample_id"]
            for r in sorted(rows, key=lambda x: (x.get("drop_swap") or 0))[:8]
        ],
        "per_sample": rows,
    }


def _drop(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def judgment_asymm_sensitivity(
    pack: Path, model_ids: list[str], *, drop_min: float = DROP_MIN
) -> dict[str, Any]:
    """On real judgments: re-score with asymm answers forced to no → expected drop."""
    bank = load_frozen_bank(pack)
    by_model = {}
    for mid in model_ids:
        js = load_judgments(pack, mid)
        drops_s, drops_a = [], []
        n = 0
        for sid, j in js.items():
            fr = bank.get(sid)
            if not fr:
                continue
            qs = fr.get("questions") or []
            ans = {
                d["id"]: d["a"]
                for d in (j.get("detail") or [])
                if d.get("id") and d.get("a")
            }
            if not ans:
                continue
            s0 = score_answers(qs, ans)
            s1 = score_answers(
                qs,
                _flip(
                    qs,
                    ans,
                    pred=lambda q: (
                        q.get("subtype") == "asymm" or q.get("bucket") == "asymm"
                    ),
                ),
            )
            d_s = _drop(s0["S_instr"], s1["S_instr"])
            d_a = _drop(s0["S_instr_asymm"], s1["S_instr_asymm"])
            if d_s is not None:
                drops_s.append(d_s)
            if d_a is not None:
                drops_a.append(d_a)
            n += 1
        by_model[mid] = {
            "n": n,
            "mean_drop_S_instr": _mean(drops_s),
            "mean_drop_S_instr_asymm": _mean(drops_a),
            "frac_drop_ge_min": (
                sum(1 for d in drops_a if d >= drop_min) / len(drops_a)
                if drops_a
                else None
            ),
        }
    return {"drop_min": drop_min, "by_model": by_model}


def cross_model_asymm_gap(pack: Path, *, drop_min: float = DROP_MIN) -> dict[str, Any]:
    """Closed vs weakest open on S_instr_asymm (paired samples)."""
    closed = ["gpt-image-2", "gemini-3-pro-image", "seedream-5-pro"]
    open_m = ["flux1-kontext-dev", "qwen-image-edit-2511"]
    jud = {m: load_judgments(pack, m) for m in closed + open_m}
    weak = "qwen-image-edit-2511"
    gaps = []
    for sid in jud[weak]:
        w = jud[weak][sid].get("S_instr_asymm")
        if w is None:
            continue
        closed_vals = []
        for m in closed:
            if sid in jud[m] and jud[m][sid].get("S_instr_asymm") is not None:
                closed_vals.append(float(jud[m][sid]["S_instr_asymm"]))
        if not closed_vals:
            continue
        gaps.append(max(closed_vals) - float(w))
    return {
        "weak_model": weak,
        "n_paired": len(gaps),
        "mean_gap_closed_max_vs_qwen_asymm": _mean(gaps),
        "frac_gap_ge_min": (
            sum(1 for g in gaps if g >= drop_min) / len(gaps) if gaps else None
        ),
        "go_cross_model": (
            (_mean(gaps) or 0) >= drop_min
            and (sum(1 for g in gaps if g >= drop_min) / len(gaps) >= 0.5)
            if gaps
            else False
        ),
    }


def model_asymm_table(pack: Path) -> dict[str, Any]:
    """Per-model Instr-asymm + p_perfect (on asymm) for main-table preview."""
    root = pack / "judgments" / "instr_v2"
    out = {}
    if not root.is_dir():
        return out
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        asymm, full = [], []
        for p in sub.glob("*.json"):
            if p.name.startswith("_"):
                continue
            try:
                j = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if j.get("confidence") == "low" or j.get("S_instr") is None:
                continue
            full.append(float(j["S_instr"]))
            if j.get("S_instr_asymm") is not None:
                asymm.append(float(j["S_instr_asymm"]))
        if not full:
            continue
        out[sub.name] = {
            "n": len(full),
            "S_instr": _mean(full),
            "S_instr_asymm": _mean(asymm),
            "p_perfect": sum(1 for s in full if s >= PERFECT_THR) / len(full),
            "p_perfect_asymm": (
                sum(1 for s in asymm if s >= PERFECT_THR) / len(asymm) if asymm else None
            ),
        }
    return out


def decide_go(
    bank_probe: dict, jud_sens: dict, cross: dict, *, drop_min: float = DROP_MIN
) -> dict[str, Any]:
    """Overall Go/No-Go for Instr adversarial hardness."""
    checks = {
        "bank_swap_drop_ge_0.3_on_90pct": bool(bank_probe.get("go_swap")),
        "bank_asymm_flip_drop_ge_0.3_on_90pct": bool(bank_probe.get("go_asymm")),
        "cross_model_asymm_gap": bool(cross.get("go_cross_model")),
    }
    # judgment sensitivity: at least one closed model mean asymm drop ≥ drop_min
    jud_ok = False
    for mid, row in (jud_sens.get("by_model") or {}).items():
        if (row.get("mean_drop_S_instr_asymm") or 0) >= drop_min:
            jud_ok = True
            break
    checks["judgment_asymm_flip_mean_ge_0.3"] = jud_ok

    passed = sum(1 for v in checks.values() if v)
    overall = "GO" if passed >= 3 else ("SOFT_GO" if passed >= 2 else "NO_GO")
    return {
        "overall": overall,
        "passed": passed,
        "total": len(checks),
        "checks": checks,
        "criterion": (
            f"Need ≥3/4 checks. Bank probes: ≥90% samples drop≥{drop_min} when "
            "swap_sensitive/asymm answers flip to no. Cross-model: mean closed−qwen "
            f"asymm gap≥{drop_min}. Judgment: ≥1 model mean asymm drop≥{drop_min}."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Instr adversarial Go/No-Go calibration")
    ap.add_argument("--pack", default="")
    ap.add_argument("--out", default="", help="JSON report path")
    ap.add_argument("--drop-min", type=float, default=DROP_MIN)
    args = ap.parse_args()
    drop_min = float(args.drop_min)

    pack = pack_root(args.pack or None)
    bank = load_frozen_bank(pack)
    if not bank:
        raise SystemExit(f"ERROR: no frozen QA bank under {pack}/instr_qa_v2")

    models = []
    root = pack / "judgments" / "instr_v2"
    if root.is_dir():
        models = sorted(
            p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
        )

    bank_probe = probe_bank(bank, drop_min=drop_min)
    # strip bulky per_sample from console; keep in file
    jud_sens = judgment_asymm_sensitivity(pack, models, drop_min=drop_min)
    cross = cross_model_asymm_gap(pack, drop_min=drop_min)
    table = model_asymm_table(pack)
    go = decide_go(bank_probe, jud_sens, cross, drop_min=drop_min)

    report = {
        "protocol": "instr_adv_calibration_v1",
        "revision": REVISION,
        "pack": str(pack),
        "drop_min": drop_min,
        "main_subtypes": list(MAIN_SUBTYPES),
        "go_nogo": go,
        "bank_probes": {k: v for k, v in bank_probe.items() if k != "per_sample"},
        "bank_probes_per_sample_n": bank_probe.get("n_samples"),
        "judgment_asymm_sensitivity": jud_sens,
        "cross_model_gap": cross,
        "model_table_preview": table,
        "note": (
            "Bank probes are answer-space adversarial (no new images). "
            "They validate that the frozen QA bank *can* punish role-swap / asymm failure. "
            "Image-level role-swap synth remains optional follow-up."
        ),
    }

    # keep fails list short in bank_probes already; attach full per_sample only to disk
    out_path = Path(args.out) if args.out else (
        pack / "judgments" / "instr_v2" / "_adv_calibration.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full = dict(report)
    full["bank_probes_per_sample"] = bank_probe.get("per_sample")
    out_path.write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")

    # also publish under eval_outputs for the evaluation dashboard
    pub = (
        Path("data") / "eval_outputs"
        / "smoke100_v3"
        / "instr_adv_calibration.json"
    )
    try:
        pub.parent.mkdir(parents=True, exist_ok=True)
        pub.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pub = None

    slim = dict(report)
    print(json.dumps(slim, ensure_ascii=False, indent=2))
    print(f"\nwrote {out_path}", flush=True)
    if pub:
        print(f"wrote {pub}", flush=True)


if __name__ == "__main__":
    main()
