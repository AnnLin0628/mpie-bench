#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Human annotation CSV / JSON → human/<ann_id>/ + human/_consensus/(majority vote).

CSV List:sample_id, model_id, intent, I0,I1,Ic,I3,Ir, A1..A5, annotator_id, seconds(Optional)
Also available --from-json-dir Read pressed ann sub-directory JSON。

usage:
  # Three people each fill in one copy CSV(column contains annotator_id）
  python import_human_checklist.py --pack "$MPIE_TEST_PACK" \\
    --csv ann1.csv --csv ann2.csv --csv ann3.csv

  # Or multiple tables annotator_id OK
  python import_human_checklist.py --pack ... --csv pilot_filled.csv --consensus
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import (  # noqa: E402
    ANAT_ITEMS,
    INTER_ITEMS,
    PROTOCOL_ID,
    apply_inter_dependencies,
    atomic_write_json,
    consensus_checklist,
    normalize_code,
    normalize_inter_item,
    pair_key,
)
from pack_io import pack_root  # noqa: E402


def _parse_cell(v: Any, *, max_int: int = 1) -> Any:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in ("null", "none", "na", "n/a", "-"):
        return None
    return normalize_code(s, allow_u=True, allow_null=True, max_int=max_int)


def row_to_record(row: dict) -> dict:
    sid = (row.get("sample_id") or "").strip()
    mid = (row.get("model_id") or "").strip()
    ann = (row.get("annotator_id") or row.get("ann_id") or "").strip()
    if not sid or not mid or not ann:
        raise ValueError(f"need sample_id, model_id, annotator_id; got {row!r}")
    intent = (row.get("intent") or row.get("intent_shown") or "unspecified").strip()
    inter = {
        k: normalize_inter_item(k, row.get(k), intent=intent) for k in INTER_ITEMS
    }
    inter = apply_inter_dependencies(inter, intent)
    from checklist_common import item_max_int  # local to avoid circular surprises

    anat = {
        k: _parse_cell(row.get(k), max_int=item_max_int(k)) for k in ANAT_ITEMS
    }
    sec = row.get("seconds")
    try:
        seconds = float(sec) if sec not in (None, "") else None
    except Exception:
        seconds = None
    return {
        "sample_id": sid,
        "model_id": mid,
        "key": pair_key(sid, mid),
        "annotator_id": ann,
        "protocol": PROTOCOL_ID,
        "intent_shown": intent,
        "inter": inter,
        "anat": anat,
        "seconds": seconds,
        "notes": (row.get("notes") or "")[:500],
        "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def load_csvs(paths: List[Path]) -> List[dict]:
    recs: List[dict] = []
    for p in paths:
        with p.open(encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                # Skip empty label lines
                filled = any(
                    str(row.get(k) or "").strip() not in ("", "null")
                    for k in list(INTER_ITEMS) + list(ANAT_ITEMS)
                )
                if not filled:
                    continue
                if not (row.get("annotator_id") or row.get("ann_id") or "").strip():
                    raise SystemExit(f"{p}: row missing annotator_id for {row.get('sample_id')}")
                recs.append(row_to_record(row))
    return recs


def load_json_dir(root: Path) -> List[dict]:
    recs: List[dict] = []
    for p in sorted(root.glob("*/*.json")):
        if p.name.startswith("_"):
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        if "annotator_id" not in obj:
            obj["annotator_id"] = p.parent.name
        if "key" not in obj:
            obj["key"] = pair_key(obj["sample_id"], obj["model_id"])
        recs.append(obj)
    return recs


def write_human(hc: Path, recs: List[dict], *, force: bool) -> int:
    n = 0
    for r in recs:
        out = hc / "human" / r["annotator_id"] / f"{r['key']}.json"
        if out.is_file() and not force:
            continue
        atomic_write_json(out, r)
        n += 1
    return n


def build_consensus(hc: Path, *, force: bool) -> dict:
    by_key: Dict[str, List[dict]] = defaultdict(list)
    human_root = hc / "human"
    if not human_root.is_dir():
        raise SystemExit(f"missing {human_root}")
    for p in human_root.glob("*/*.json"):
        if p.parent.name.startswith("_"):
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        by_key[obj["key"]].append(obj)

    cons_dir = human_root / "_consensus"
    cons_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = 0
    summary_rows = []
    for key, anns in sorted(by_key.items()):
        out = cons_dir / f"{key}.json"
        if out.is_file() and not force:
            n_skip += 1
            continue
        intent = anns[0].get("intent_shown") or "unspecified"
        # If there are many people intent If inconsistent, take the mode.
        intents = [a.get("intent_shown") for a in anns if a.get("intent_shown")]
        if intents:
            intent = max(set(intents), key=intents.count)
        c = consensus_checklist(anns, intent=str(intent))
        payload = {
            "sample_id": anns[0]["sample_id"],
            "model_id": anns[0]["model_id"],
            "key": key,
            "protocol": PROTOCOL_ID,
            "intent_shown": intent,
            "annotator_ids": [a["annotator_id"] for a in anns],
            **c,
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        atomic_write_json(out, payload)
        n_ok += 1
        summary_rows.append(
            {
                "key": key,
                "n_ann": len(anns),
                "Inter_pass": c.get("Inter_pass"),
                "Anat_pass": c.get("Anat_pass"),
                "n_dropped": len(c.get("dropped_items") or {}),
            }
        )
    return {"n_keys": len(by_key), "n_written": n_ok, "n_skip": n_skip, "rows": summary_rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default="")
    ap.add_argument("--csv", action="append", default=[], help="Can be repeated; required annotator_id")
    ap.add_argument("--from-json-dir", default="", help="Already human/ The tree only does consensus")
    ap.add_argument("--consensus", action="store_true", default=True)
    ap.add_argument("--no-consensus", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = pack_root(args.pack or None)
    hc = root / "judgments" / "human_consistency"
    hc.mkdir(parents=True, exist_ok=True)

    recs: List[dict] = []
    if args.csv:
        recs = load_csvs([Path(p).expanduser() for p in args.csv])
        n = write_human(hc, recs, force=args.force)
        print(json.dumps({"imported_rows": len(recs), "wrote_json": n}, ensure_ascii=False))
    elif args.from_json_dir:
        src = Path(args.from_json_dir).expanduser()
        recs = load_json_dir(src)
        n = write_human(hc, recs, force=args.force)
        print(json.dumps({"imported_rows": len(recs), "wrote_json": n}, ensure_ascii=False))

    do_cons = args.consensus and not args.no_consensus
    if do_cons and (hc / "human").is_dir():
        summary = build_consensus(hc, force=args.force)
        atomic_write_json(
            hc / "human" / "_consensus" / "_import_summary.json",
            {k: summary[k] for k in ("n_keys", "n_written", "n_skip")},
        )
        print(
            json.dumps(
                {
                    "consensus_keys": summary["n_keys"],
                    "consensus_written": summary["n_written"],
                    "consensus_skip": summary["n_skip"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif not args.csv and not args.from_json_dir:
        raise SystemExit("provide --csv and/or run with existing human/ for --consensus")


if __name__ == "__main__":
    main()
