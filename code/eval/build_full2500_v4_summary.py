#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build paper-rule full2500 summary: Anat v4 + Inter v3.1.

Aggregation matches rescore_mesh_inter.summarize():
  mean over ok==True rows (recon_fail typically has S_*=0).

Does not overwrite mesh_v3. Writes:
  data/eval_outputs/full2500_v4/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_mesh_v3 import (  # noqa: E402
    CONTACT_LEVELS,
    DEFAULT_TAGGED_CSV,
    load_contact_c_map,
)
from compare_mesh_v31_subset import recompose as recompose_v31  # noqa: E402
from rescore_anat_v4_exp import (  # noqa: E402
    PAPER_MODELS,
    compose_v4_from_rec,
    load_protocol,
)
from rescore_mesh_inter import summarize  # noqa: E402

MODEL_ORDER = list(PAPER_MODELS)


def score_pair(
    rec: Dict[str, Any],
    proto: Dict[str, Any],
    inter_rec: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a judgment-like dict with S_anat_mesh=Anat_v4, S_inter_mesh=Inter_v3.1."""
    out = dict(rec)
    v4 = compose_v4_from_rec(rec, proto)
    out["S_anat_mesh"] = float(v4["S_anat_v4"])
    out["S_anat_v4"] = float(v4["S_anat_v4"])
    out["anat_protocol"] = "anat_v4_exp"
    for k, v in v4.items():
        if k.startswith("P_anat_") or k in ("attached_signature", "anat_formula"):
            out[k] = v
    if inter_rec is not None and inter_rec.get("S_inter_mesh") is not None:
        out["S_inter_mesh"] = float(inter_rec["S_inter_mesh"])
        # keep pen diagnostics from inter_rec when present
        for k in (
            "P_fuse",
            "P_miss",
            "P_qual",
            "S_pen",
            "S_prox",
            "inter_protocol",
            "pen_signal",
        ):
            if k in inter_rec:
                out[k] = inter_rec[k]
    else:
        _, i31, meta = recompose_v31(rec)
        out["S_inter_mesh"] = float(i31)
        out["inter_protocol"] = "inter_v3.1"
        for k, v in meta.items():
            if k.startswith("P_") or k in ("S_pen", "S_prox", "pen_signal"):
                out[k.replace("inter_", "") if k.startswith("inter_") else k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument(
        "--inter-root",
        default="",
        help="optional judgments/mesh_v31 root (else recompose Inter on the fly)",
    )
    ap.add_argument(
        "--protocol-json",
        default="",
    )
    ap.add_argument(
        "--tagged-csv",
        default=str(DEFAULT_TAGGED_CSV),
    )
    ap.add_argument(
        "--out",
        default=str(
            Path("data") / "eval_outputs"
            / "full2500_v4"
            / "summary.json"
        ),
    )
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    proto_path = (
        Path(args.protocol_json).expanduser()
        if args.protocol_json
        else pack / "judgments" / "mesh_anat_exp" / "_protocol.json"
    )
    proto = load_protocol(proto_path if proto_path.is_file() else None)
    inter_root = (
        Path(args.inter_root).expanduser().resolve()
        if args.inter_root
        else None
    )
    # auto-detect transfer unpack
    if inter_root is None:
        cand = (
            Path.home()
            / "transfer"
            / "mpie_mesh_v31_full_20260722"
            / "judgments"
            / "mesh_v31"
        )
        if cand.is_dir():
            inter_root = cand

    contact = load_contact_c_map(Path(args.tagged_csv).expanduser())
    models: Dict[str, dict] = {}
    by_density: Dict[str, dict] = {}

    for mid in MODEL_ORDER:
        src = pack / "judgments" / "mesh_v3" / mid
        if not src.is_dir():
            print(f"SKIP {mid}")
            continue
        recs: List[dict] = []
        dens_buckets = {c: {"anat": [], "inter": []} for c in CONTACT_LEVELS}
        for p in sorted(src.glob("*.json")):
            if p.name.startswith("_"):
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            irec = None
            if inter_root is not None:
                ip = inter_root / mid / p.name
                if ip.is_file():
                    irec = json.loads(ip.read_text(encoding="utf-8"))
            scored = score_pair(rec, proto, irec)
            recs.append(scored)
            if scored.get("ok") and scored.get("S_anat_mesh") is not None:
                sid = scored.get("sample_id") or p.stem
                cc = contact.get(sid)
                if cc in dens_buckets:
                    dens_buckets[cc]["anat"].append(float(scored["S_anat_mesh"]))
                    dens_buckets[cc]["inter"].append(float(scored["S_inter_mesh"]))

        summary = summarize(recs)
        models[mid] = {
            "S_anat_mesh": summary.get("S_anat_mesh"),
            "S_inter_mesh": summary.get("S_inter_mesh"),
            "n": summary.get("n"),
            "n_ok": summary.get("n_ok"),
            "recon_fail_rate": summary.get("recon_fail_rate"),
            "anat_protocol": "anat_v4_exp",
            "inter_protocol": "inter_v3.1",
        }
        by_c = {}
        for c in CONTACT_LEVELS:
            a = dens_buckets[c]["anat"]
            i = dens_buckets[c]["inter"]
            by_c[c] = {
                "Anat": float(np.mean(a)) if a else None,
                "Inter": float(np.mean(i)) if i else None,
                "n": len(a),
            }
        a0, a3 = by_c["C0"]["Anat"], by_c["C3"]["Anat"]
        i0, i3 = by_c["C0"]["Inter"], by_c["C3"]["Inter"]
        by_density[mid] = {
            "by_c": by_c,
            "n_ok_tagged": sum(by_c[c]["n"] for c in CONTACT_LEVELS),
            "delta_anat_c3_c0": (a3 - a0) if a0 is not None and a3 is not None else None,
            "delta_inter_c3_c0": (i3 - i0) if i0 is not None and i3 is not None else None,
        }
        print(
            f"{mid:28s} Anat={models[mid]['S_anat_mesh']:.4f} "
            f"Inter={models[mid]['S_inter_mesh']:.4f} "
            f"n_ok={models[mid]['n_ok']}",
            flush=True,
        )

    # ranking helpers
    def rank_key(axis: str):
        return sorted(
            models.keys(),
            key=lambda m: -(models[m].get(axis) or -1.0),
        )

    out = {
        "protocol": "anat_v4_exp+inter_v3.1",
        "pack": str(pack),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": 2500,
        "models": models,
        "ranking": {
            "Anat": rank_key("S_anat_mesh"),
            "Inter": rank_key("S_inter_mesh"),
        },
        "by_density": by_density,
        "anat_protocol_path": str(proto_path),
        "inter_root": str(inter_root) if inter_root else "recompose_v31",
        "aggregation": "summarize(): mean over ok==True (recon_fail→0 scores)",
        "note": "Paper cutover table source. Does not mutate judgments/mesh_v3.",
    }
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    # also compact 2dp table
    table = []
    for mid in MODEL_ORDER:
        if mid not in models:
            continue
        table.append(
            {
                "model": mid,
                "Anat": round(float(models[mid]["S_anat_mesh"]), 2),
                "Inter": round(float(models[mid]["S_inter_mesh"]), 2),
                "Anat_raw": models[mid]["S_anat_mesh"],
                "Inter_raw": models[mid]["S_inter_mesh"],
            }
        )
    (out_path.parent / "tab_main_anat_inter.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out_path}")
    print(f"wrote {out_path.parent / 'tab_main_anat_inter.json'}")
    for t in table:
        print(f"  {t['model']:28s} {t['Anat']:.2f}  {t['Inter']:.2f}")


if __name__ == "__main__":
    main()
