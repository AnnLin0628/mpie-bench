#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remesh Checklist_M for consistency pool under Anat v4 + Inter v3.1.

Writes enriched mesh overlays → mesh_bin via frozen item_map_v6.2, then
recomputes itemwise Spearman ρ(H,M)/ρ(H,V) into reports/itemwise_HMV_v4.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_full2500_v4_summary import score_pair  # noqa: E402
from checklist_common import (  # noqa: E402
    ANAT_ITEMS,
    INTER_ITEMS,
    atomic_write_json,
    load_thresholds,
    map_mesh_to_checklist,
    pair_key,
)
from rescore_anat_v4_exp import load_protocol  # noqa: E402


def spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None

    def ranks(vals: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx < 1e-12 or deny < 1e-12:
        return None
    return float(num / (denx * deny))


def _to_num(v: Any) -> Optional[float]:
    if v is None or v == "U" or v == "u":
        return None
    try:
        return float(v)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pack",
        default=str(Path.home() / "mpie_testset_pack"),
    )
    ap.add_argument(
        "--full-pack",
        default=str(Path.home() / "mpie_testset_pack"),
    )
    ap.add_argument(
        "--inter-root",
        default=str(
            Path.home()
            / "transfer"
            / "mpie_mesh_v31_full_20260722"
            / "judgments"
            / "mesh_v31"
        ),
    )
    ap.add_argument("--judge-model", default="gpt-5.5")
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    full = Path(args.full_pack).expanduser().resolve()
    inter_root = Path(args.inter_root).expanduser()
    if not inter_root.is_dir():
        inter_root = None  # type: ignore

    hc = pack / "judgments" / "human_consistency"
    split = json.loads((hc / "_split.json").read_text(encoding="utf-8"))
    thr = load_thresholds(hc / "_thresholds.json")
    item_map = (
        json.loads((hc / "_item_map_calib.json").read_text(encoding="utf-8"))
        if (hc / "_item_map_calib.json").is_file()
        else None
    )
    calib_path = full / "judgments" / "mesh_v3" / "_calibration.json"
    calib = (
        json.loads(calib_path.read_text(encoding="utf-8"))
        if calib_path.is_file()
        else {}
    )
    proto = load_protocol(full / "judgments" / "mesh_anat_exp" / "_protocol.json")

    out_bin = hc / "mesh_bin_v4"
    out_bin.mkdir(parents=True, exist_ok=True)
    overlay_dir = hc / "mesh_overlay_v4"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    n = n_miss = 0
    rows: List[dict] = []
    for name, rs in (split.get("splits") or {}).items():
        for r in rs:
            rows.append(r)

    for r in rows:
        sid, mid = r["sample_id"], r["model_id"]
        src = full / "judgments" / "mesh_v3" / mid / f"{sid}.json"
        if not src.is_file():
            src = pack / "judgments" / "mesh_v3" / mid / f"{sid}.json"
        if not src.is_file():
            n_miss += 1
            continue
        rec = json.loads(src.read_text(encoding="utf-8"))
        inter_rec = None
        if inter_root is not None:
            ip = inter_root / mid / f"{sid}.json"
            if ip.is_file():
                inter_rec = json.loads(ip.read_text(encoding="utf-8"))
        scored = score_pair(rec, proto, inter_rec)
        # Feed attach/orphan into A1 feature used by frozen map
        scored["P_anat_extra"] = float(
            max(
                float(scored.get("P_anat_attach") or scored.get("P_anat_extra") or 0),
                float(scored.get("P_anat_orphan") or 0),
            )
        )
        scored["model_id"] = mid
        scored["sample_id"] = sid
        atomic_write_json(overlay_dir / f"{pair_key(sid, mid)}.json", scored)
        mapped = map_mesh_to_checklist(
            scored, thresholds=thr, calib=calib, item_map=item_map
        )
        mapped["model_id"] = mid
        mapped["sample_id"] = sid
        mapped["anat_protocol"] = "anat_v4_exp"
        mapped["inter_protocol"] = "inter_v3.1"
        mapped["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        atomic_write_json(out_bin / f"{pair_key(sid, mid)}.json", mapped)
        n += 1
    print(f"wrote {n} mesh_bin_v4 (miss={n_miss})", flush=True)

    # Promote to mesh_bin for agreement tools (backup once)
    live = hc / "mesh_bin"
    bak = hc / "mesh_bin_mesh_v3_backup"
    if live.is_dir() and not bak.exists():
        live.rename(bak)
        print(f"backed up mesh_bin → {bak}", flush=True)
    if out_bin.is_dir():
        if live.exists() or live.is_symlink():
            if live.is_symlink() or live.is_file():
                live.unlink()
            else:
                # replace contents
                import shutil

                shutil.rmtree(live)
        # copy v4 → live
        import shutil

        shutil.copytree(out_bin, live)
        print(f"installed mesh_bin from mesh_bin_v4", flush=True)

    # Itemwise Spearman vs human consensus + VLM
    cons_dir = hc / "human" / "_consensus"
    vroot = hc / "checklist_vlm" / args.judge_model.replace("/", "_")
    items = list(INTER_ITEMS) + list(ANAT_ITEMS)
    report_items: Dict[str, Any] = {}
    rho_hm_all: List[float] = []
    rho_hv_all: List[float] = []
    wins = 0
    retained = 0
    for item in items:
        hs, ms, vs = [], [], []
        for r in rows:
            key = pair_key(r["sample_id"], r["model_id"])
            hp = cons_dir / f"{key}.json"
            mp = live / f"{key}.json"
            # VLM may be nested by model
            vp = None
            for cand in (
                vroot / r["model_id"] / f"{r['sample_id']}.json",
                vroot / f"{key}.json",
            ):
                if cand.is_file():
                    vp = cand
                    break
            if not (hp.is_file() and mp.is_file() and vp is not None):
                continue
            H = json.loads(hp.read_text(encoding="utf-8"))
            M = json.loads(mp.read_text(encoding="utf-8"))
            V = json.loads(vp.read_text(encoding="utf-8"))
            # answers may be nested
            h_ans = H.get("answers") or H
            m_ans = M.get("answers") or M
            v_ans = V.get("answers") or V
            hv = _to_num(h_ans.get(item))
            mv = _to_num(m_ans.get(item))
            vv = _to_num(v_ans.get(item))
            if hv is None or mv is None or vv is None:
                continue
            hs.append(hv)
            ms.append(mv)
            vs.append(vv)
        rho_m = spearman(hs, ms)
        rho_v = spearman(hs, vs)
        agree_m = (
            sum(int(a == b) for a, b in zip(hs, ms)) / len(hs) if hs else None
        )
        agree_v = (
            sum(int(a == b) for a, b in zip(hs, vs)) / len(hs) if hs else None
        )
        winner = "—"
        if rho_m is not None and rho_v is not None:
            if item != "I0":
                retained += 1
                if rho_m > rho_v:
                    wins += 1
                    winner = "M"
                elif rho_v > rho_m:
                    winner = "V"
                else:
                    winner = "≈"
            else:
                winner = "V" if (rho_v or -9) > (rho_m or -9) else "M"
        report_items[item] = {
            "rho_HM": rho_m,
            "rho_HV": rho_v,
            "agree_HM": agree_m,
            "agree_HV": agree_v,
            "winner": winner,
            "n": len(hs),
        }
        if item != "I0" and rho_m is not None:
            rho_hm_all.append(rho_m)
        if item != "I0" and rho_v is not None:
            rho_hv_all.append(rho_v)

    overall_hm = sum(rho_hm_all) / len(rho_hm_all) if rho_hm_all else None
    overall_hv = sum(rho_hv_all) / len(rho_hv_all) if rho_hv_all else None

    # Preference anchors: per-editor mean Q vs mean S
    from collections import defaultdict

    by_mid: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"Qa": [], "Qi": [], "Sa": [], "Si": []}
    )
    for r in rows:
        key = pair_key(r["sample_id"], r["model_id"])
        hp = cons_dir / f"{key}.json"
        op = overlay_dir / f"{r['sample_id']}.json"
        # overlay named by sid only — may collide across models; use mesh_bin features
        mp = live / f"{key}.json"
        if not (hp.is_file() and mp.is_file()):
            continue
        H = json.loads(hp.read_text(encoding="utf-8"))
        M = json.loads(mp.read_text(encoding="utf-8"))
        h_ans = H.get("answers") or H
        feats = M.get("features") or M
        qa, qi = _to_num(h_ans.get("Q_anat")), _to_num(h_ans.get("Q_inter"))
        sa = _to_num(feats.get("S_anat_mesh") or M.get("S_anat_mesh_raw"))
        si = _to_num(feats.get("S_inter_mesh") or M.get("S_inter_mesh_raw"))
        mid = r["model_id"]
        if qa is not None and sa is not None:
            by_mid[mid]["Qa"].append(qa)
            by_mid[mid]["Sa"].append(sa)
        if qi is not None and si is not None:
            by_mid[mid]["Qi"].append(qi)
            by_mid[mid]["Si"].append(si)

    def _mean(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    Qa_m, Sa_m, Qi_m, Si_m = [], [], [], []
    for mid, d in by_mid.items():
        qa, sa, qi, si = (
            _mean(d["Qa"]),
            _mean(d["Sa"]),
            _mean(d["Qi"]),
            _mean(d["Si"]),
        )
        if qa is not None and sa is not None:
            Qa_m.append(qa)
            Sa_m.append(sa)
        if qi is not None and si is not None:
            Qi_m.append(qi)
            Si_m.append(si)
    pref = {
        "rho_Qanat_Sanat": spearman(Qa_m, Sa_m),
        "rho_Qinter_Sinter": spearman(Qi_m, Si_m),
        "n_editors_anat": len(Qa_m),
        "n_editors_inter": len(Qi_m),
    }

    out = {
        "version": "anat_v4+inter_v3.1_item_map_v6.2",
        "wins_M_over_V_excl_I0": wins,
        "retained_excl_I0": retained,
        "overall_mean_rho_HM": overall_hm,
        "overall_mean_rho_HV": overall_hv,
        "items": report_items,
        "preference_anchors": pref,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    reports = hc / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    atomic_write_json(reports / "itemwise_HMV_v4.json", out)
    print(json.dumps(out, indent=2)[:2500], flush=True)
    print(f"wrote {reports / 'itemwise_HMV_v4.json'}", flush=True)


if __name__ == "__main__":
    main()
