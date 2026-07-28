#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preliminary: ann_01 (H) vs Mesh (M) vs Checklist_V (V)."""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_EVAL = Path(__file__).resolve().parent
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from checklist_common import ANAT_ITEMS, INTER_ITEMS, pair_key  # noqa: E402
from pack_io import pack_root  # noqa: E402


def mean(xs: Sequence) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None and isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def spearman(xs, ys) -> Tuple[Optional[float], int]:
    pairs = [
        (float(a), float(b))
        for a, b in zip(xs, ys)
        if a is not None and b is not None and isinstance(a, (int, float)) and isinstance(b, (int, float))
    ]
    n = len(pairs)
    if n < 3:
        return None, n

    def ranks(vals: List[float]) -> List[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(vals):
            j = i
            while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    xv = [p[0] for p in pairs]
    yv = [p[1] for p in pairs]
    rx, ry = ranks(xv), ranks(yv)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx < 1e-12 or deny < 1e-12:
        return None, n
    return num / (denx * deny), n


def bin_stats(a: List, b: List) -> Optional[dict]:
    pairs = []
    for x, y in zip(a, b):
        if x is None or y is None or x == "U" or y == "U":
            continue
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            pairs.append((int(x), int(y)))
    if not pairs:
        return None
    n = len(pairs)
    acc = sum(1 for x, y in pairs if x == y) / n
    out = {"n": n, "acc": acc, "kappa": None}
    if all(x in (0, 1) and y in (0, 1) for x, y in pairs):
        tp = sum(1 for x, y in pairs if x == 1 and y == 1)
        tn = sum(1 for x, y in pairs if x == 0 and y == 0)
        fp = sum(1 for x, y in pairs if x == 0 and y == 1)
        fn = sum(1 for x, y in pairs if x == 1 and y == 0)
        po = (tp + tn) / n
        pe = ((tp + fp) / n) * ((tp + fn) / n) + ((tn + fn) / n) * ((tn + fp) / n)
        if abs(1 - pe) < 1e-12:
            kappa = 1.0 if po == 1 else 0.0
        else:
            kappa = (po - pe) / (1 - pe)
        out.update(
            {
                "kappa": kappa,
                "pos_h": (tp + fn) / n,
                "pos_other": (tp + fp) / n,
            }
        )
    return out


def load_ann(hc: Path, ann_id: str) -> Dict[str, dict]:
    root = hc / "human" / ann_id
    out: Dict[str, dict] = {}
    if not root.is_dir():
        return out
    for p in root.glob("*.json"):
        o = json.loads(p.read_text(encoding="utf-8"))
        out[o.get("key") or pair_key(o["sample_id"], o["model_id"])] = o
    return out


def load_M(hc: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for p in (hc / "mesh_bin").rglob("*.json"):
        if p.name.startswith("_"):
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        k = o.get("key") or pair_key(o.get("sample_id") or "", o.get("model_id") or "")
        out[k] = o
        out[p.stem] = o
    return out


def load_V(hc: Path, judge: str = "gpt-5.5") -> Dict[str, dict]:
    root = hc / "checklist_vlm" / judge
    out: Dict[str, dict] = {}
    for p in root.rglob("*.json"):
        if p.name.startswith("_"):
            continue
        o = json.loads(p.read_text(encoding="utf-8"))
        out[o.get("key") or pair_key(o["sample_id"], o["model_id"])] = o
    return out


def fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def main() -> None:
    root = pack_root()
    hc = root / "judgments" / "human_consistency"
    split = json.loads((hc / "_split.json").read_text(encoding="utf-8"))
    rows = []
    for name, rs in (split.get("splits") or {}).items():
        for r in rs:
            d = dict(r)
            d["split"] = name
            d["key"] = d.get("key") or pair_key(d["sample_id"], d["model_id"])
            rows.append(d)

    H = load_ann(hc, "ann_01")
    M = load_M(hc)
    V = load_V(hc)

    records = []
    for r in rows:
        k = r["key"]
        m = M.get(k) or M.get(f"{r['sample_id']}__{r['model_id']}")
        records.append({"meta": r, "H": H.get(k), "M": m, "V": V.get(k)})

    n_h = sum(1 for x in records if x["H"])
    n_v = sum(1 for x in records if x["V"])
    n_m = sum(1 for x in records if x["M"])
    hm = [x for x in records if x["H"] and x["M"]]
    hv = [x for x in records if x["H"] and x["V"]]
    hmv = [x for x in records if x["H"] and x["M"] and x["V"]]

    # --- H by model ---
    by_m: Dict[str, dict] = defaultdict(
        lambda: {"n": 0, "Qi": [], "Qa": [], "Si": [], "Sa": []}
    )
    for x in records:
        h = x["H"]
        if not h:
            continue
        mid = x["meta"]["model_id"]
        by_m[mid]["n"] += 1
        by_m[mid]["Qi"].append(h.get("Q_inter"))
        by_m[mid]["Qa"].append(h.get("Q_anat"))
        by_m[mid]["Si"].append(h.get("S_inter_H"))
        by_m[mid]["Sa"].append(h.get("S_anat_H"))

    Qi_h = [x["H"]["Q_inter"] for x in hv]
    Qi_v = [x["V"]["Q_inter"] for x in hv]
    Qa_h = [x["H"]["Q_anat"] for x in hv]
    Qa_v = [x["V"]["Q_anat"] for x in hv]
    sp_qi, _ = spearman(Qi_h, Qi_v)
    sp_qa, _ = spearman(Qa_h, Qa_v)
    dQi = mean(Qi_v) - mean(Qi_h) if mean(Qi_v) is not None and mean(Qi_h) is not None else None
    dQa = mean(Qa_v) - mean(Qa_h) if mean(Qa_v) is not None and mean(Qa_h) is not None else None

    Qi = [x["H"]["Q_inter"] for x in hm]
    Qm = [x["M"].get("S_inter_mesh_raw") for x in hm]
    Qa = [x["H"]["Q_anat"] for x in hm]
    Qa_m = [x["M"].get("S_anat_mesh_raw") for x in hm]
    sp_qim, n_qim = spearman(Qi, Qm)
    sp_qam, n_qam = spearman(Qa, Qa_m)
    # same length 1–5：Q_H ↔ Q_M(Depend on S_mesh Binning mapping)
    Qi_m5 = [x["M"].get("Q_inter") for x in hm]
    Qa_m5 = [x["M"].get("Q_anat") for x in hm]
    sp_qi_qm, n_qi_qm = spearman(Qi, Qi_m5)
    sp_qa_qm, n_qa_qm = spearman(Qa, Qa_m5)
    Si_h = [x["H"].get("S_inter_H_geom") or x["H"].get("S_inter_H") for x in hm]
    Si_m = [x["M"].get("S_inter_M") for x in hm]
    Sa_h = [x["H"].get("S_anat_H_geom") or x["H"].get("S_anat_H") for x in hm]
    Sa_m = [x["M"].get("S_anat_M") for x in hm]
    sp_sim, n_sim = spearman(Si_h, Si_m)
    sp_sam, n_sam = spearman(Sa_h, Sa_m)

    lines: List[str] = []
    lines.append("# Preliminary comparison:ann_01 (H) · Mesh (M) · Checklist_V (V)\n\n")
    lines.append(
        "> **Notice**: No three-person majority vote gold standard yet; here **H = only ann_01**. The conclusion is only a direction judgment.\n\n"
    )
    lines.append(
        f"- cover:H={n_h}/170，V={n_v}/170，Mhit={n_m}，"
        f"H∩M={len(hm)}，H∩V={len(hv)}, three parties={len(hmv)}\n\n"
    )

    lines.append("## 1. ann_01 by model\n\n")
    lines.append("| model | n | Q_inter | Q_anat | S_inter_H | S_anat_H |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for m in sorted(by_m, key=lambda k: -by_m[k]["n"]):
        d = by_m[m]
        lines.append(
            f"| `{m}` | {d['n']} | {fmt(mean(d['Qi']),2)} | {fmt(mean(d['Qa']),2)} | "
            f"{fmt(mean(d['Si']),2)} | {fmt(mean(d['Sa']),2)} |\n"
        )
    all_qi = [x["H"]["Q_inter"] for x in records if x["H"]]
    all_qa = [x["H"]["Q_anat"] for x in records if x["H"]]
    lines.append(
        f"\nwhole:Q_inter mean={fmt(mean(all_qi))} dist={dict(Counter(all_qi))}；"
        f"Q_anat mean={fmt(mean(all_qa))} dist={dict(Counter(all_qa))}\n"
    )
    hs_i = [x["H"]["S_inter_H"] for x in records if x["H"] and x["H"].get("S_inter_H") is not None]
    hq_i = [x["H"]["Q_inter"] for x in records if x["H"] and x["H"].get("S_inter_H") is not None]
    hs_a = [x["H"]["S_anat_H"] for x in records if x["H"] and x["H"].get("S_anat_H") is not None]
    hq_a = [x["H"]["Q_anat"] for x in records if x["H"] and x["H"].get("S_anat_H") is not None]
    sp_hsqi, n1 = spearman(hs_i, hq_i)
    sp_hsqa, n2 = spearman(hs_a, hq_a)
    lines.append(
        f"- item-level construct vs overall:ρ(S_inter_H,Q_inter)={fmt(sp_hsqi)} (n={n1})；"
        f"ρ(S_anat_H,Q_anat)={fmt(sp_hsqa)} (n={n2})\n"
    )

    lines.append("\n## 2. H vs V(Inflated / related)\n\n")
    lines.append("| index | H | V | V−H |\n|---|---:|---:|---:|\n")
    lines.append(
        f"| Q_inter | {fmt(mean(Qi_h))} | {fmt(mean(Qi_v))} | {fmt(dQi)} |\n"
    )
    lines.append(
        f"| Q_anat | {fmt(mean(Qa_h))} | {fmt(mean(Qa_v))} | {fmt(dQa)} |\n"
    )
    lines.append(
        f"\n- Spearman(Q_inter H,V)={fmt(sp_qi)}\n"
        f"- Spearman(Q_anat H,V)={fmt(sp_qa)}\n\n"
    )

    # S means HV
    Si_h_all = [x["H"].get("S_inter_H") for x in hv]
    Si_v_all = [x["V"].get("S_inter_V") or x["V"].get("S_inter_H") for x in hv]
    Sa_h_all = [x["H"].get("S_anat_H") for x in hv]
    Sa_v_all = [x["V"].get("S_anat_V") or x["V"].get("S_anat_H") for x in hv]
    lines.append(
        f"- S_inter mean H={fmt(mean(Si_h_all))} V={fmt(mean(Si_v_all))} "
        f"(Not empty n_H={sum(1 for x in Si_h_all if x is not None)}, "
        f"n_V={sum(1 for x in Si_v_all if x is not None)})\n"
    )
    lines.append(
        f"- S_anat mean H={fmt(mean(Sa_h_all))} V={fmt(mean(Sa_v_all))} "
        f"(Not empty n_H={sum(1 for x in Sa_h_all if x is not None)}, "
        f"n_V={sum(1 for x in Sa_v_all if x is not None)})\n\n"
    )

    lines.append("### H↔V Item level (exact consistency / κ）\n\n")
    lines.append("| item | n | acc | κ | posH | posV |\n|---|---:|---:|---:|---:|---:|\n")
    for item in list(INTER_ITEMS) + list(ANAT_ITEMS):
        ah, av = [], []
        for x in hv:
            g = "inter" if item in INTER_ITEMS else "anat"
            ah.append((x["H"].get(g) or {}).get(item))
            av.append((x["V"].get(g) or {}).get(item))
        st = bin_stats(ah, av)
        if not st:
            continue
        lines.append(
            f"| {item} | {st['n']} | {st['acc']:.3f} | {fmt(st.get('kappa'))} | "
            f"{fmt(st.get('pos_h'))} | {fmt(st.get('pos_other'))} |\n"
        )

    lines.append("\n## 3. H vs M(have mesh subset)\n\n")
    lines.append(
        f"n={len(hm)}  by_model={dict(Counter(x['meta']['model_id'] for x in hm))}\n\n"
    )
    lines.append("| contrast | Spearman ρ | n |\n|---|---:|---:|\n")
    lines.append(f"| **Q_inter ↔ Q_inter_M (1–5same length)** | {fmt(sp_qi_qm)} | {n_qi_qm} |\n")
    lines.append(f"| **Q_anat ↔ Q_anat_M (1–5same length)** | {fmt(sp_qa_qm)} | {n_qa_qm} |\n")
    lines.append(f"| Q_inter ↔ S_inter_mesh (0–1) | {fmt(sp_qim)} | {n_qim} |\n")
    lines.append(f"| Q_anat ↔ S_anat_mesh (0–1) | {fmt(sp_qam)} | {n_qam} |\n")
    lines.append(f"| S_inter_H_geom ↔ S_inter_M | {fmt(sp_sim)} | {n_sim} |\n")
    lines.append(f"| S_anat_H_geom ↔ S_anat_M | {fmt(sp_sam)} | {n_sam} |\n")
    lines.append(
        f"\n- Q_M Mean:Q_inter_M={fmt(mean(Qi_m5))}，Q_anat_M={fmt(mean(Qa_m5))}；"
        f"See the mapping rules mesh_bin Inside `Q_map`(default S<0.4→1 … ≥0.85→5）\n\n"
    )

    lines.append("### H↔M item level\n\n")
    lines.append("| item | n | acc | κ |\n|---|---:|---:|---:|\n")
    for item in ("I0", "I1", "Ic", "A1", "A2", "A3", "A4"):
        ah, am = [], []
        g = "inter" if item in INTER_ITEMS or item == "Ic" else "anat"
        for x in hm:
            ah.append((x["H"].get(g) or {}).get(item))
            am.append((x["M"].get(g) or {}).get(item))
        st = bin_stats(ah, am)
        if st:
            lines.append(
                f"| {item} | {st['n']} | {st['acc']:.3f} | {fmt(st.get('kappa'))} |\n"
            )

    lines.append("\n## 4. Who is more attached to the same subset? H（H∩M∩V）\n\n")
    lines.append(f"n={len(hmv)}\n\n")
    lines.append("| anchor | ρ(H,M) | ρ(H,V) | More stickers |\n|---|---:|---:|:---:|\n")
    triples = [
        (
            "Q_inter (1–5same length)",
            lambda h: h.get("Q_inter"),
            lambda m: m.get("Q_inter"),
            lambda v: v.get("Q_inter"),
        ),
        (
            "Q_anat (1–5same length)",
            lambda h: h.get("Q_anat"),
            lambda m: m.get("Q_anat"),
            lambda v: v.get("Q_anat"),
        ),
        (
            "Q_inter ↔ S_mesh",
            lambda h: h.get("Q_inter"),
            lambda m: m.get("S_inter_mesh_raw"),
            lambda v: v.get("Q_inter"),
        ),
        (
            "Q_anat ↔ S_mesh",
            lambda h: h.get("Q_anat"),
            lambda m: m.get("S_anat_mesh_raw"),
            lambda v: v.get("Q_anat"),
        ),
        (
            "S_inter_geom",
            lambda h: h.get("S_inter_H_geom") or h.get("S_inter_H"),
            lambda m: m.get("S_inter_M"),
            lambda v: v.get("S_inter_V") or v.get("S_inter_H"),
        ),
        (
            "S_anat_geom",
            lambda h: h.get("S_anat_H_geom") or h.get("S_anat_H"),
            lambda m: m.get("S_anat_M"),
            lambda v: v.get("S_anat_V") or v.get("S_anat_H"),
        ),
    ]
    win_m = win_v = 0
    for label, hg, mg, vg in triples:
        hs, ms, vs = [], [], []
        for x in hmv:
            a, b, c = hg(x["H"]), mg(x["M"]), vg(x["V"])
            if a is None or b is None or c is None:
                continue
            hs.append(a)
            ms.append(b)
            vs.append(c)
        sp_m, n = spearman(hs, ms)
        sp_v, _ = spearman(hs, vs)
        if sp_m is not None and (sp_v is None or sp_m > sp_v):
            win = "M"
            win_m += 1
        elif sp_v is not None and (sp_m is None or sp_v > sp_m):
            win = "V"
            win_v += 1
        else:
            win = "≈"
        lines.append(f"| {label} | {fmt(sp_m)} | {fmt(sp_v)} | {win} |\n")

    lines.append("\n## 5. Whether the desired results are obtained (preliminary judgment)\n\n")
    inflate = (dQi is not None and dQi > 0.15) or (dQa is not None and dQa > 0.15)
    m_ok = (sp_qim is not None and sp_qim >= 0.25) or (sp_qam is not None and sp_qam >= 0.25)
    lines.append(
        f"1. **V Falsely high**：Q_inter Δ={fmt(dQi)}，Q_anat Δ={fmt(dQa)} → "
        f"{'**There are signs of false highs**' if inflate else 'The virtual height is not obvious (or H On the high side/Scale squeeze)'}\n"
    )
    lines.append(
        f"2. **M Stick to human preferences**：ρ(Q_inter,S_inter_mesh)={fmt(sp_qim)}，"
        f"ρ(Q_anat,S_anat_mesh)={fmt(sp_qam)} → "
        f"{'**Directional support**' if m_ok else '**Weak**, three people are required to get the gold standard/Let’s look at hard cases again'}\n"
    )
    lines.append(
        f"3. **More posts with the same subset**: Three-way comparison in progress M win {win_m} item / V win {win_v} item"
        f"(by ρ high-low meter)\n"
    )
    lines.append(
        "4. **limit**:Single player H；H↔M Mainly in flux/dreamo；A5/Ir Blind spots will drag down the overall concept S Related.\n"
    )

    reports = hc / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    out = reports / "prelim_ann01_HMV.md"
    out.write_text("".join(lines), encoding="utf-8")

    # also json for numbers
    summary = {
        "H": "ann_01_only",
        "n_H": n_h,
        "n_V": n_v,
        "n_HM": len(hm),
        "n_HV": len(hv),
        "n_HMV": len(hmv),
        "Q_inter_H": mean(Qi_h),
        "Q_inter_V": mean(Qi_v),
        "Q_anat_H": mean(Qa_h),
        "Q_anat_V": mean(Qa_v),
        "delta_Q_inter_V_minus_H": dQi,
        "delta_Q_anat_V_minus_H": dQa,
        "rho_Q_inter_HV": sp_qi,
        "rho_Q_anat_HV": sp_qa,
        "rho_Q_inter_S_inter_mesh": sp_qim,
        "rho_Q_anat_S_anat_mesh": sp_qam,
        "rho_Q_inter_HM_1to5": sp_qi_qm,
        "rho_Q_anat_HM_1to5": sp_qa_qm,
        "Q_inter_M_mean": mean(Qi_m5),
        "Q_anat_M_mean": mean(Qa_m5),
        "rho_S_inter_HM_geom": sp_sim,
        "rho_S_anat_HM_geom": sp_sam,
        "by_model_H": {
            m: {
                "n": by_m[m]["n"],
                "Q_inter": mean(by_m[m]["Qi"]),
                "Q_anat": mean(by_m[m]["Qa"]),
            }
            for m in by_m
        },
    }
    (reports / "prelim_ann01_HMV.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("".join(lines))
    print(f"\nwrote {out}")
    print(f"wrote {reports / 'prelim_ann01_HMV.json'}")


if __name__ == "__main__":
    main()
