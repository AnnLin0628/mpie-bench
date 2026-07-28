#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scene-cluster bootstrap CIs under Anat v4 + Inter v3.1 (paper summarize rule)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_scene_ci import (  # noqa: E402
    DISPLAY,
    MODEL_ORDER,
    group_by_scene,
    paired_scene_delta,
    scene_bootstrap,
)
from build_full2500_v4_summary import score_pair  # noqa: E402
from rescore_anat_v4_exp import load_protocol  # noqa: E402


def load_v4_scores(
    mesh_dir: Path,
    proto: Dict[str, Any],
    inter_dir: Optional[Path],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for p in mesh_dir.glob("*.json"):
        if p.name.startswith("_"):
            continue
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("ok") is False:
            continue
        inter_rec = None
        if inter_dir is not None:
            ip = inter_dir / p.name
            if ip.is_file():
                try:
                    inter_rec = json.loads(ip.read_text(encoding="utf-8"))
                except Exception:
                    inter_rec = None
        scored = score_pair(rec, proto, inter_rec)
        sid = scored.get("sample_id") or p.stem
        out[sid] = {
            "Anat": float(scored["S_anat_mesh"]),
            "Inter": float(scored["S_inter_mesh"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
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
    ap.add_argument(
        "--out",
        default=str(
            Path("data") / "eval_outputs"
            / "full2500_v4"
            / "scene_ci.json"
        ),
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    proto = load_protocol(pack / "judgments" / "mesh_anat_exp" / "_protocol.json")
    inter_root = Path(args.inter_root).expanduser() if args.inter_root else None
    if inter_root is not None and not inter_root.is_dir():
        inter_root = None

    results: Dict[str, Any] = {
        "protocol": "anat_v4_exp+inter_v3.1",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "models": {},
    }
    by_model_scene: Dict[str, Dict[str, List[Dict[str, float]]]] = {}

    for mid in MODEL_ORDER:
        mdir = pack / "judgments" / "mesh_v3" / mid
        idir = (inter_root / mid) if inter_root is not None else None
        scores = load_v4_scores(mdir, proto, idir)
        by_sc = group_by_scene(scores)
        by_model_scene[mid] = by_sc
        stats = scene_bootstrap(by_sc, n_boot=args.n_boot, seed=args.seed)
        stats["display"] = DISPLAY.get(mid, mid)
        results["models"][mid] = stats
        a, i = stats["axes"]["Anat"], stats["axes"]["Inter"]
        print(
            f"{DISPLAY.get(mid, mid):22s}  "
            f"Anat {a['mean']:.3f} [{a['ci95'][0]:.3f},{a['ci95'][1]:.3f}]  "
            f"Inter {i['mean']:.3f} [{i['ci95'][0]:.3f},{i['ci95'][1]:.3f}]",
            flush=True,
        )

    a, b = "seedream-5-pro", "gemini-3-pro-image"
    results["paired"] = {
        f"{a}_minus_{b}_Inter": paired_scene_delta(
            by_model_scene[a],
            by_model_scene[b],
            axis="Inter",
            n_boot=args.n_boot,
            seed=args.seed + 17,
        )
    }
    print("paired:", json.dumps(results["paired"], indent=2), flush=True)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    tex = []
    for mid in MODEL_ORDER:
        st = results["models"][mid]
        a, i = st["axes"]["Anat"], st["axes"]["Inter"]
        tex.append(
            "{} & {:.2f} [{:.2f},{:.2f}] & {:.2f} [{:.2f},{:.2f}] \\\\".format(
                st["display"],
                a["mean"],
                a["ci95"][0],
                a["ci95"][1],
                i["mean"],
                i["ci95"][0],
                i["ci95"][1],
            )
        )
    out.with_suffix(".texrows.txt").write_text("\n".join(tex) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
