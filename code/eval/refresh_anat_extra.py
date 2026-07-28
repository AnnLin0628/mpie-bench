#!/usr/bin/env python3
"""Refresh only Anat P_extra / leftover (keep Inter & volume fields).

Needs Multi-HMR again (j2d was not stored / was mis-aligned), but skips
trimesh volume — much cheaper than full FORCE score_mesh_v3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anat_extended import (  # noqa: E402
    compose_anat_score,
    explain_residual_proxy,
    structure_overcount_score,
)
from rescore_mesh_inter import summarize  # noqa: E402
from score_mesh_v3 import (  # noqa: E402
    MultiHMRBackend,
    load_manifest,
    select_top_k_humans,
)

try:
    from score_mesh_v3 import n_expected  # type: ignore
except ImportError:  # pragma: no cover
    import re

    def n_expected(row: dict) -> int:
        prompt = row.get("prompt") or ""
        rs = set(re.findall(r"\bR(\d+)\b", prompt))
        return max(1, len(rs)) if rs else int(row.get("n_expected") or 2)


def refresh_one(
    backend: MultiHMRBackend,
    rec: dict,
    row: dict,
    img: Path,
) -> dict:
    if not rec.get("ok") or rec.get("recon_fail"):
        return rec
    if not img.is_file():
        rec = dict(rec)
        rec["anat_overcount_note"] = f"missing_img:{img}"
        return rec

    n_exp = int(rec.get("n_expected") or n_expected(row))
    verts_all, poses_all, j3ds_all, shapes_all, j2ds_all, scores_all, ms = backend.infer(
        img
    )
    verts, poses, j3ds, shapes, j2ds, scores, keep_idx = select_top_k_humans(
        verts_all,
        poses_all,
        j3ds_all,
        shapes_all,
        j2ds_all,
        scores_all,
        n_exp,
    )
    img_size = int(getattr(backend.model, "img_size", 896) or 896)
    over = structure_overcount_score(
        img,
        n_exp,
        j2ds=j2ds,
        j3ds=j3ds,
        verts=verts,
        img_size=img_size,
    )
    resid = explain_residual_proxy(img, j2ds, img_size=img_size)
    if not over.get("available"):
        print(
            f"  WARN overcount unavailable note={over.get('note')} "
            f"n_j2d={len(j2ds)} img_size={img_size} img={img}",
            flush=True,
        )

    composed = compose_anat_score(
        s_residual=resid.get("score")
        if resid.get("available")
        else rec.get("S_anat_residual"),
        s_overcount=over.get("score") if over.get("available") else None,
        s_scale=rec.get("S_anat_scale"),
        s_ownership=rec.get("S_anat_ownership"),
        s_part_mesh=rec.get("S_anat_part_mesh"),
        s_person=rec.get("S_anat_person"),
        s_abhuman=rec.get("S_anat_abhuman"),
        under_detect=bool(rec.get("under_detect")),
        recon_fail=False,
        n_detected_raw=rec.get("n_detected_raw"),
        n_expected=n_exp,
        leftover_frac=over.get("leftover_frac"),
        n_leftover_blobs=over.get("n_leftover_blobs"),
    )
    out = dict(rec)
    out.update(composed)
    out["S_anat_residual"] = resid.get("score")
    out["S_anat_overcount"] = over.get("score")
    out["anat_leftover_frac"] = over.get("leftover_frac")
    out["anat_orphan_frac"] = over.get("orphan_frac")
    out["anat_n_leftover_blobs"] = over.get("n_leftover_blobs")
    out["anat_overcount_note"] = over.get("note")
    out["anat_scene"] = {
        **(rec.get("anat_scene") or {}),
        "structure_overcount": over,
        "explain_residual": resid,
        **{k: composed[k] for k in composed if k.startswith("P_") or k.startswith("w_")},
    }
    out["anat_refresh_ms"] = round(ms, 1)
    out["keep_idx"] = keep_idx
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    # default=None so empty `--pack "$PACK"` (unset env) does not override
    ap.add_argument("--pack", default=None)
    ap.add_argument("--multihmr-repo", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pack_s = (args.pack or os.environ.get("PACK") or "").strip()
    if not pack_s:
        pack_s = str(Path.home() / "mpie_testset_pack")
    repo_s = (args.multihmr_repo or os.environ.get("MULTIHMR_REPO") or "").strip()
    if not repo_s:
        repo_s = str(Path.home() / "models" / "multi-hmr")

    pack = Path(pack_s).expanduser().resolve()
    repo = Path(repo_s).expanduser().resolve()
    if not (pack / "manifest.jsonl").is_file():
        raise SystemExit(
            f"manifest.jsonl not found under pack={pack}\n"
            f"Export PACK or pass --pack "$MPIE_TEST_PACK""
        )
    print(f"pack={pack}\nmultihmr-repo={repo}", flush=True)
    rows = {r["sample_id"]: r for r in load_manifest(pack)}
    backend = MultiHMRBackend(repo)

    root = pack / "judgments" / "mesh_v3"
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        recs: List[dict] = []
        paths = sorted(p for p in sub.glob("*.json") if not p.name.startswith("_"))
        if args.limit > 0:
            paths = paths[: args.limit]
        for i, p in enumerate(paths):
            rec = json.loads(p.read_text())
            sid = rec.get("sample_id") or p.stem
            row = rows.get(sid, rec)
            # resolve image
            img = Path(rec["img"]) if rec.get("img") else None
            if img is None or not img.is_file():
                rel = row.get("gt_relpath") if sub.name == "_gt" else None
                if rel:
                    img = pack / rel
                else:
                    # outputs/<model>/<sample>.png variants
                    for ext in (".png", ".jpg", ".jpeg", ".webp"):
                        cand = pack / "outputs" / sub.name / f"{sid}{ext}"
                        if cand.is_file():
                            img = cand
                            break
            if img is None:
                img = Path("/nonexistent")
            print(
                f"[{sub.name} {i+1}/{len(paths)}] {sid} ...",
                flush=True,
            )
            new = refresh_one(backend, rec, row, img)
            p.write_text(json.dumps(new, ensure_ascii=False, indent=2))
            recs.append(new)
        # --limit: summarize refreshed samples only (avoid mixing stale jsons)
        summary = summarize(recs)
        if args.limit > 0:
            print(
                f"NOTE {sub.name}: --limit={args.limit} → summary over "
                f"{len(recs)} refreshed only (skip writing _summary.json)",
                flush=True,
            )
        else:
            (sub / "_summary.json").write_text(json.dumps(summary, indent=2))
        print(
            sub.name,
            {
                k: summary.get(k)
                for k in (
                    "S_anat_mesh",
                    "S_anat_overcount_mean",
                    "P_anat_extra_mean",
                    "anat_orphan_frac_mean",
                    "anat_leftover_frac_mean",
                    "S_inter_mesh",
                )
            },
        )


if __name__ == "__main__":
    main()
