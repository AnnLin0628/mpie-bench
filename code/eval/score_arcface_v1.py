#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Smoke / pack ArcFace ID scorer (eval protocol v2).

Implements a practical subset of docs/02_pipeline_design/eval_id_protocol.md:
  - ref embeddings from pack images (largest face; optional freeze file)
  - gen faces × refs → Hungarian assignment
  - GT-visible branch: if GT has no face match for a ref → exclude that person
  - GT-visible ∧ gen unmatched / below threshold → score 0 (anti hide-face)
  - report S_id, face_visible_rate, match_rate, matched_similarity, M_CP

Yaw-bucket calibration table is NOT required for smoke; use --sim-threshold
(provisional). Mark results with protocol_note=provisional_threshold.

Usage (conda env mpie):
  python score_arcface_v1.py --pack "$MPIE_TEST_PACK" --model-id gpt-image-2
  python score_arcface_v1.py --pack "$MPIE_TEST_PACK" --all-models
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# local
from pack_io import load_manifest, count_outputs  # type: ignore


def list_model_ids(pack: Path) -> list:
    out = pack / "outputs"
    if not out.is_dir():
        return []
    return sorted(
        p.name for p in out.iterdir()
        if p.is_dir() and not p.name.startswith("_") and count_outputs(pack, p.name) > 0
    )


def _face_app(ctx_id: int = 0):
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="antelopev2", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _embs(app, img_bgr) -> List[np.ndarray]:
    if img_bgr is None:
        return []
    faces = app.get(img_bgr)
    out = []
    for f in faces:
        e = getattr(f, "normed_embedding", None)
        if e is None:
            continue
        out.append(np.asarray(e, dtype=np.float32))
    return out


def _ref_emb(app, img_bgr) -> Optional[np.ndarray]:
    faces = app.get(img_bgr) if img_bgr is not None else []
    f = _largest_face(faces)
    if f is None:
        return None
    e = getattr(f, "normed_embedding", None)
    return np.asarray(e, dtype=np.float32) if e is not None else None


def _hungarian_sims(ref_embs: List[Optional[np.ndarray]], gen_embs: List[np.ndarray]) -> Tuple[List[float], List[Optional[int]]]:
    """Return per-ref similarity (0 if unmatched) and gen index assigned (or None)."""
    from scipy.optimize import linear_sum_assignment
    m, n = len(ref_embs), len(gen_embs)
    sims = [0.0] * m
    assign = [None] * m
    if n == 0 or m == 0:
        return sims, assign
    C = np.zeros((m, n), dtype=np.float64)
    valid_rows = []
    for i, e in enumerate(ref_embs):
        if e is None:
            continue
        valid_rows.append(i)
        C[i] = -(gen_embs @ e)  # maximize sim ↔ minimize -sim
    if not valid_rows:
        return sims, assign
    # run on full matrix; invalid rows (None emb) stay with cost 0
    ri, gi = linear_sum_assignment(C)
    for a, b in zip(ri, gi):
        if ref_embs[a] is None:
            continue
        sims[a] = float(-C[a, b])
        assign[a] = int(b)
    return sims, assign


def _mcp(app, gen_bgr, ref_bgr, gen_face_idx: Optional[int]) -> float:
    """Cheap DCT correlation copy-paste proxy (same idea as metrics_lite)."""
    if gen_face_idx is None or gen_bgr is None or ref_bgr is None:
        return 0.0
    gen_faces = app.get(gen_bgr)
    ref_faces = app.get(ref_bgr)
    if not gen_faces or gen_face_idx >= len(gen_faces) or not ref_faces:
        return 0.0
    gf = gen_faces[gen_face_idx].bbox.astype(int)
    rf = _largest_face(ref_faces).bbox.astype(int)
    crop = gen_bgr[max(0, gf[1]):gf[3], max(0, gf[0]):gf[2]]
    rcrop = ref_bgr[max(0, rf[1]):rf[3], max(0, rf[0]):rf[2]]
    if crop.size == 0 or rcrop.size == 0:
        return 0.0
    h1 = cv2.dct(np.float32(cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 32))))[:8, :8]
    h2 = cv2.dct(np.float32(cv2.resize(cv2.cvtColor(rcrop, cv2.COLOR_BGR2GRAY), (32, 32))))[:8, :8]
    c = float(np.corrcoef(h1.ravel(), h2.ravel())[0, 1])
    return 0.0 if np.isnan(c) else c


def score_one(app, pack: Path, row: dict, model_id: str, thr: float) -> dict:
    sid = row["sample_id"]
    gen_path = pack / "outputs" / model_id / f"{sid}.png"
    if not gen_path.is_file():
        # try jpg
        alt = pack / "outputs" / model_id / f"{sid}.jpg"
        gen_path = alt if alt.is_file() else gen_path
    if not gen_path.is_file():
        return {"sample_id": sid, "model_id": model_id, "ok": False, "error": "missing_gen"}

    gen = cv2.imread(str(gen_path))
    ref_imgs, ref_embs = [], []
    for rp in row.get("ref_relpaths") or []:
        p = pack / rp
        img = cv2.imread(str(p)) if p.is_file() else None
        ref_imgs.append(img)
        ref_embs.append(_ref_emb(app, img))

    gen_embs = _embs(app, gen)
    raw_sims, assign = _hungarian_sims(ref_embs, gen_embs)

    # GT visibility: match each ref emb to GT faces; if no GT or no match ≥ thr → ¬gt_visible
    gt_path = row.get("gt_relpath")
    gt_vis = []
    if gt_path:
        gt = cv2.imread(str(pack / gt_path)) if (pack / gt_path).is_file() else None
        gt_embs = _embs(app, gt)
        for e in ref_embs:
            if e is None or not gt_embs:
                gt_vis.append(False)
                continue
            best = max(float(g @ e) for g in gt_embs)
            gt_vis.append(best >= thr)
    else:
        # no GT → treat all refs with emb as visible (conservative for anti-cheat)
        gt_vis = [e is not None for e in ref_embs]

    per = []
    scored = []
    n_gt_vis = 0
    n_gen_vis = 0
    matched_sims = []
    mcp_vals = []
    for i, (sim, gi) in enumerate(zip(raw_sims, assign)):
        visible = bool(gt_vis[i]) if i < len(gt_vis) else False
        if not visible:
            per.append({"ref": f"R{i+1}", "gt_visible": False, "gen_matched": False,
                        "sim": None, "score": None, "excluded": True})
            continue
        n_gt_vis += 1
        matched = gi is not None and sim >= thr
        if matched:
            n_gen_vis += 1
            matched_sims.append(sim)
            score = float(sim)  # keep raw cosine in [~0,1]
            mcp_vals.append(_mcp(app, gen, ref_imgs[i], gi))
        else:
            score = 0.0  # hide-face / fail match → floor
        scored.append(score)
        per.append({
            "ref": f"R{i+1}",
            "gt_visible": True,
            "gen_matched": matched,
            "sim": round(float(sim), 4) if gi is not None else None,
            "score": round(score, 4),
            "excluded": False,
        })

    S_id = float(np.mean(scored)) if scored else None
    return {
        "sample_id": sid,
        "model_id": model_id,
        "ok": True,
        "S_id": None if S_id is None else round(S_id, 4),
        "face_visible_rate": round(n_gen_vis / n_gt_vis, 4) if n_gt_vis else None,
        "match_rate": round(n_gen_vis / n_gt_vis, 4) if n_gt_vis else None,
        "matched_similarity": round(float(np.mean(matched_sims)), 4) if matched_sims else None,
        "M_CP": round(float(max(mcp_vals)), 4) if mcp_vals else 0.0,
        "sim_threshold": thr,
        "n_ref": len(ref_embs),
        "n_gen_faces": len(gen_embs),
        "n_gt_visible": n_gt_vis,
        "per_ref": per,
        "protocol_note": "provisional_threshold; yaw buckets not applied",
        "gen_relpath": str(gen_path.relative_to(pack)),
        "written_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_model(
    pack: Path,
    model_id: str,
    thr: float,
    ctx_id: int,
    resume: bool,
    limit: int,
    shard_id: int = 0,
    num_shards: int = 1,
):
    out_dir = pack / "judgments" / "arcface_v1" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    app = _face_app(ctx_id)
    rows = load_manifest(pack)
    if limit > 0:
        rows = rows[:limit]
    if num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % num_shards == shard_id]
    ok = skip = fail = miss = 0
    t0 = time.time()
    for row in rows:
        sid = row["sample_id"]
        out_p = out_dir / f"{sid}.json"
        if resume and out_p.is_file() and out_p.stat().st_size > 50:
            skip += 1
            continue
        gen = pack / "outputs" / model_id / f"{sid}.png"
        if not gen.is_file() and not (pack / "outputs" / model_id / f"{sid}.jpg").is_file():
            miss += 1
            continue
        try:
            res = score_one(app, pack, row, model_id, thr)
            tmp = out_p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, out_p)
            if res.get("ok"):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            err = {"sample_id": sid, "model_id": model_id, "ok": False, "error": repr(e)}
            out_p.write_text(json.dumps(err, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[fail] {sid}: {e}", flush=True)
    summary = {
        "model_id": model_id,
        "shard_id": shard_id,
        "num_shards": num_shards,
        "ok": ok, "skip": skip, "fail": fail, "missing_gen": miss,
        "elapsed_sec": round(time.time() - t0, 1),
        "sim_threshold": thr,
        "out_dir": str(out_dir),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    tag = f"_run_summary_shard{shard_id}.json" if num_shards > 1 else "_run_summary.json"
    (out_dir / tag).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--model-id", default="")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--sim-threshold", type=float, default=0.25,
                    help="provisional cosine gate after Hungarian (yaw buckets later)")
    ap.add_argument("--ctx-id", type=int, default=0,
                    help="insightface GPU id (use with CUDA_VISIBLE_DEVICES=N --ctx-id 0)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1,
                    help="split manifest by index %% num_shards == shard_id (multi-GPU)")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    pack = Path(args.pack).expanduser().resolve()
    resume = not args.no_resume
    if args.all_models:
        models = list_model_ids(pack)
    elif args.model_id:
        models = [args.model_id]
    else:
        raise SystemExit("pass --model-id or --all-models")
    for mid in models:
        print(f"=== arcface {mid} shard {args.shard_id}/{args.num_shards} ===", flush=True)
        run_model(
            pack, mid, args.sim_threshold, args.ctx_id, resume, args.limit,
            shard_id=args.shard_id, num_shards=args.num_shards,
        )


if __name__ == "__main__":
    main()
