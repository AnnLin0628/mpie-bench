#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MPIE eval protocol v3 — Multi-HMR mesh scoring on a pack.

Requires Multi-HMR installed (see install_multihmr.sh) + SMPLX_NEUTRAL.npz.

Examples (on a GPU host):
  # GT Good set (calibration / go-no-go)
  python score_mesh_v3.py --pack "$MPIE_TEST_PACK" --gt-only --limit 50

  # one model
  python score_mesh_v3.py --pack "$MPIE_TEST_PACK" --model-id firered

  # all models under outputs/
  python score_mesh_v3.py --pack "$MPIE_TEST_PACK" --all-models

  # calibrate tau_pen from GT then re-score
  python score_mesh_v3.py --pack "$MPIE_TEST_PACK" --gt-only --calibrate
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from mesh_metrics import (
    compose_inter_score,
    infer_needs_contact,
    prompt_contact_intent,
    score_humans,
)
from rescore_mesh_inter import summarize

# ---------------------------------------------------------------------------
# Multi-HMR lazy loader (repo must be on PYTHONPATH or installed editable)
# ---------------------------------------------------------------------------

_MHMR = None


def _import_multihmr(repo: Optional[Path]):
    global _MHMR
    if _MHMR is not None:
        return _MHMR
    if repo:
        sys.path.insert(0, str(repo))
    try:
        # demo.py helpers from naver/multi-hmr
        from demo import (  # type: ignore
            forward_model,
            get_camera_parameters,
            load_model,
            open_image,
        )
    except Exception as e:
        raise RuntimeError(
            "Cannot import Multi-HMR demo helpers. "
            "Clone naver/multi-hmr, pip install -r requirements.txt, "
            "and pass --multihmr-repo /path/to/multi-hmr. "
            f"Original error: {e}"
        ) from e
    _MHMR = {
        "load_model": load_model,
        "open_image": open_image,
        "get_camera_parameters": get_camera_parameters,
        "forward_model": forward_model,
    }
    return _MHMR


class MultiHMRBackend:
    def __init__(
        self,
        repo: Optional[Path],
        model_name: str = "multiHMR_896_L",
        det_thresh: float = 0.3,
        device: str = "cuda",
    ):
        import torch

        self.torch = torch
        assert torch.cuda.is_available(), "score_mesh_v3 needs CUDA"
        if not repo:
            raise RuntimeError("pass --multihmr-repo /path/to/multi-hmr")
        repo = Path(repo).expanduser().resolve()
        self.repo = repo

        # demo.py uses *relative* paths (models/multiHMR/...). Must run with cwd=repo.
        ckpt = repo / "models" / "multiHMR" / f"{model_name}.pt"
        if not ckpt.is_file() or ckpt.stat().st_size < 100_000_000:
            # remove empty 404 leftovers
            if ckpt.is_file() and ckpt.stat().st_size < 100_000_000:
                ckpt.unlink()
            raise FileNotFoundError(
                f"Missing Multi-HMR ckpt at {ckpt}\n"
                f"Download once (from repo root):\n"
                f"  mkdir -p {repo}/models/multiHMR\n"
                f"  wget -c -O {ckpt} \\\n"
                f"    https://download.europe.naverlabs.com/ComputerVision/MultiHMR/{model_name}.pt\n"
                f"(old URL .../multihmr/{model_name}.pt is 404 — do not use it)"
            )

        m = _import_multihmr(repo)
        prev = Path.cwd()
        try:
            os.chdir(repo)
            self.model = m["load_model"](model_name, device=torch.device(device))
        finally:
            os.chdir(prev)
        self._open = m["open_image"]
        self._cam = m["get_camera_parameters"]
        self._fwd = m["forward_model"]
        self.det_thresh = det_thresh
        self.model_name = model_name
        self.name = "multi_hmr"
        self.img_size = int(getattr(self.model, "img_size", 896) or 896)
        # faces
        if hasattr(self.model, "smpl_layer"):
            self.faces = np.asarray(
                self.model.smpl_layer["neutral_10"].bm_x.faces, dtype=np.int64
            )
        else:
            self.faces = np.asarray(self.model.body_model.faces.cpu().numpy(), dtype=np.int64)

    def infer(
        self, img_path: Path
    ) -> Tuple[
        List[np.ndarray], List[Any], List[Any], List[Any], List[Any], List[float], float
    ]:
        """Returns verts, poses, j3ds, shapes, j2ds, det_scores, elapsed_ms."""
        t0 = time.time()
        prev = Path.cwd()
        try:
            os.chdir(self.repo)
            x, _ = self._open(str(img_path), self.model.img_size)
            K = self._cam(self.model.img_size)
            humans = self._fwd(self.model, x, K, det_thresh=self.det_thresh)
        finally:
            os.chdir(prev)
        verts_list, poses, j3ds, shapes, j2ds, scores = [], [], [], [], [], []
        for h in humans:
            name = "verts_smplx" if "verts_smplx" in h else "v3d"
            v = h[name]
            if hasattr(v, "detach"):
                v = v.detach().cpu().numpy()
            verts_list.append(np.asarray(v, dtype=np.float64))
            pose = None
            for k in ("pose", "body_pose", "rotvec", "smplx_pose"):
                if k in h:
                    pose = h[k]
                    break
            poses.append(pose)
            j = h.get("j3d")
            if j is not None and hasattr(j, "detach"):
                j = j.detach().cpu().numpy()
            j3ds.append(np.asarray(j, dtype=np.float64) if j is not None else None)
            sh = h.get("shape")
            if sh is not None and hasattr(sh, "detach"):
                sh = sh.detach().cpu().numpy()
            shapes.append(np.asarray(sh, dtype=np.float64) if sh is not None else None)
            j2 = h.get("j2d")
            if j2 is not None and hasattr(j2, "detach"):
                j2 = j2.detach().cpu().numpy()
            j2ds.append(np.asarray(j2, dtype=np.float64) if j2 is not None else None)
            sc = h.get("scores")
            if sc is None:
                scores.append(0.0)
            else:
                if hasattr(sc, "detach"):
                    sc = sc.detach().cpu().numpy()
                scores.append(float(np.asarray(sc).reshape(-1)[0]))
        return verts_list, poses, j3ds, shapes, j2ds, scores, (time.time() - t0) * 1000.0


def select_top_k_humans(
    verts: List[np.ndarray],
    poses: List[Any],
    j3ds: List[Any],
    shapes: List[Any],
    j2ds: List[Any],
    scores: List[float],
    k: int,
) -> Tuple[
    List[np.ndarray], List[Any], List[Any], List[Any], List[Any], List[float], List[int]
]:
    """Keep top-k detections by Multi-HMR score (ignore bystanders)."""
    n = len(verts)
    if k <= 0 or n == 0:
        return [], [], [], [], [], [], []
    if n <= k:
        return verts, poses, j3ds, shapes, j2ds, scores, list(range(n))
    order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    order = sorted(order)
    return (
        [verts[i] for i in order],
        [poses[i] for i in order],
        [j3ds[i] for i in order],
        [shapes[i] for i in order],
        [j2ds[i] for i in order],
        [scores[i] for i in order],
        order,
    )


# ---------------------------------------------------------------------------
# Pack I/O
# ---------------------------------------------------------------------------

def load_manifest(pack: Path) -> List[dict]:
    rows = []
    with open(pack / "manifest.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def n_expected(row: dict) -> int:
    """Expected person count = unique R# mentioned in the edit prompt.

    Do NOT use raw detector count (background bystanders). Prefer prompt R1,R2,…
    over len(ref_relpaths): packs sometimes ship extra refs not named in the prompt.
    """
    for k in ("n_person", "n_people", "num_person"):
        if k in row and row[k] is not None:
            return int(row[k])
    prompt = row.get("prompt") or ""
    ids = set(re.findall(r"\bR(\d+)\b", prompt))
    if ids:
        return int(len(ids))
    # fallback: unique R# in ref filenames
    refs = row.get("ref_relpaths") or []
    ref_ids = set()
    for p in refs:
        m = re.search(r"(?:^|/|\\)R(\d+)_", str(p))
        if m:
            ref_ids.add(m.group(1))
    if ref_ids:
        return int(len(ref_ids))
    return max(2, len(refs)) if refs else 2


def needs_contact(row: dict) -> bool:
    return infer_needs_contact(
        prompt=row.get("prompt"),
        cat=row.get("cat"),
        contact_density=row.get("contact_density") or row.get("contact_density_level"),
    )


def atomic_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    arr = np.sort(np.asarray(xs, dtype=np.float64))
    idx = float(np.clip(q, 0.0, 1.0)) * (len(arr) - 1)
    lo, hi = int(np.floor(idx)), int(np.ceil(idx))
    if lo == hi:
        return float(arr[lo])
    w = idx - lo
    return float(arr[lo] * (1 - w) + arr[hi] * w)


def calibrate_inter_params(gt_records: List[dict]) -> Dict[str, float]:
    """Calibrate two-regime Inter bands from Good-set geometry."""
    vols = [
        float(r["pen_volume_m3"])
        for r in gt_records
        if r.get("ok") and r.get("pen_volume_m3") is not None
    ]
    pens = [
        float(r["pen_vert_ratio"])
        for r in gt_records
        if r.get("ok") and r.get("pen_vert_ratio") is not None
    ]
    # proximity band: only samples whose *prompt* requires contact
    dists = [
        float(r["min_surf_dist"])
        for r in gt_records
        if r.get("ok")
        and r.get("min_surf_dist") is not None
        and (r.get("min_surf_dist") == r.get("min_surf_dist"))
        and (
            r.get("contact_intent") == "required"
            or infer_needs_contact(prompt=r.get("prompt"), cat=r.get("cat"))
        )
    ]

    vol_ok = max(0.01, _percentile(vols, 0.55)) if vols else 0.05
    vol_bad = max(vol_ok * 1.8, _percentile(vols, 0.95) * 1.15) if vols else 0.15
    # legacy single-threshold (still written for older tools)
    tau_vol = max(vol_ok, _percentile(vols, 0.85) * 1.05) if vols else 0.05
    tau_pen = max(0.02, _percentile(pens, 0.85) * 1.05) if pens else 0.05

    d_good = 0.05
    d_fail = 0.40
    if dists:
        # GT contact scenes: allow up to ~p80 distance as "good enough"
        d_good = float(max(0.03, min(0.12, _percentile(dists, 0.80))))
        d_fail = float(max(d_good + 0.15, min(0.60, _percentile(dists, 0.98) + 0.15)))

    return {
        "vol_ok": float(vol_ok),
        "vol_bad": float(vol_bad),
        "tau_vol": float(tau_vol),
        "tau_pen": float(tau_pen),
        "d_good": float(d_good),
        "d_fail": float(d_fail),
        "tau_contact": 0.02,
    }


def rescore_inter_scalars(rec: dict, cal: Dict[str, float]) -> dict:
    """Recompute S_inter_mesh from cached geometry (no Multi-HMR)."""
    if not rec.get("ok") or rec.get("recon_fail"):
        return rec
    intent = rec.get("contact_intent")
    prompt = rec.get("prompt")
    if intent is None and prompt:
        intent = prompt_contact_intent(prompt)
    needs = rec.get("needs_contact")
    if needs is None and intent is not None:
        needs = intent == "required"
    if needs is None:
        needs = infer_needs_contact(cat=rec.get("cat"), prompt=prompt)
    inter = compose_inter_score(
        needs_contact=bool(needs),
        contact_intent=intent,
        prompt=prompt,
        min_surf_dist=float(
            rec["min_surf_dist"]
            if rec.get("min_surf_dist") is not None
            else float("nan")
        ),
        pen_volume_m3=rec.get("pen_volume_m3"),
        pen_vert_ratio=rec.get("pen_vert_ratio")
        if rec.get("pen_volume_m3") is None
        else None,
        under_detect=bool(rec.get("under_detect")),
        vol_ok=float(cal.get("vol_ok", 0.05)),
        vol_bad=float(cal.get("vol_bad", 0.15)),
        tau_pen=float(cal.get("tau_pen", 0.15)),
        d_good=float(cal.get("d_good", 0.05)),
        d_fail=float(cal.get("d_fail", 0.40)),
    )
    rec = dict(rec)
    rec.update(inter)
    rec["tau_pen"] = float(cal.get("tau_pen", rec.get("tau_pen") or 0.15))
    rec["tau_vol"] = float(cal.get("tau_vol", rec.get("tau_vol") or 0.05))
    rec["rescored_at"] = datetime.now().isoformat(timespec="seconds")
    return rec


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def score_one(
    backend: MultiHMRBackend,
    img_path: Path,
    row: dict,
    *,
    cal: Dict[str, Any],
    model_id: str,
    use_volume: bool = True,
) -> dict:
    sample_id = row["sample_id"]
    if not img_path.is_file():
        return {
            "sample_id": sample_id,
            "model_id": model_id,
            "ok": False,
            "error": f"missing image: {img_path}",
            "recon_fail": True,
        }
    try:
        n_exp = n_expected(row)
        verts_all, poses_all, j3ds_all, shapes_all, j2ds_all, scores_all, ms = backend.infer(
            img_path
        )
        n_raw = len(verts_all)
        verts, poses, j3ds, shapes, j2ds, scores, keep_idx = select_top_k_humans(
            verts_all,
            poses_all,
            j3ds_all,
            shapes_all,
            j2ds_all,
            scores_all,
            n_exp,
        )
        geo = score_humans(
            verts,
            backend.faces,
            n_exp,
            body_poses=poses,
            j3ds=j3ds,
            shapes=shapes,
            j2ds=j2ds,
            img_path=img_path,
            abhuman_weights=cal.get("abhuman_weights"),
            hmr_img_size=int(getattr(backend.model, "img_size", 896) or 896)
            if getattr(backend, "model", None) is not None
            else 896,
            needs_contact=needs_contact(row),
            prompt=row.get("prompt"),
            tau_pen=float(cal.get("tau_pen", 0.15)),
            tau_contact=float(cal.get("tau_contact", 0.02)),
            tau_vol=float(cal.get("tau_vol", 0.05)),
            vol_ok=float(cal.get("vol_ok", 0.05)),
            vol_bad=float(cal.get("vol_bad", 0.15)),
            d_good=float(cal.get("d_good", 0.05)),
            d_fail=float(cal.get("d_fail", 0.40)),
            use_volume=use_volume,
            n_detected_raw=n_raw,
        )
        return {
            "sample_id": sample_id,
            "model_id": model_id,
            "ok": True,
            "backend": getattr(backend, "name", "multi_hmr"),
            "backend_model": getattr(backend, "model_name", "multi_hmr"),
            "elapsed_ms": round(ms, 1),
            "cat": row.get("cat"),
            "prompt": row.get("prompt"),
            "img": str(img_path),
            "det_scores_kept": [round(s, 4) for s in scores],
            "keep_idx": keep_idx,
            "person_select": "top_k_by_score_vs_prompt_R",
            **geo,
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as e:
        return {
            "sample_id": sample_id,
            "model_id": model_id,
            "ok": False,
            "error": repr(e),
            "recon_fail": True,
            "written_at": datetime.now().isoformat(timespec="seconds"),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--multihmr-repo", default=os.environ.get("MULTIHMR_REPO", ""))
    ap.add_argument("--backend-model", default="multiHMR_896_L",
                    help="Multi-HMR ckpt stem; use multiHMR_896_L (synth name is 404)")
    ap.add_argument("--model-id", default="", help="outputs/<model_id>/")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--gt-only", action="store_true", help="score GT target frames into judgments/mesh_v3/_gt/")
    ap.add_argument("--calibrate", action="store_true", help="write _calibration.json from GT pens")
    ap.add_argument("--tau-pen", type=float, default=None)
    ap.add_argument("--tau-vol", type=float, default=None)
    ap.add_argument("--vol-ok", type=float, default=None)
    ap.add_argument("--vol-bad", type=float, default=None)
    ap.add_argument("--d-good", type=float, default=None)
    ap.add_argument("--d-fail", type=float, default=None)
    ap.add_argument("--tau-contact", type=float, default=0.02)
    ap.add_argument("--no-volume", action="store_true",
                    help="Disable trimesh.contains volume; use proximity proxy only")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="split manifest by index %% num_shards == shard_id (multi-GPU)",
    )
    ap.add_argument("--det-thresh", type=float, default=0.2,
                    help="Multi-HMR person detection threshold (lower=more people); "
                         "bystanders are dropped by top-k vs prompt R# anyway")
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--abhuman-weights",
        default=os.environ.get("ABHUMAN_WEIGHTS", ""),
        help="Optional AbHuman/YOLO weights; empty → ~/models/abhuman/best.pt if present",
    )
    args = ap.parse_args()

    pack = Path(args.pack).expanduser().resolve()
    repo = Path(args.multihmr_repo).expanduser() if args.multihmr_repo else None
    rows = load_manifest(pack)
    if args.limit > 0:
        rows = rows[: args.limit]
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard_id]
        print(
            f"[mesh] shard {args.shard_id}/{args.num_shards} → {len(rows)} samples",
            flush=True,
        )

    cal_path = pack / "judgments" / "mesh_v3" / "_calibration.json"
    cal: Dict[str, Any] = {}
    if cal_path.is_file():
        raw = json.loads(cal_path.read_text())
        for k in ("tau_pen", "tau_vol", "tau_contact", "vol_ok", "vol_bad", "d_good", "d_fail"):
            if k in raw and raw[k] is not None:
                cal[k] = float(raw[k])
        if raw.get("abhuman_weights"):
            cal["abhuman_weights"] = raw["abhuman_weights"]
    # defaults + CLI overrides
    cal.setdefault("tau_pen", 0.05)
    cal.setdefault("tau_vol", 0.05)
    cal.setdefault("tau_contact", float(args.tau_contact))
    cal.setdefault("vol_ok", 0.05)
    cal.setdefault("vol_bad", 0.15)
    cal.setdefault("d_good", 0.05)
    cal.setdefault("d_fail", 0.40)
    if args.tau_pen is not None:
        cal["tau_pen"] = args.tau_pen
    if args.tau_vol is not None:
        cal["tau_vol"] = args.tau_vol
    if args.vol_ok is not None:
        cal["vol_ok"] = args.vol_ok
    if args.vol_bad is not None:
        cal["vol_bad"] = args.vol_bad
    if args.d_good is not None:
        cal["d_good"] = args.d_good
    if args.d_fail is not None:
        cal["d_fail"] = args.d_fail
    if args.abhuman_weights:
        cal["abhuman_weights"] = args.abhuman_weights
    use_volume = not args.no_volume

    backend = MultiHMRBackend(
        repo, model_name=args.backend_model, det_thresh=args.det_thresh
    )

    # --- GT ---
    if args.gt_only or args.calibrate:
        out_dir = pack / "judgments" / "mesh_v3" / "_gt"
        recs = []
        for row in rows:
            rel = row.get("gt_relpath")
            if not rel:
                continue
            img = pack / rel
            out_p = out_dir / f"{row['sample_id']}.json"
            if out_p.exists() and not args.force:
                recs.append(json.loads(out_p.read_text()))
                continue
            print(f"[mesh] start {row['sample_id']} ...", flush=True)
            r = score_one(
                backend,
                img,
                row,
                cal=cal,
                model_id="_gt",
                use_volume=use_volume,
            )
            atomic_write(out_p, r)
            recs.append(r)
            print(
                json.dumps(
                    {
                        "gt": row["sample_id"],
                        "ok": r.get("ok"),
                        "regime": r.get("inter_regime"),
                        "n_exp": r.get("n_expected"),
                        "n_raw": r.get("n_detected_raw"),
                        "n": r.get("n_humans"),
                        "vol": r.get("pen_volume_m3"),
                        "dist": r.get("min_surf_dist"),
                        "S_anat": r.get("S_anat_mesh"),
                        "S_inter": r.get("S_inter_mesh"),
                        "ms": r.get("elapsed_ms"),
                        "err": (r.get("error") or "")[:160],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        summary = summarize(recs)
        atomic_write(out_dir / "_summary.json", summary)
        if args.calibrate:
            ok_recs = [
                r
                for r in recs
                if r.get("ok") and r.get("pen_vert_ratio") is not None
            ]
            cal = calibrate_inter_params(ok_recs)
            cal["tau_contact"] = float(args.tau_contact)
            rescored = []
            for r in recs:
                rr = rescore_inter_scalars(r, cal)
                rescored.append(rr)
                out_p = out_dir / f"{rr['sample_id']}.json"
                if rr.get("ok"):
                    atomic_write(out_p, rr)
            recs = rescored
            summary = summarize(recs)
            atomic_write(out_dir / "_summary.json", summary)
            cal_out = {
                **cal,
                "n_gt": len(ok_recs),
                "gt_summary": summary,
                "person_select": "top_k_by_score_vs_prompt_R",
                "inter_formula": (
                    "penalty: S=1-P_fuse[-P_miss|-P_unwanted]; "
                    "intent from prompt text (required/forbidden/unspecified)"
                ),
                "written_at": datetime.now().isoformat(timespec="seconds"),
                "note": (
                    "vol_ok/vol_bad from GT volumes; d_* from prompt-required "
                    "contact subset; intent from prompt not category"
                ),
            }
            atomic_write(cal_path, cal_out)
            print(json.dumps({"calibration": cal_out}, ensure_ascii=False, indent=2))
        print(json.dumps({"gt_summary": summary}, ensure_ascii=False, indent=2))
        if args.gt_only and not args.all_models and not args.model_id:
            return

    # --- models ---
    models: List[str] = []
    if args.all_models:
        out_root = pack / "outputs"
        models = sorted([p.name for p in out_root.iterdir() if p.is_dir()]) if out_root.is_dir() else []
    elif args.model_id:
        models = [args.model_id]
    else:
        if not args.gt_only:
            ap.error("pass --model-id / --all-models / --gt-only")

    if cal_path.is_file():
        raw = json.loads(cal_path.read_text())
        for k in ("tau_pen", "tau_vol", "tau_contact", "vol_ok", "vol_bad", "d_good", "d_fail"):
            if k in raw and raw[k] is not None:
                cal[k] = float(raw[k])

    for mid in models:
        out_dir = pack / "judgments" / "mesh_v3" / mid
        recs = []
        for row in rows:
            sid = row["sample_id"]
            cand = [
                pack / "outputs" / mid / f"{sid}.png",
                pack / "outputs" / mid / f"{sid}.jpg",
                pack / "outputs" / mid / f"{sid}.webp",
            ]
            img = next((c for c in cand if c.is_file()), cand[0])
            out_p = out_dir / f"{sid}.json"
            if out_p.exists() and not args.force:
                recs.append(json.loads(out_p.read_text()))
                continue
            r = score_one(
                backend,
                img,
                row,
                cal=cal,
                model_id=mid,
                use_volume=use_volume,
            )
            atomic_write(out_p, r)
            recs.append(r)
            print(
                json.dumps(
                    {
                        "model": mid,
                        "sample": sid,
                        "ok": r.get("ok"),
                        "regime": r.get("inter_regime"),
                        "S_inter": r.get("S_inter_mesh"),
                        "S_anat": r.get("S_anat_mesh"),
                        "vol": r.get("pen_volume_m3"),
                        "dist": r.get("min_surf_dist"),
                        "n": r.get("n_humans"),
                        "n_raw": r.get("n_detected_raw"),
                    },
                    ensure_ascii=False,
                )
            )
        # rescore with latest cal so S_inter uses band formula even if geometry
        # was written under an older default
        recs = [rescore_inter_scalars(r, cal) for r in recs]
        for r in recs:
            if r.get("ok") and r.get("sample_id"):
                atomic_write(out_dir / f"{r['sample_id']}.json", r)
        summary = summarize(recs)
        summary["shard_id"] = args.shard_id
        summary["num_shards"] = args.num_shards
        if args.num_shards > 1:
            atomic_write(out_dir / f"_summary_shard{args.shard_id}.json", summary)
        else:
            atomic_write(out_dir / "_summary.json", summary)
        print(json.dumps({"model": mid, "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
