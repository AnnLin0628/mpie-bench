#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HPSv2 Qual scorer — offline-friendly for GPU hosts (no HF DNS).

Env / flags:
  MPIE_HPS_CKPT   path to HPS_v2.1_compressed.pt (or v2.0)
  MPIE_CLIP_CKPT  path to open_clip ViT-H-14 laion2B weights (.bin)

If not set, looks under:
  ~/mpie_weights/hpsv2/HPS_v2.1_compressed.pt
  ~/mpie_weights/hpsv2/open_clip_pytorch_model.bin
  ~/.cache/hpsv2/...
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

from pack_io import load_manifest, count_outputs  # type: ignore

_MODEL = {}


def list_model_ids(pack: Path) -> list:
    out = pack / "outputs"
    if not out.is_dir():
        return []
    return sorted(
        p.name for p in out.iterdir()
        if p.is_dir() and not p.name.startswith("_") and count_outputs(pack, p.name) > 0
    )


def _find_file(candidates: list) -> Path | None:
    for c in candidates:
        if not c:
            continue
        p = Path(c).expanduser()
        if p.is_file():
            return p.resolve()
    return None


def resolve_ckpts(hps_version: str) -> tuple[Path, Path]:
    hps_name = "HPS_v2.1_compressed.pt" if hps_version.startswith("v2.1") else "HPS_v2_compressed.pt"
    home = Path.home()
    hps = _find_file([
        os.environ.get("MPIE_HPS_CKPT"),
        str(home / "mpie_weights" / "hpsv2" / hps_name),
        str(home / ".cache" / "hpsv2" / hps_name),
        str(home / "mpie_weights" / "hpsv2" / "HPS_v2_compressed.pt"),
    ])
    clip = _find_file([
        os.environ.get("MPIE_CLIP_CKPT"),
        str(home / "mpie_weights" / "hpsv2" / "open_clip_pytorch_model.bin"),
        str(home / ".cache" / "hpsv2" / "open_clip_pytorch_model.bin"),
        ])
    if hps is None or clip is None:
        raise FileNotFoundError(
            "offline HPS weights missing.\n"
            f"  HPS ckpt ({hps_name}): {hps}\n"
            f"  CLIP ViT-H-14 bin: {clip}\n"
            "Place them under ~/mpie_weights/hpsv2/ or set MPIE_HPS_CKPT / MPIE_CLIP_CKPT.\n"
            "Download HPSv2 weights into MPIE_WEIGHTS/hpsv2 (see docs)."
        )
    return hps, clip


def get_model(hps_version: str = "v2.1"):
    if _MODEL:
        return _MODEL
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer

    hps_ckpt, clip_ckpt = resolve_ckpts(hps_version)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hps] clip={clip_ckpt}", flush=True)
    print(f"[hps] hps_ckpt={hps_ckpt} device={device}", flush=True)

    # pretrained=local path → open_clip will NOT hit HuggingFace
    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        str(clip_ckpt),
        precision="amp",
        device=device,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )
    checkpoint = torch.load(str(hps_ckpt), map_location=device)
    state = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer("ViT-H-14")
    _MODEL.update({
        "model": model,
        "preprocess": preprocess_val,
        "tokenizer": tokenizer,
        "device": device,
        "hps_ckpt": str(hps_ckpt),
        "clip_ckpt": str(clip_ckpt),
    })
    return _MODEL


@torch.no_grad()
def score_image(img_path: Path, prompt: str, hps_version: str) -> float:
    m = get_model(hps_version)
    device = m["device"]
    image = m["preprocess"](Image.open(img_path).convert("RGB")).unsqueeze(0).to(device=device, non_blocking=True)
    text = m["tokenizer"]([prompt]).to(device=device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=(device == "cuda")):
        outputs = m["model"](image, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits = image_features @ text_features.T
        score = float(torch.diagonal(logits).cpu().numpy()[0])
    return score


def _already_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 40:
        return False
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(j.get("ok")) and j.get("hpsv2") is not None


def run_model(pack, model_id, hps_version, resume, limit, shard_id=0, num_shards=1):
    out_dir = pack / "judgments" / "hpsv2" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # load model once per process
    get_model(hps_version)
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
        if resume and _already_ok(out_p):
            skip += 1
            continue
        gen = pack / "outputs" / model_id / f"{sid}.png"
        if not gen.is_file():
            gen = pack / "outputs" / model_id / f"{sid}.jpg"
        if not gen.is_file():
            miss += 1
            continue
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            fail += 1
            continue
        try:
            s = score_image(gen, prompt, hps_version)
            res = {
                "sample_id": sid,
                "model_id": model_id,
                "ok": True,
                "hpsv2": round(s, 6),
                "hps_version": hps_version,
                "prompt_used": prompt[:500],
                "gen_relpath": str(gen.relative_to(pack)),
                "written_at": datetime.now().isoformat(timespec="seconds"),
            }
            tmp = out_p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, out_p)
            ok += 1
            if ok % 10 == 0:
                print(f"[{model_id} shard{shard_id}] ok={ok}", flush=True)
        except Exception as e:
            fail += 1
            out_p.write_text(json.dumps({
                "sample_id": sid, "model_id": model_id, "ok": False,
                "error": repr(e), "traceback": traceback.format_exc()[-2000:],
            }, indent=2), encoding="utf-8")
            print(f"[fail] {sid}: {e}", flush=True)
    summary = {
        "model_id": model_id, "shard_id": shard_id, "num_shards": num_shards,
        "ok": ok, "skip": skip, "fail": fail, "missing_gen": miss,
        "hps_version": hps_version,
        "elapsed_sec": round(time.time() - t0, 1),
        "out_dir": str(out_dir),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    tag = f"_run_summary_shard{shard_id}.json" if num_shards > 1 else "_run_summary.json"
    (out_dir / tag).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=str(Path.home() / "mpie_testset_pack"))
    ap.add_argument("--model-id", default="")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--hps-version", default="v2.1")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
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
        print(f"=== hpsv2 {mid} shard {args.shard_id}/{args.num_shards} ===", flush=True)
        run_model(pack, mid, args.hps_version, resume, args.limit, args.shard_id, args.num_shards)


if __name__ == "__main__":
    main()
