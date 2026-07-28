# Open-Source Evaluation Image Generation Guide

> Model inventory: [`eval_model_zoo.md`](eval_model_zoo.md) · Entry script: `code/eval/run_opensource_full_8gpu.sh`  
---

# Open-Source Baseline Image Generation

## One-Line Status (paper main-table 7)

| model_id | Smoke test LIMIT=2 | Key blockers & fixes (summary) |
|---|---|---|
| flux1-kontext-dev | ✅ | Terminal "hang" is log redirection; CLIP truncation can be ignored |
| dreamo | ✅ | Missing cv2/einops/timm; local symlink LoRA; BEN2+Turbo under mirror; facexlib weights on first run; Turbo `load_lora_weights` offline needs `weight_name` |
| omnigen2 | ✅ | `--offload model` |
| uno | ✅ | Offline missing CLIP/T5; path name must contain `clip`; use bf16 on-the-fly conversion when no standalone fp8 |
| ace | ✅ | Missing scepter/diffusers/peft; whole-model `.to(cuda)` OOM → sequential+768; offline LoRA needs `weight_name`; do not use `pillow_convert`; new diffusers `preprocess(None)` needs patch |
| bagel | ✅ | Missing `flash_attn` (cross-device link failure → install wheel directly) |
| firered | ✅ | conda `firered`; weights `FireRed-Image-Edit-1.1`; `--offload none --max-refs 3` on high-VRAM GPUs |

Extra (not in paper main table): `qwen-image-edit-2511` — `--offload sequential`; runner still available.

## Environment & Paths (Final)

```bash
export MPIE_TEST_PACK="${MPIE_TEST_PACK:-$PWD/data/testset}"   # manifest must have 2500 lines
export MPIE_WEIGHTS="${MPIE_WEIGHTS:-$HOME/mpie_weights}"
export MPIE_CODE="${MPIE_CODE:-$HOME/mpie_code}"

# ACE
export FLUX_FILL_PATH=~/mpie_weights/flux1-fill-dev
export SUBJECT_MODEL_PATH=~/mpie_weights/ace_plus/subject/comfyui_subject_lora16.safetensors

# UNO (offline text encoders; path name must contain clip / xflux naming works)
export CLIP=~/mpie_weights/clip-vit-large-patch14          # or a directory assembled from kontext text_encoder
export T5=~/mpie_weights/xflux_text_encoders               # or assembled from kontext text_encoder_2
export FLUX_DEV=~/mpie_weights/flux1-dev/flux1-dev.safetensors
export FLUX_DEV_FP8=~/mpie_weights/flux1-dev/flux1-dev.safetensors   # fall back to bf16 when no standalone fp8 file
export AE=~/mpie_weights/flux1-dev/ae.safetensors
export LORA=~/mpie_weights/uno/dit_lora.safetensors

# Usually enabled for full/smoke runs: avoid accidental network access (once local files are in place)
export HF_HUB_OFFLINE=1
```

| conda | Purpose |
|---|---|
| `mpie_edit` | kontext (+ optional qwen extra) |
| `omnigen2` / `uno` / `ace` / `bagel` / `dreamo` / `firered` | One env per model |
| launcher | **Auto** `conda activate`; no manual switching needed |

Logs: stdout goes to `/tmp/mpie_opensource_full_logs/<model_id>_shard*.log`; seeing only `[launch]` in the foreground is normal.

---

## Smoke Test Flow (Verified)

```bash
cd "$MPIE_ROOT/code/eval"
chmod +x run_opensource_full_8gpu.sh

# Models run serially; NGPU=1 for single GPU (paper main-table 7)
for m in kontext dreamo omnigen2 uno ace bagel firered; do
  echo "===== SMOKE $m ====="
  LIMIT=2 NGPU=1 bash run_opensource_full_8gpu.sh "$m" || {
    echo "FAIL $m"; tail -100 /tmp/mpie_opensource_full_logs/${m}*_shard0.log
    break
  }
done

for mid in flux1-kontext-dev dreamo omnigen2 uno ace bagel firered; do
  echo -n "$mid "; ls "$MPIE_TEST_PACK/outputs/$mid"/*.png 2>/dev/null | wc -l
done
# Target: 2 per model
```

Full run (after all smoke tests pass):

```bash
# Use the same shell with CLIP/T5/FLUX_* / ACE exports
# `all` = paper 7 open-source models (not Qwen)
nohup bash run_opensource_full_8gpu.sh all > /tmp/mpie_opensource_full_master.log 2>&1 &
```

Rough estimate: 7 models × 2500 × 8-GPU sharding, serial ~ **1–2 days** (ACE sequential is slower).

---

## Per-Model Pitfall Notes

### 0) Common Issues

1. **Foreground "hang"**: launcher redirects output to log; use `tail -f /tmp/mpie_opensource_full_logs/<mid>_shard0.log`.
2. **CLIP 77-token truncation / `187>77`**: common on FLUX family; T5 still handles long text — safe to ignore.
3. **`HF_HUB_OFFLINE=1`**: OK when local files are complete; must `unset HF_HUB_OFFLINE` (or `=0`) for downloads. Mirror: `export HF_ENDPOINT=https://hf-mirror.com`.
4. **Offline `load_lora_weights`**: must pass explicit `weight_name` (ACE Turbo/subject, DreamO Turbo both hit this).
5. **pack**: full run uses the official N=2500 pack (`data/testset`).

### 1) flux1-kontext-dev / omnigen2 / firered (+ optional qwen)

- kontext: loading + fp8 quant takes several minutes on first run; ~1 min per sample.
- omnigen2: `--offload model`.
- firered: `bash run_opensource_full_8gpu.sh firered`; weights dir `FireRed-Image-Edit-1.1`.
- optional qwen (not paper main table): `--offload sequential` (24G GPU).

### 2) UNO

**Symptom A**: `Cannot find ... huggingface.co` / offline, crash in `load_clip`.  
**Cause**: UNO defaults to HF ids; offline environment.  
**Fix**:

```bash
# Assemble text towers from kontext diffusers with "clip" in the path
# ⚠️ HFEmbedder: is_clip = "clip" in path.lower()
#    Pointing directly at text_encoder (no "clip" in name) is treated as T5 → many MISSING keys
mkdir -p ~/mpie_weights/clip-vit-large-patch14 ~/mpie_weights/xflux_text_encoders
cp -a ~/mpie_weights/flux1-kontext-dev/text_encoder/.  ~/mpie_weights/clip-vit-large-patch14/
cp -a ~/mpie_weights/flux1-kontext-dev/tokenizer/.     ~/mpie_weights/clip-vit-large-patch14/
cp -a ~/mpie_weights/flux1-kontext-dev/text_encoder_2/. ~/mpie_weights/xflux_text_encoders/
cp -a ~/mpie_weights/flux1-kontext-dev/tokenizer_2/.    ~/mpie_weights/xflux_text_encoders/
export CLIP=~/mpie_weights/clip-vit-large-patch14
export T5=~/mpie_weights/xflux_text_encoders
```

**Symptom B**: no `flux1-dev.fp8.safetensors`.  
**Fix**: point `FLUX_DEV_FP8` at bf16 `flux1-dev.safetensors`; UNO officially **converts bf16→fp8 on the fly** (log explains; conversion takes a few minutes).

**False alarm**: `conda run -n uno -c "from uno..."` missing modules — needs `PYTHONPATH=~/mpie_code/UNO`; runner handles this.

### 3) ACE++

| Stage | Error | Fix |
|---|---|---|
| import | `No module named 'scepter'` | `pip install scepter` (and oss2/aliyun if stuck) |
| import | `No module named 'diffusers'` | `pip install diffusers transformers accelerate ...` |
| import | `peft>=0.17 required, found 0.6.2` | `pip install -U "peft>=0.17.0"` |
| load | `CUDA OOM` on `.to(cuda)` | change source to `enable_sequential_cpu_offload()`; smoke test `--size 768` |
| infer | `offline ... must specify weight_name` | `load_lora_weights(dir, weight_name=basename)` |
| infer | `Input is in incorrect format` (PIL…) | ① do not use `pillow_convert`, use plain `PIL.Image`; ② new FluxFill still calls `preprocess(image=None)` when only `masked_image_latents` is passed — patch `image_processor.preprocess`, `None` → zero tensor |
| infer | model offload still OOM at step 0/28 | switch to **sequential** + **size 768** (512 if needed) |

Key ACE source patches: `~/mpie_code/ACE_plus/inference/ace_plus_diffusers.py`  
(remove `.to(we.device_id)`; LoRA `weight_name`; `preprocess(None)` fallback.)

`run_ace.py` supports `--offload sequential|model|none` and offline LoRA wrapper; if runners are older, source patches take precedence. Launcher recommendation: `--offload sequential --size 768`.

### 4) BAGEL

**Symptom**: `No module named 'flash_attn'`.  
**Background**: `setup_bagel_dreamo.sh` intentionally skipped flash-attn.  
**Fix** (torch=`2.5.1+cu124`, py310):

```bash
# If pip build hits Invalid cross-device link: skip pip cache across disks, download wheel directly
cd /tmp
wget -c https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3.post1/flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
conda activate bagel
pip install --no-deps ./flash_attn-2.8.3.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

First startup loading `ema.safetensors` + offload can take tens of minutes; `safetensors ... no metadata` warnings are safe to ignore. Params: `--max-mem 20GiB`, offload dir `/tmp/bagel_offload`.

### 5) DreamO

**Dependencies (previously missing in dreamo env)**: `opencv-python-headless` · `einops` · `timm` · plus non-torch items from `requirements.txt` (do not blindly `pip install torch==pinned` and overwrite existing CUDA wheels).

**Weight layout** (cwd=`~/mpie_code/DreamO`):

```bash
mkdir -p models/v1.1 models/black-forest-labs
ln -sfn ~/mpie_weights/flux1-dev models/black-forest-labs/FLUX.1-dev

W=~/mpie_weights/DreamO   # official safetensors already present (not comfyui-only)
ln -sfn "$W/dreamo.safetensors" models/
ln -sfn "$W/dreamo_cfg_distill.safetensors" models/
ln -sfn "$W/dreamo_quality_lora_pos.safetensors" models/
ln -sfn "$W/dreamo_quality_lora_neg.safetensors" models/
ln -sfn "$W/v1.1/dreamo_sft_lora.safetensors" models/v1.1/
ln -sfn "$W/v1.1/dreamo_dpo_lora.safetensors" models/v1.1/
```

**Additional downloads (mirror; disable offline first)**:

```bash
unset HF_HUB_OFFLINE; export HF_HUB_OFFLINE=0
export HF_ENDPOINT=https://hf-mirror.com
# BEN2 + Turbo → models/BEN2_Base.pth · models/diffusion_pytorch_model.safetensors
```

Note: `huggingface-hub>=1` breaks legacy `huggingface-cli`; when conflicting with `transformers==4.45`, keep `huggingface_hub>=0.23,<1` and use `hf download` or `hf_hub_download`.

**facexlib (first run only)**: startup pulls GitHub weights to  
`.../envs/dreamo/lib/python3.11/site-packages/facexlib/weights/`  

- `detection_Resnet50_Final.pth` (v0.1.0)  
- `parsing_parsenet.pth` (v0.2.2)  

Download once and persist; **not re-downloaded on every run**. For slow GitHub, prefix with `ghfast.top` / `mirror.ghproxy.com`.

**Turbo offline LoRA**:

```python
# dreamo/dreamo_pipeline.py · use_turbo branch
self.load_lora_weights(
    os.path.dirname(turbo_path) or "models",
    weight_name=os.path.basename(turbo_path),
    adapter_name="turbo",
)
```

`Couldn't access the Hub... Defaulting to existing file` = local file hit — safe to ignore.

**Full 8-GPU run**: `int8` quantizes on CPU/RAM; 8 simultaneous starts OOM (shard gets `Killed`). Launcher staggers dreamo by default: start next GPU after previous log shows `dreamo ready`; **inference phase still runs 8 processes in parallel**. Do not run bare `NGPU=8` without stagger.

---

## Pre–Full-Run Checklist

- [ ] `wc -l $MPIE_TEST_PACK/manifest.jsonl` → 2500  
- [ ] All 7 models smoke-test to 2 pngs each  
- [ ] UNO: `CLIP`/`T5`/`FLUX_DEV_FP8` exported (path contains `clip`)  
- [ ] ACE: sequential offload + size 768 in source or launcher  
- [ ] DreamO: LoRA+BEN2+Turbo+FLUX chain under `models/`; facexlib weights present; turbo `weight_name` patch; **staggered loading** (`STAGGER_READY_RE`)  
- [ ] bagel: `flash_attn` import OK  
- [ ] Full-run `nohup` shell has all exports; `HF_HUB_OFFLINE=1`  
- [ ] Disk: `df -h /home` (was 99% full — watch offload/output space)

## Packaging Results

```bash
cd "$MPIE_TEST_PACK"
tar czf ~/mpie_full_outputs_opensource7_$(date +%Y%m%d).tgz \
  outputs/flux1-kontext-dev outputs/dreamo \
  outputs/omnigen2 outputs/uno outputs/ace outputs/bagel outputs/firered
```

## Related Files

| Path | Description |
|---|---|
| `code/eval/run_opensource_full_8gpu.sh` | 8-GPU launcher |
| `code/eval/opensource/run_*.py` | Per-model runners |
| `code/eval/check_opensource_paths.sh` | Path inventory check |
| [`eval_model_zoo.md`](eval_model_zoo.md) | Model inventory status |

---

## Shared Storage / FUSE Notes

If weights or pack live on network storage / FUSE (not local NVMe):

- Multiple processes **mmap-loading** large weights can saturate I/O; processes stall in uninterruptible disk wait.
- Stagger loading: `STAGGER_READY_RE='pipe ready'` or `STAGGER_SEC=60`; run all GPUs in parallel during inference.
- Point paths via environment variables, e.g.:

```bash
export MPIE_TEST_PACK=/path/to/shared/data/mpie_testset_pack
export MPIE_WEIGHTS=/path/to/shared/models/mpie
export MPIE_BENCH_EVAL=$PWD/opensource   # when under code/eval
HIGH_VRAM=1 NGPU=8 bash run_opensource_full_8gpu.sh kontext
```

With `HIGH_VRAM=1`, launcher enables `pipe ready` stagger for kontext/qwen by default and can disable fp8 / adjust offload (see script comments).
