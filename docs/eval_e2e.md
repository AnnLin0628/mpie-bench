# End-to-end evaluation (protocol v3)

One entrypoint scores all **six** main-table axes after you place generated images in the pack.

| Axis | Scorer | Needs |
|------|--------|-------|
| **Count** | `score_vlm_v1.py` | OpenAI-compatible VLM API |
| **Identity** | `score_arcface_v1.py` | conda env with InsightFace (GPU recommended) |
| **Anatomy** | `score_mesh_v3.py` | Multi-HMR env + `MULTIHMR_REPO` |
| **Interaction** | same mesh run | same |
| **Instruction** | `score_instr_v2.py` | VLM API + frozen QA bank (`instr_qa_v2/`) |
| **Quality** | `score_hpsv2.py` | HPSv2 weights in the eval env |

Orchestrator: [`code/eval/run_eval_e2e.sh`](../code/eval/run_eval_e2e.sh).

## 1. Prepare

```bash
git clone https://github.com/AnnLin0628/mpie-bench.git
cd mpie-bench

# Full conda/pip + weight steps: ../INSTALL.md
#   conda env mpie      → ArcFace / HPSv2 / VLM clients
#   conda env multihmr  → Multi-HMR Anat+Inter
bash env/setup_eval.sh --arcface-weights   # after pip install -r env/requirements-eval.txt

cp configs/eval.env.example configs/eval.env
# Edit: MPIE_TEST_PACK, MULTIHMR_REPO, AI_GATEWAY_URL, AI_GATEWAY_KEY, NGPU, …
```

Pack layout (minimum):

```text
$MPIE_TEST_PACK/
  manifest.jsonl          # or equivalent pack manifest used by pack_io
  images/ …               # GT / refs as used by the pack
  outputs/<model_id>/*.png
  instr_qa_v2/            # optional; built on first Instr run if missing
  judgments/              # written by scorers
```

Put your model’s generations at `outputs/<model_id>/` with filenames aligned to sample ids (same convention as the baseline runners).

## 2. One command

```bash
export MPIE_EVAL_ENV_FILE=$PWD/configs/eval.env   # optional if file is at default path
bash code/eval/run_eval_e2e.sh --model my-model
```

This runs Count → Instr → ID → Qual → Mesh(Anat+Inter) → aggregate.

Outputs:

- Per-axis JSON under `$MPIE_TEST_PACK/judgments/...`
- Leaderboard HTML/JSON at `$MPIE_EVAL_OUT` (default `data/eval_outputs/latest/`)

## 3. Common variants

```bash
# Subset of axes (e.g. local GPU metrics only)
bash code/eval/run_eval_e2e.sh --model my-model --axes id,qual,mesh

# Multi-GPU shards for ID / Qual / Mesh
NGPU=8 bash code/eval/run_eval_e2e.sh --model my-model

# Score every folder under outputs/
bash code/eval/run_eval_e2e.sh --all-models

# Rebuild leaderboard only
bash code/eval/run_eval_e2e.sh --model my-model --aggregate-only

# Smoke on first N samples
EVAL_LIMIT=20 bash code/eval/run_eval_e2e.sh --model my-model
```

## 4. Generation (optional, before scoring)

Open-source baseline runners live under `code/eval/opensource/` and `run_opensource_full_8gpu.sh`. Closed-source: `code/eval/closedsource/`. Generation is **not** folded into `run_eval_e2e.sh` so you can bring any model that writes the pack layout.

## 5. Resume and failures

All scorers support resume (skip existing judgment JSON). Re-run the same command after a crash. Logs: `/tmp/mpie_eval_e2e/<model_id>/`.

If Count/Instr fail, check `AI_GATEWAY_URL` + `AI_GATEWAY_KEY` (or `AI_GATEWAY_KEY`). If Mesh fails, check `MULTIHMR_REPO` and conda env `multihmr`.

## 6. Protocol authority

Score definitions and coverage rules: [eval_protocol_v3.md](eval_protocol_v3.md).
