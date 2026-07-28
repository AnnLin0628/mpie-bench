# Environment setup (evaluation)

This guide installs the **scoring** stack used by `code/eval/run_eval_e2e.sh`.
It does **not** install open-source image editors (optional; see
[`02_pipeline_design/eval_opensource.md`](02_pipeline_design/eval_opensource.md)).

You need:

- Linux + NVIDIA GPU (CUDA) recommended for ID / Qual / Mesh
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- An OpenAI-compatible chat API for **Count** and **Instruction** axes

## 1. Clone

```bash
git clone https://github.com/AnnLin0628/mpie-bench.git
cd mpie-bench
export MPIE_ROOT="$PWD"
export MPIE_TEST_PACK="$PWD/data/testset"
```

## 2. Create two conda environments

### 2a. `mpie` — ArcFace (ID), HPSv2 (Qual), VLM clients

```bash
conda create -y -n mpie python=3.10
conda activate mpie
pip install -U pip
pip install -r env/requirements-eval.txt
```

Download InsightFace **antelopev2** weights (once):

```bash
bash env/setup_eval.sh --arcface-weights
```

Download / place HPSv2 weights under `$MPIE_WEIGHTS/hpsv2/` (default
`~/mpie_weights/hpsv2`):

| File | Notes |
|------|--------|
| `HPS_v2.1_compressed.pt` | preferred (v2.0 `HPS_v2_compressed.pt` also works) |
| `open_clip_pytorch_model.bin` | OpenCLIP ViT-H-14 (laion2B) |

Upstream references:

- HPSv2: <https://github.com/tgxs002/HPSv2>
- InsightFace antelopev2 release zip:
  <https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip>

Helper (after weights are on disk):

```bash
export MPIE_WEIGHTS="${MPIE_WEIGHTS:-$HOME/mpie_weights}"
bash env/setup_eval.sh --check-hps
```

### 2b. `multihmr` — Multi-HMR mesh (Anatomy + Interaction)

```bash
conda create -y -n multihmr python=3.10
conda activate multihmr
pip install -U pip
pip install -r env/requirements-mesh.txt

# Clone Multi-HMR (naver) and install its requirements
git clone https://github.com/naver/multi-hmr.git "$HOME/models/multi-hmr"
cd "$HOME/models/multi-hmr"
pip install -r requirements.txt
# Follow upstream README for SMPL-X neutral model (SMPLX_NEUTRAL.npz)
cd "$MPIE_ROOT"
export MULTIHMR_REPO="${MULTIHMR_REPO:-$HOME/models/multi-hmr}"
```

## 3. Configure evaluation

```bash
cp configs/eval.env.example configs/eval.env
```

Edit at least:

```bash
export MPIE_TEST_PACK="$PWD/data/testset"
export MULTIHMR_REPO="$HOME/models/multi-hmr"
export MPIE_WEIGHTS="$HOME/mpie_weights"
export AI_GATEWAY_URL="https://<your-openai-compatible-host>/v1"
export AI_GATEWAY_KEY="sk-..."
export MPIE_JUDGE_MODEL="gpt-4.1"   # or your gateway's chat model id
```

## 4. Smoke-test scorers

Place a few generated images (or use the optional closed-source baselines; see
[`BASELINES.md`](BASELINES.md)), then:

```bash
EVAL_LIMIT=4 bash code/eval/run_eval_e2e.sh --model <model_id> --axes id,qual
# Mesh needs the multihmr env + MULTIHMR_REPO:
EVAL_LIMIT=4 bash code/eval/run_eval_e2e.sh --model <model_id> --axes mesh
# Count / Instr need the VLM gateway:
EVAL_LIMIT=4 bash code/eval/run_eval_e2e.sh --model <model_id> --axes count,instr
```

Full six-axis run:

```bash
bash code/eval/run_eval_e2e.sh --model <model_id>
```

## 5. What this repo does / does not ship

| Shipped | Not shipped |
|---------|-------------|
| N=2500 test pack (`data/testset`) | Open-source model generations |
| Scoring code + protocol docs | Full 10-model paper leaderboard dumps |
| Optional **3 closed-source** judgments (see [`BASELINES.md`](BASELINES.md)) | Cluster / private install scripts |

Model **outputs** (PNGs) for the optional closed-source baselines are published
separately as a release asset (too large for git). Judgments JSON for those
three models can live under `data/testset/judgments/`.
