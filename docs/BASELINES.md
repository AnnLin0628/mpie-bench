# Optional closed-source baselines

This repository’s **public claim** is the test pack + scoring code. We optionally
publish artifacts for **three closed-source editors only**:

| Model id | Role |
|----------|------|
| `gpt-image-2` | Closed-source baseline |
| `gemini-3-pro-image` | Closed-source baseline |
| `seedream-5-pro` | Closed-source baseline |

Open-source model generations and their judgment dumps are **not** published.
That is intentional: outsiders can verify the protocol on these three systems
(or on their own models) without obtaining a full multi-model leaderboard dump.

## Layout

```text
data/testset/
  outputs/<model_id>/<sample_id>.png     # PNGs: via release download (large)
  judgments/
    vlm_judge_v1/<model_id>/*.json       # Count (and legacy VLM fields)
    instr_v2/<model_id>/*.json
    arcface_v1/<model_id>/*.json
    hpsv2/<model_id>/*.json
    mesh_v3/<model_id>/*.json            # Anat + Inter
    mesh_v3/_calibration.json
```

Judgment JSON for the three models is in-repo under `data/testset/judgments/`.

PNG outputs live on the Git LFS branch `assets/closed3` (too large for `main`).
Fetch them with:

```bash
# requires git-lfs
bash scripts/fetch_closed_outputs.sh
```

## Re-aggregate the three models

```bash
export MPIE_TEST_PACK="$PWD/data/testset"
python code/eval/aggregate_mesh_v3.py \
  --pack "$MPIE_TEST_PACK" \
  --models gpt-image-2 gemini-3-pro-image seedream-5-pro \
  --out "$PWD/data/testset/baselines/closed3"
```

## Re-score from PNGs (optional)

After outputs are in place and environments are installed ([`INSTALL.md`](INSTALL.md)):

```bash
bash code/eval/run_eval_e2e.sh --model gpt-image-2
bash code/eval/run_eval_e2e.sh --model gemini-3-pro-image
bash code/eval/run_eval_e2e.sh --model seedream-5-pro
```

Slight numeric drift is possible across GPU / library versions; the shipped
judgment JSON is the frozen reference for these three baselines.
