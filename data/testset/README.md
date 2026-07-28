# MPIE-Bench test set (N=2500)

Official evaluation pack: reference / GT images, sample manifest, and frozen
Instr v2 questions. **No model outputs or judgment score dumps.**

## Layout

```text
manifest.jsonl      # 2500 samples (prompt, refs, GT paths)
pack_meta.json      # pack metadata
images/<cat>/<id>/  # R* reference crops + GT_*.jpg
instr_qa_v2/        # frozen Instr v2 questions (one JSON per sample)
gallery/            # optional local browser for refs/GT
```

## Quick use

From the repository root:

```bash
export MPIE_TEST_PACK="$PWD/data/testset"
cp configs/eval.env.example configs/eval.env
# edit configs/eval.env, then:
bash code/eval/run_eval_e2e.sh --model <model_id>
```

Place your model generations at:

```text
$MPIE_TEST_PACK/outputs/<model_id>/<sample_id>.png
```

## Optional baselines

Frozen judgments for **three closed-source** models may be present under
`judgments/` (`gpt-image-2`, `gemini-3-pro-image`, `seedream-5-pro`). See
[`docs/BASELINES.md`](../../docs/BASELINES.md). Open-source dumps are not
published. PNG outputs for those three models are fetched separately
(`scripts/fetch_closed_outputs.sh`).

## What is not included

| Path | Why omitted |
|------|-------------|
| Full `outputs/` for all models | Generations are submitter-produced / release assets |
| Open-source judgments | Not published (closed-source-only optional dumps) |

Source video used during construction is not redistributed here.
