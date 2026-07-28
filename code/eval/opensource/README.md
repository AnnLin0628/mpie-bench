# Open-Source Model Batch Generation on the Test Set

Pipeline: SG reverse-captions prompts → `export_pack.py` builds pack → sync to high-VRAM → run per-model runners → tar `outputs/` back to SG.

**Unified open/closed layout**: [`../RESULT_LAYOUT.md`](../RESULT_LAYOUT.md) (required reading). Closed-source entry: `../closedsource/`, `../run_full_closed.sh`.

## Pack Layout (summary)

```
$PACK/   # smoke100 or full set
  manifest.jsonl
  pack_meta.json
  images/<cat>/<anchor>/R*.jpg , GT_*.jpg
  outputs/<model_id>/<sample_id>.png          # same rule for open/closed
  outputs/<model_id>/_meta/<sample_id>.json
  judgments/vlm_judge_v1/<model_id>/<sample_id>.json
```

After image generation is complete, see **full-set VLM/Agent batch judging** in `docs/02_pipeline_design/eval_vlm_judge_v1.md`.

## Model ↔ Environment ↔ Weights (7 models, docs frozen)

| model_id | conda env | weights dir | runner |
|---|---|---|---|
| flux1-kontext-dev | mpie_edit | ~/mpie_weights/flux1-kontext-dev | run_kontext.py |
| qwen-image-edit-2511 | mpie_edit | ~/mpie_weights/qwen-image-edit-2511 | run_qwen_edit.py |
| omnigen2 | omnigen2 | ~/mpie_weights/omnigen2 | run_omnigen2.py |
| uno | uno | flux1-dev + uno | run_uno.py |
| ace | ace | flux1-fill-dev + ace_plus | run_ace.py |
| bagel | bagel | BAGEL-7B-MoT | run_bagel.py |
| dreamo | dreamo | DreamO + flux1-dev | run_dreamo.py |

## Full 2500 × multi-GPU

See parent [`RUN_OPENSOURCE_FULL.md`](../RUN_OPENSOURCE_FULL.md) and `run_opensource_full_8gpu.sh all`.
