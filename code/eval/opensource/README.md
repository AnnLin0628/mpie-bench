# Open-Source Model Batch Generation on the Test Set

Generate PNGs into the official pack layout, then score with the six-axis E2E pipeline.

**Model inventory:** [`docs/eval_model_zoo.md`](../../../docs/eval_model_zoo.md)  
**Generation notes:** [`docs/eval_opensource.md`](../../../docs/eval_opensource.md)  
**Scoring:** [`docs/eval_e2e.md`](../../../docs/eval_e2e.md)  
Closed-source entry: `../closedsource/`.

## Pack Layout

```
$MPIE_TEST_PACK/   # default: repo data/testset
  manifest.jsonl
  images/<cat>/<anchor>/R*.jpg , GT_*.jpg
  outputs/<model_id>/<sample_id>.png
  outputs/<model_id>/_meta/<sample_id>.json
  judgments/<metric>/<model_id>/<sample_id>.json
```

## Paper main table (7 open-source)

| model_id | conda env | weights (under `$MPIE_WEIGHTS`) | runner |
|---|---|---|---|
| flux1-kontext-dev | mpie_edit | flux1-kontext-dev | run_kontext.py |
| dreamo | dreamo | DreamO + flux1-dev | run_dreamo.py |
| omnigen2 | omnigen2 | omnigen2 | run_omnigen2.py |
| uno | uno | flux1-dev + uno | run_uno.py |
| ace | ace | flux1-fill-dev + ace_plus | run_ace.py |
| bagel | bagel | BAGEL-7B-MoT | run_bagel.py |
| firered | firered | FireRed-Image-Edit-1.1 | run_firered.py |

## Full 2500 × multi-GPU

```bash
export MPIE_TEST_PACK="$PWD/data/testset"
export MPIE_WEIGHTS="${MPIE_WEIGHTS:-$HOME/mpie_weights}"
cd code/eval
bash run_opensource_full_8gpu.sh all          # paper open-7
# or one model:
bash run_opensource_full_8gpu.sh firered
```
