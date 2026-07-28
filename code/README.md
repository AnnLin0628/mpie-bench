# MPIE-Bench code

## Environment

Follow [`docs/INSTALL.md`](../docs/INSTALL.md):

```bash
conda create -y -n mpie python=3.10 && conda activate mpie
pip install -r ../env/requirements-eval.txt
bash ../env/setup_eval.sh --arcface-weights

conda create -y -n multihmr python=3.10 && conda activate multihmr
pip install -r ../env/requirements-mesh.txt
# clone naver/multi-hmr and set MULTIHMR_REPO
```

```bash
export MPIE_TEST_PACK="$PWD/../data/testset"
export MPIE_WEIGHTS="${MPIE_WEIGHTS:-$HOME/mpie_weights}"
export MULTIHMR_REPO="${MULTIHMR_REPO:-$HOME/models/multi-hmr}"
```

## Evaluation (six axes, end-to-end)

```bash
cp ../configs/eval.env.example ../configs/eval.env   # edit paths + API key
bash eval/run_eval_e2e.sh --model <model_id>
```

Guide: [`docs/eval_e2e.md`](../docs/eval_e2e.md).  
Optional closed-source baselines: [`docs/BASELINES.md`](../docs/BASELINES.md).

## Data pipeline (optional; for extending the benchmark)

```bash
DB=/path/to/mpie.db

python pipeline/02_extract_frames/shot_split_extract.py \
  --videos /path/to/raw_video --dataset <name> --license <tag> \
  --out /path/to/mpie_data --db $DB --fps 4

python pipeline/03_coarse_scan/coarse_scan.py --frames /path/to/frames --db $DB
python pipeline/04_peak_refine/interaction_density.py --db $DB
python pipeline/05_track_identity/identity_cluster.py --db $DB --out /path/to/crops
python pipeline/05_track_identity/identity_cluster.py --db $DB --out /path/to/crops --cluster
python pipeline/06_ref_crop/ref_crop.py --db $DB --out /path/to/crops
python pipeline/07_caption/caption_client.py --db $DB
python common/make_splits.py --db $DB --test-size 2500 --val-size 500 --dry-run
```

## Baseline generation (optional)

Paper open-source zoo (7): `kontext` `dreamo` `omnigen2` `uno` `ace` `bagel` `firered`  
(see [`docs/eval_model_zoo.md`](../docs/eval_model_zoo.md)).

```bash
cd eval
LIMIT=2 NGPU=1 bash run_opensource_full_8gpu.sh kontext
# bash run_opensource_full_8gpu.sh all   # paper open-7
# Closed-source APIs — see eval/closedsource/README.md
```
