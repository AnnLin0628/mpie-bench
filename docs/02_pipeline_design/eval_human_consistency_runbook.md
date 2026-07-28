# Anat/Inter Consistency Experiment · Runbook (Engineering)

Authoritative protocol: [`eval_human_consistency_anat_inter.md`](eval_human_consistency_anat_inter.md)  
Annotation guide: [`eval_human_consistency_annot_guide.md`](eval_human_consistency_annot_guide.md)

Python: `$HOME/miniconda3/bin/python` (system python3=3.6 is unusable)

## 0. One-shot self-check

```bash
cd "$MPIE_ROOT/code/eval"
python prep_consistency_selfcheck.py "$MPIE_TEST_PACK"
```

## 1. Subset (skip if already done)

```bash
python select_consistency_split.py --pack "$MPIE_TEST_PACK" --force
python build_annot_preview.py --pack "$MPIE_TEST_PACK"
```

Outputs: `$PACK/judgments/human_consistency/{_split,_protocol,_thresholds}.json`  
CSV: `annot_templates/` · Preview: `annot_preview/`

## 2. Human annotation

```bash
# Frontend (recommended): 3 annotator entry points ann_01 / ann_02 / ann_03
bash "$MPIE_ROOT/code/eval/run_annot_frontend.sh
# Open http://127.0.0.1:8080/

# After all three finish, consensus can be computed directly (JSON already under human/)
python import_human_checklist.py \
  --pack "$MPIE_TEST_PACK" --consensus
```

## 3. Mesh → M (requires mesh_v3)

```bash
# After syncing judgments/mesh_v3 from the GPU host to the local pack:
python export_mesh_checklist.py --pack "$MPIE_TEST_PACK" --force
```

After pilot H* is ready, lock τ:

```bash
python calibrate_checklist_thresholds.py \
  --pack "$MPIE_TEST_PACK" --split pilot --freeze
python export_mesh_checklist.py --pack "$MPIE_TEST_PACK" --force
```

## 4. VLM → V (same gateway as VQA, gpt-5.5)

```bash
# Credentials same as score_vlm_v1: AI_GATEWAY_URL + AI_GATEWAY_KEY
# Script auto-sources your env file if present
cd "$MPIE_ROOT/code/eval"
bash run_score_checklist_vlm.sh pilot 4
```

## 5. Generate κ tables

```bash
python compute_agreement.py \
  --pack "$MPIE_TEST_PACK" --split pilot --judge-model gpt-5.5
# → judgments/human_consistency/reports/tables_for_paper.md
```

## Current blockers

| Item | Status |
|----|------|
| Subset / CSV / preview / guide | ✅ ( **final HC: $N{=}170$, nine models balanced, C2--C3**; paper uses this) |
| All scripts | ✅ |
| mesh_v3 format | ✅ (flux/dreamo fields complete in full pack; mapping works directly) |
| mesh_v3 coverage | ⚠️ Only **flux ≈54** of 170 subset entries currently have mesh; gpt/gemini/seedream **do not yet** have mesh_v3 |
| Human annotation | ❌ Recruiting pending |
| Gateway key (for V) | ❌ Local `~/.mpie_env` lacks AI_GATEWAY_* |
