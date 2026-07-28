# Evaluation and Pipeline Design

This directory holds MPIE-Bench evaluation protocols, baseline notes, and data-construction essentials. See the master index at [`../README.md`](../README.md).

## Read by Task

| If you need to… | Read |
|--------|-----|
| **End-to-end scoring (recommended entry)** | [eval_e2e.md](eval_e2e.md) |
| **Current main-table protocol (v3)** | [eval_protocol_v3.md](eval_protocol_v3.md) |
| **Core claim: Anat/Inter construct validity** | [eval_construct_validity_principle.md](eval_construct_validity_principle.md) |
| **Human consistency / construct checklist wording** | [eval_human_consistency_anat_inter.md](eval_human_consistency_anat_inter.md) |
| **Analysis protocol (S/U/gold labels/blind spots)** | [eval_human_consistency_analysis_protocol.md](eval_human_consistency_analysis_protocol.md) |
| **Human consistency runbook** | [eval_human_consistency_runbook.md](eval_human_consistency_runbook.md) |
| **Contact density C0–C3** | [contact_density_c3_merge.md](contact_density_c3_merge.md) |
| Inter formulas / intent | [eval_mesh_inter_spec.md](eval_mesh_inter_spec.md) |
| Anat formulas / residual signals | [eval_mesh_anat_spec.md](eval_mesh_anat_spec.md) |
| Identity (ArcFace) | [eval_id_protocol.md](eval_id_protocol.md) |
| Model list | [eval_model_zoo.md](eval_model_zoo.md) |
| Open-source image generation | [eval_opensource.md](eval_opensource.md) |
| Test-set split | [testset_split.md](testset_split.md) |
| Six-step data pipeline | [standard_process.md](standard_process.md) |
| Lessons from related benchmarks | [bench_lessons_from_sota.md](bench_lessons_from_sota.md) |

Scoring and runner code live under `code/eval/`. Configure paths via `MPIE_TEST_PACK` / `MPIE_WEIGHTS` / `MULTIHMR_REPO` (see `configs/eval.env.example`).
