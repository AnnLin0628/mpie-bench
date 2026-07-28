# MPIE-Bench Evaluation Protocol v3

> Main table: **Anat / Inter from Multi-HMR mesh**; Count / Instr from VLM; ID=ArcFace; Qual=HPSv2.  
> Contact density: C0–C3 ([contact_density_c3_merge.md](contact_density_c3_merge.md)).  
> Specs: [`eval_mesh_inter_spec.md`](eval_mesh_inter_spec.md) · [`eval_mesh_anat_spec.md`](eval_mesh_anat_spec.md).  
> Entrypoint: [`run_eval_e2e.sh`](../code/eval/run_eval_e2e.sh) ([eval_e2e.md](eval_e2e.md)).

Index: [`README.md`](README.md).

---

## 0. How the Main Table Scores

| Axis | Method |
|------|--------|
| **Count** | VLM judge |
| **ID** | ArcFace ([`eval_id_protocol.md`](eval_id_protocol.md)) |
| **Anat** | **Multi-HMR mesh** → `S_anat_mesh` |
| **Inter** | **Multi-HMR mesh** → `S_inter_mesh` |
| **Instr** | Frozen Instr QA bank (`instr_qa_v2/`) scored by VLM |
| **Qual** | HPSv2 (raw scores typically 0.20–0.35; **no Overall**) |

### When Main Table Writes Scores (Override Rules)

- Default: axis mean written only when valid samples ≥ **~90% of pack** (2500 → ≥2250), avoid mid-run subset inflation.  
- **Exception (2026-07-18)**: output count ≥ **~80% of pack**, and axis scored on **all available outputs** → treat as complete.  
  Example: **gpt-image-2 outputs 2078**, ID/Qual/Mesh shown at full quota.  
- Implementation: `code/eval/aggregate_mesh_v3.py` → `_coverage_ok`.

### Design Rationale (Six Points)

1. **Who to score = prompt `R#`** → Multi-HMR top-k keep, avoid bystanders.  
2. **Contact or not = prompt keywords** (`required` / `forbidden` / `unspecified`), not dataset `cat` (future gold labels/classifier).  
3. **Inter** = 3D penetration volume + surface distance, additive compose by intent (detailed spec).  
4. **Anat skips SMPL prior** (≈0.99 saturation) → image-side “human mass skeleton cannot explain”.  
5. **Extra hands etc. via unified leftover**, no per-part counting; skin-tone bleeding still VQA.  
6. **Additive scoring (main formula)**: `S = Σ w_i s_i`, `s_i = 1 − P_i`; numerically equivalent to `1 − Σ w_i P_i`; diagnostics still persist `P_*`.

```text
n_expected = |{Rk in prompt}|
keep       = Multi-HMR score top-k, k = n_expected
```

`j2d` on 896 canvas; must `j2d_canvas_to_original` before image metrics.

---

## 1. Inter (Summary → Detailed Spec)

Prompt involved in two places only: ① count R# for selection; ② keywords for `contact_intent`.

```text
s_pen / s_prox / s_clear  ← volume and distance + intent (s = 1 − P)
S_inter_mesh = Σ w s        # required: 0.55 s_pen + 0.45 s_prox; ×0.5 when under_detect
```

Full symbols, ramp, calibration, examples: [`eval_mesh_inter_spec.md`](eval_mesh_inter_spec.md).  
Known: closed-source with less penetration can have Inter **> GT**; intent keywords are weak heuristic.

---

## 2. Anat (Summary → Detailed Spec)

```text
leftover → orphan → s_extra = 1 − max(P_area, P_blob)
S_anat_mesh = 0.40 s_extra + 0.30 s_resid + 0.15 s_struct + 0.15 s_detect
```

Full formulas, meanings, design rationale, example image: [`eval_mesh_anat_spec.md`](eval_mesh_anat_spec.md).

---

## 3. Persistence and Code

```text
$PACK/judgments/mesh_v3/<model_id|/ _gt>/<sample_id>.json
$PACK/judgments/arcface_v1/<model_id>/
$PACK/judgments/hpsv2/<model_id>/
```

| File | Role |
|------|------|
| `mesh_metrics.py` | Inter |
| `anat_extended.py` | Anat leftover / compose |
| `score_mesh_v3.py` | batch write json |
| `score_arcface_v1.py` / `score_hpsv2.py` | ID / Qual |
| `rescore_mesh_inter.py` / `rescore_mesh_anat.py` | offline rescore |
| `run_eval_e2e.sh` | **Public entrypoint**: Count→Instr→ID→Qual→Mesh→aggregate ([eval_e2e.md](eval_e2e.md)) |
| `run_model_eval_full_8gpu.sh` | Legacy multi-GPU helper: ID→Qual→Mesh only |
| `aggregate_mesh_v3.py` | Leaderboard HTML/JSON |

```bash
python aggregate_mesh_v3.py \
  --pack "$PWD/data/testset" \
  --out "$PWD/data/eval_outputs/latest"
```

Before mesh re-run, delete stubs with image but `ok:false`, else resume skips those samples.

---

## 4. Scope Boundaries

Photorealistic domain; top-k selection is not identity alignment to R1/R2; penetration volume is approximate; leftover is a 2D proxy; Inter above GT does not imply better interaction semantics.
