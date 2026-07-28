# Anat / Inter Construct Validity Experiment: Human Preference Anchor · Mesh · VLM

> **Status: 2026-07-18 plan v5.1 (ordered severity + human all Inter items mandatory)** · Follows core principles  
> 　[`eval_construct_validity_principle.md`](eval_construct_validity_principle.md) (**required reading**)  
> 　[`eval_human_consistency_analysis_protocol.md`](eval_human_consistency_analysis_protocol.md) (**peer to checklist**: S / U·null / gold / blind spots / §0 human hard rules)  
> **Protocol ID**: `checklist_anat_inter_v5` · analysis `anat_inter_analysis_v5` · scheme **v5.1** / `mesh_bin_map_v5`  
> **Mesh initial version**: Anat / Inter continuous scores **v0.1** (evaluation dashboard main-table continuous scores unchanged)  
>
> **Core questions (non-negotiable)**  
> 1. Does the geometry system track real human preference (main paper: `Q*` ↔ continuous `S_*_mesh`)?  
> 2. On the same construct, is V inflated and less aligned with H?  
> 3. Are checklist items (mesh-inspired) validated by **independent human answers**—**forbidden** to hide items via system intent.
>
> **⚠ Annotators / guide review must use only these two sources, not legacy tables below:**  
> 1. Frontend 8080 item stems + `GUIDELINES.md` in pack  
> 2. [`eval_human_consistency_analysis_protocol.md`](eval_human_consistency_analysis_protocol.md)  
> Any I2/I4/A0/A6 etc. appearing below are **v2/v3 historical residue, deprecated**.

---

## 0. Experiment Overview

### 0.1 Three Judges (Same Sample Sheet)

| Code | Judge | How 0/1 Is Produced |
|------|-------|---------------------|
| **H** | Human annotator | View image + prompt, fill checklist item by item 0/1/U |
| **M** | Mesh initial version (geometry) | Continuous intermediates → **threshold mapping** to the same checklist (humans do not compute volume) |
| **V** | VLM / LLM-as-judge | **Same checklist JSON as humans** (not the legacy six-axis free-form list) |

```text
                    Same Checklist (Anat + Inter)
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   Human (H)            Mesh→bin (M)           VLM (V)
   3-person majority     τ mapped from mesh_v3   Same prompt template
        │                     │                     │
        └──────────┬──────────┴──────────┬──────────┘
                   ▼                     ▼
            Human–human IAA          Pairwise agreement
         (Fleiss / α)     H↔M · H↔V · M↔V
                          κ / F1 / confusion matrix
```

### 0.2 Main Comparisons (Paper Narrative)

| Priority | Comparison | Meaning |
|----------|------------|---------|
| **P0** | **H ↔ M** | Does the 3D mesh initial version track humans? |
| **P0** | **H ↔ V** | Does VLM on the checklist track humans? |
| P1 | M ↔ V | Do the two automatic judges disagree? |
| P1 | Human–human | Is annotation reliable? (otherwise P0 is meaningless) |

**Gold (main paper)**: Human item-level **mean of 10 scores then round**; axis-level pass from AND of main items. Mesh→item mapping see analysis protocol §7b (item_map_v6).  
**Not done**: Ask humans to reproduce Multi-HMR / `pen_volume`; do not use Likert 1–5 as the primary protocol.

### 0.3 Relation to Legacy VLM Judge v1

| | Plan ① `score_vlm_v1` | This experiment **V** |
|--|----------------------|------------------------|
| Output | Six mixed axes + open-set `anat_errors` list | **Only** Anat/Inter checklist fields |
| Inter | semantic / contact / penetration three levels | Same items as human **I0–I5** |
| Anat | Error list → anat_pass | Same items as human **A1–A5** (v4) |
| Use | Historical main table / appendix | **Dedicated** alignment with H/M |

Legacy v1 results may appear in appendix; **do not replace** this experiment's V.

---

## 1. Literature and Design Principles

| Practice | Basis | How We Use It |
|----------|-------|---------------|
| Atomic Yes/No | Otani et al. CVPR'23 (concrete items → higher IAA) | Each checklist item is one observable fact |
| Binary + Fleiss κ | TRUE / text–image alignment annotation | 0/1; report κ and α |
| Checklist / VQA decomposition | TIFA, DSG, composition checklist | Anat/Inter sub-items |
| Anatomy binary artifact | ASAP / HAF-Bench human eval | A1 and multi-limb items |
| LLM-as-judge report κ+confusion matrix | Binary judge eval convention | H↔V same table |
| Threshold calibration | Standard continuous→binary flow | M's τ locked in pilot |

**Hard principles**

1. **Same item stems and encoding for all three judges** (1=pass/no error, 0=fail/has error, U=cannot judge).  
2. Humans and VLM **view image (+prompt) only**; M **reads mesh persisted fields only** (visualizations may be stored separately, but not in the annotation UI).  
3. Stabilize **human–human** first, then H↔M / H↔V.  
4. Contact intent must be shown; in formal phase **intent gold** takes priority over keywords.

---

## 2. Research Questions and Success Thresholds

| ID | Question |
|----|----------|
| Q1 | Is human–human checklist reliable? |
| Q2 | **Is M close to H?** (primary) |
| Q3 | **Is V close to H?** (primary) |
| Q4 | Which is closer to H, M or V? Which items are stronger/weaker? |
| Q5 | Is axis-level binary `1[S_mesh≥τ]` worse than item-level mapping? |

| Metric | Suggested threshold (adjustable after pilot) |
|--------|-----------------------------------------------|
| Human–human item-level **Krippendorff α** (includes U; primary) | ≥ 0.40; I1/A1/Ic target ≥ 0.50 |
| Human–human Fleiss κ (appendix, U removed) | For reference, not primary threshold |
| H↔M construct correlation (stratified by intent) | Spearman primary; appendix reports κ |
| Very low human–human | **Fix item stems/diagrams first**, do not reject mesh first |

---

## 3. Unified Construct Measurement (Human UI = VLM Checklist) · **v5.1**

> **v2–v4.1 deprecated for primary analysis.** Below is v5.1. Principles in [`eval_construct_validity_principle.md`](eval_construct_validity_principle.md).

### 3.0 v5.1 Checklist Quick Reference

| ID | Construct role | Encoding | **Human response** |
|----|----------------|----------|---------------------|
| Q_inter / Q_anat | Overall preference anchor | 1–5 | **Always mandatory** |
| I0 | Person count | 0/1 | **Always mandatory** |
| I1 | Penetration / blob severity | 0/1/2 | **Always mandatory** |
| Ic | Contact fit | 0/1/2 | **Always mandatory** (do not skip by system intent) |
| I3 | Forbidden contact / no extra entanglement | 0/1 | **Always mandatory** |
| Ir | Body region | 0/1; Ic=0→null only | **Mandatory when Ic≥1** |
| A1/A2/A3 | Extra / shape / ownership severity | 0/1/2 | **Always mandatory** |
| A4 / A5 | Proportion / hands | 0/1 | **Always mandatory** |

- Annotators also fill **`intent_human`** (system `intent_system` reference only)  
- `S_inter_H` / `S_anat_H`: by family using `intent_human` (see analysis protocol)  
- Mesh blind spots: `Ir`, `A5` → M reports geometry subset + `blind_items`  
- **Forbidden** for frontend to show "not applicable" from keywords/mesh intent and skip Inter items

### 3.1 Inputs

| Slot | H | V | M |
|------|---|---|---|
| Generated image | ✅ | ✅ | Not used (mesh json already exists) |
| Edit prompt | ✅ | ✅ | intent / n_expected |
| Reference image | Main task ❌ | ❌ | — |
| intent | **Human selects** `intent_human`; system guess stored separately | Same items or read system intent | `contact_intent` |

### 3.2 Encoding

| Code | Meaning |
|------|---------|
| **1** | Pass / no such error / requirement met |
| **0** | Fail / has error / requirement not met |
| **U** | Cannot judge (H may use; **V should minimize**, guide requires forced 0/1; M **forbidden U**) |

### 3.3–3.6 Inter / Anat / Guide / Gold (**Analysis Protocol Authoritative**)

Full item stems, decision trees, **S formulas**, U/null, gold, blind-spot reporting →  
[`eval_human_consistency_analysis_protocol.md`](eval_human_consistency_analysis_protocol.md) + frontend / `GUIDELINES.md`.

Researcher summary only here (**no I2/A0/A6**):

| Axis | Items | M |
|------|-------|---|
| Inter | I0, I1, Ic(0/1/2), I3, Ir | Ir blind spot; Ic←`P_miss`/`min_surf_dist` two levels |
| Anat | A1–A5 (**no A0**) | A5 blind spot |

- Primary metrics: `S_inter_*` / `S_anat` (intent families); appendix uses `*_pass`  
- Gold: item-level majority vote → then compute S  
- IAA primary: Krippendorff α (includes U)

### 3.7 Annotation Interface (Spreadsheet First, UI Later OK)

**Formal**: Web `annot_frontend` port **8080** (image + checklist + hotkeys 1/0/U); must not show any mesh/VLM scores.  
Persist: `human/<ann_id>/<key>.json` → `human/_consensus/`. CSV template may serve as backup import.

---

## 4. Mesh Initial Version → Checklist_M (Binary Mapping)

**1=pass, 0=fail**. τ maximizes H↔M κ on pilot set then **frozen** →  
`$PACK/judgments/human_consistency/_thresholds.json`.

### 4.1 Inter_M (`mesh_bin_map_v4`)

| ID | Starting rule (`map_v4`) | Fields |
|----|--------------------------|--------|
| I0 | `¬under_detect` ∧ `n_humans==n_expected` | mesh json |
| I1 | `P_fuse < τ_fuse` (0.30) ∨ `pen_volume ≤ vol_ok` | fuse / volume |
| Ic | required: `P_miss` two levels + `min_surf_dist` → {0,1,2} | miss / dist |
| I3 | forbidden: `P_unwanted < τ_unw` (0.40) | unwanted |
| Ir | **Blind spot** `null` | — |

### 4.2 Anat_M

| ID | Starting rule | Fields |
|----|---------------|--------|
| A1 | `P_anat_extra < 0.30` ∨ (orphan/leftover) | extra |
| A2 | `P_anat_resid < 0.18` or `S_anat_mesh` | resid |
| A3 / A4 | ownership / scale thresholds | — |
| A5 | **Blind spot** `null` | — |
| ~~A0~~ | Removed (person count only via I0) | — |

### 4.3 Ablations (Required)

| Variant | Definition |
|---------|------------|
| **M-item** (primary) | Item-level table above → AND for axis-level pass |
| **M-score** | `Inter_pass = 1[S_inter_mesh≥τ_S]`; Anat likewise |
| Report | κ to H for each; expect M-item ≥ M-score |

### 4.4 Calibration

1. Pilot ~100 samples, 3 annotators → majority-vote H*.  
2. Scan τ per item, maximize Cohen κ(M, H*) (or Youden).  
3. Freeze; holdout evaluated once only.

---

## 5. VLM-as-Judge → Checklist_V

### 5.1 Protocol

- **Model**: Same gateway as plan ① (e.g. `gpt-5.5`); appendix may add Gemini Flash etc. for sensitivity.  
- **Input**: Generated image + edit prompt + `intent` + `n_expected` (**main task does not feed reference image**, avoid ID entanglement).  
- **Output**: JSON **field-for-field** with humans; no Likert; no item ID changes.  
- **Temperature**: 0–0.1; one run per model for primary result; optional 3-run mode for robust appendix.

### 5.2 System Prompt

**Code authoritative**: `SYSTEM_SKELETON` in `code/eval/score_checklist_vlm.py` (**v4.1**: I0/I1/Ic/I3/Ir + A1–A5).  
This doc no longer pastes JSON skeleton that goes stale vs frontend dual-source.  
Do **not** batch-run V before human guide review; legacy v2/v3 V results archive only.

### 5.3 Optional V Alignment with Legacy anat_errors (Appendix)

| Checklist | If legacy v1 also run, coarse mapping (appendix only) |
|-----------|--------------------------------------------------------|
| A1=0 | Contains `extra_limb` / `floating_part` |
| A2=0 | Shape/deformity classes (legacy `missing_limb` coarse approx only) |
| A3=0 | Contains `limb_ownership_error` |
| I1=0 | `no_pathological_penetration != yes` or `merged_body` |
| — | **Main paper reports Checklist_V only** |

### 5.4 V Quality Control

- JSON schema validation; retry ≤3 on missing fields.  
- U rate > 15% → check prompt/resolution.  
- Compute κ with H on pilot; if item κ&lt;0.2, tighten stem examples then freeze.

---

## 6. Sample Design

| Phase | N | Purpose |
|-------|---|---------|
| Guide refinement | 20 | Item stems + diagrams |
| **Pilot** | 100 | Human–human κ; lock τ; tune V prompt |
| **Main experiment** | 250–300 | Formal H↔M↔V |
| Hard-case boost | +50 | High pen_volume / high leftover / low Inter / multi-hand |

Stratification: intent ≈ 60/30/10 (oversample if forbidden insufficient); closed-source 3 + open-source flux (experiment set dropped qwen); random 70% + hard 30%.

**All three judges must use the same sample ID set** (run M mapping and V on H-labeled set).

### 6.1 Starter Sample Pool (Engineering)

| Pool | Source | Use |
|------|--------|-----|
| smoke100 × existing image models | `"$MPIE_TEST_PACK"` + evaluation dashboard clickable images | Guide 20 + pilot 100 |
| Hard-case candidates | mesh_v3 high `P_fuse` / high `P_extra` / low `S_inter` / high leftover | 30% hard boost |
| intent gold | Pilot may manually fix `contact_intent`; keywords pre-fill only | Frozen to `_split.json` in formal phase |

`select_consistency_split.py` (planned): read manifest + mesh summary → write `_split.json` (pilot / holdout / main / hard).

---

## 7. Persistence and Directory Layout

```text
$PACK/judgments/human_consistency/
  _protocol.json              # plan version, checklist id, encoding
  _thresholds.json            # M τ (frozen)
  _split.json                 # pilot / holdout / main sample lists
  human/<annotator_id>/<sample_id>.json
  human/_consensus/<sample_id>.json    # majority vote
  mesh_bin/<sample_id>.json            # Checklist_M
  checklist_vlm/<judge_model>/<model_id>/<sample_id>.json
  reports/
    iaa_human.json
    agree_H_M.json
    agree_H_V.json
    agree_M_V.json
    tables_for_paper.md
```

Human single-record example (v4.1):

```json
{
  "sample_id": "arm_around_shoulder__000ad2a1de74__T1",
  "model_id": "gpt-image-2",
  "annotator_id": "ann_03",
  "protocol": "checklist_anat_inter_v4",
  "scheme": "v4.1",
  "intent_shown": "required",
  "inter": {"I0": 1, "I1": 1, "Ic": 2, "I3": null, "Ir": 1},
  "anat": {"A1": 1, "A2": 1, "A3": 1, "A4": 1, "A5": 0},
  "S_inter_family": "req",
  "S_inter_H": 1.0,
  "S_anat_H": 0.8,
  "seconds": 45
}
```

Mesh mapping examples in live `mesh_bin/` (includes `S_inter_M` / `blind_items`); fields per `checklist_common.map_mesh_to_checklist`.

---

## 8. Statistical Analysis (Complete)

### 8.1 Human–Human (Reliability)

- Item-level: **Krippendorff α (includes U, primary)**; Fleiss κ (U removed, appendix).  
- **Stratify by intent**; null excluded from denominator. See analysis protocol.  
- (Historical) Fleiss was threshold; now appendix only.
- Axis-level: Inter_pass / Anat_pass.  
- Appendix by intent, by model.

### 8.2 Pairwise Agreement (Main Paper)

For each pair **(X, Y) ∈ {(H,M),(H,V),(M,V)}**, on consensus-decidable samples:

| Granularity | Report |
|-------------|--------|
| Item-level | Acc, P, R, F1, Cohen κ, confusion matrix, positive rate |
| Axis-level | Same (Inter_pass / Anat_pass) |
| Stratification | × intent × model × hard/random |

**Main-paper recommended table (illustrative)**

| Axis | H↔M κ | H↔V κ | M↔V κ | Human–human κ |
|------|-------|-------|-------|---------------|
| Inter_pass | … | … | … | … |
| Anat_pass | … | … | … | … |
| I1 / A1 (core items) | … | … | … | … |

### 8.3 Who Is Closer to Humans?

At axis or item level: compare κ(H,M) vs κ(H,V); bootstrap CI optional.  
Narrative may allow: **Mesh stronger on I1 (penetration), VLM stronger on I4/A5 (region/hands)** etc. per item.

### 8.4 Ablations and Sensitivity

- M-item vs M-score  
- τ ±10%  
- V with different judge model  
- κ after removing hard-case boost  

### 8.5 External Messaging

> We validate whether geometry binarization (M) and VLM checklist (V) reproduce human judgments on **the same human-actionable error checklist**; we do not validate that humans compute 3D volume.

---

## 9. Engineering Checklist

Analysis protocol: [`eval_human_consistency_analysis_protocol.md`](eval_human_consistency_analysis_protocol.md)

### 9.1 Samples and Annotation

- [x] `select_consistency_split.py` → `_split.json` (+ `annot_templates/*.csv`, `_protocol.json`, starter `_thresholds.json`)  
- [x] Annotation preview HTML: `build_annot_preview.py` → `annot_preview/` (CSV template synced)  
- [x] **Annotation frontend 8080**: `annot_frontend/app.py` (homepage shows 3 annotators needed: ann_01/02/03; principles+checklist+persist)  
- [x] Checklist wording in this document (construct items + severity)  

- [x] `import_human_checklist.py` → `human/` + `_consensus/` (majority vote §3.6)  
- [ ] Recruit 3 annotators for real fill + onboarding quiz  

### 9.2 Mesh Side

- [x] `export_mesh_checklist.py`: read `mesh_v3` → `mesh_bin/` (mapping in `checklist_common.map_mesh_to_checklist`)  
- [x] `calibrate_checklist_thresholds.py`: lock τ in pilot (script ready; pending H*+mesh)  
- [x] `_thresholds.json` version `map_v1` (starter default; `status=default_unfrozen`)  
- [ ] Run export after syncing smoke-related `mesh_v3`  

### 9.3 VLM Side (Same Human Sheet, Not Legacy v1)

- [x] `score_checklist_vlm.py` + `run_score_checklist_vlm.sh` (schema + resume; dry-run OK)  
- [ ] Configure gateway, batch pilot → `checklist_vlm/`  

### 9.4 Analysis

- [x] `compute_agreement.py`: IAA + H↔M / H↔V / M↔V + `tables_for_paper.md` (script ready; pending three-way data)  
- [x] `prep_consistency_selfcheck.py` self-check entry  

### 9.5 Timeline

```text
Week 1: Guide+20 refine → pilot 100 (H) → lock τ → run M_bin + V
Week 2–3: Main 250–300 (H) → full M/V → κ tables
Week 4: Hard-case review, sensitivity, paper paragraphs
```

---

## 10. Checklist Quick Card (v4.1 · Shared by All Three)

> Full wording per 8080 / `GUIDELINES.md`.

### Inter

| # | Question | Encoding |
|---|----------|----------|
| I0 | Person count (named roles only) | 0/1 |
| I1 | Penetration / inseparable blob (hand blob → A5) | 0/1 |
| Ic | Ordered contact fit | 0/1/2 · required |
| I3 | Forbidden contact | 0/1 · forbidden |
| Ir | Body region (not 0 for poor fit) | 0/1 · required; Ic=0→null |

### Anat

| # | Question |
|---|----------|
| A1 | No extra/floating limbs |
| A2 | Body shape (exaggerated pose=1; break=0; gray=U) |
| A3 | No wrong-person attachment |
| A4 | Proportion |
| A5 | Hands identifiable (in construct) |

---

## 11. Relation to Main Table / Versions

| | main evaluation table | This experiment |
|--|------------------|-----------------|
| Anat/Inter | Continuous `S_*_mesh` **initial v0.1** | Checklist 0/1: H / M / V |
| Overall | Removed | Not involved |
| Change τ / V prompt | Does not change v0.1 continuous definition | Mapping or judge version increments |
| Change mesh formula | Bump v0.2 | Re-run M_bin and comparisons |

**Metric separation (important)**: Main table keeps continuous `S_anat_mesh` / `S_inter_mesh` for model ranking; this experiment only asks "does binary checklist look human". Both must pass to claim mesh metrics are trustworthy; high main-table score but very low H↔M κ → fix formula, not just presentation.

---

## 12. Known Limitations and Risks (Write into Appendix)

| Risk | Impact | Mitigation |
|------|--------|------------|
| Mesh intent still keyword-based | Ic/I3 mapping drift | Intent gold in formal phase |
| Blob merge → Anat marked U | `S_anat` effective n drops | Pilot report n_S_anat and blob rate |
| I5/A5/A6 not mapped by M | Region/hand/break only H↔V | Acknowledge geometry blind spots in narrative |
| A2 uses human "deformity" vs mesh resid | Semantics not fully isomorphic | Watch κ in pilot; tighten stems if needed |
| Anat `P_extra` similar across models | A1 M may be globally loose/tight | Lock τ in pilot; hard-case boost |
| Low human–human κ | All automatic comparisons meaningless | Fix stems/diagrams first |
| VLM checklist too lenient → near perfect | H↔V no discrimination | **v3 tightened stems**; run V after human review |
| Annotators see "bad image" bias | Axis all 0 | Guide stresses item-independent scoring |

---

## 13. Paper Presentation (Suggested)

| Location | Content |
|----------|---------|
| Main text short paragraph | "To validate Anat/Inter geometry metrics, we compare humans, mesh binarization, and VLM-as-judge on a unified checklist" |
| **Main table** | Axis κ: H↔M / H↔V / human–human; core items I1, A1 |
| Figure | Confusion matrices (I1, A1); bar chart by intent |
| Appendix | Full-item κ, M-item vs M-score, τ sensitivity, legacy v1 coarse mapping, alternate judge model |

---

## 14. Compute / Cost Rough Estimate (Pilot Scale)

| Item | Rough amount |
|------|--------------|
| Human: pilot 100 × 3 × ~45s | ~4 person-hours; main 300 × 3 ≈ 11 person-hours |
| Mesh M_bin | Near zero if `mesh_v3` exists (mapping only) |
| V checklist: 100×5 models×1 judge | Same API order as legacy v1; scale main by same samples |
| Analysis scripts | CPU minutes |

---

## 15. Related Docs and Code (Planned)

| Path | Role |
|------|------|
| This doc | Experiment master spec (authoritative record) |
| `eval_mesh_*_spec.md` | Mesh initial continuous scores |
| VLM Count scorer (`score_vlm_v1.py`) | Legacy multi-axis VLM fields may exist in dumps; main table uses Count only |
| `code/eval/checklist_common.py` | Encoding / pass / majority vote / default τ / M mapping |
| `code/eval/select_consistency_split.py` | Stratified sampling → `_split.json` |
| `code/eval/export_mesh_checklist.py` | M mapping → `mesh_bin/` |
| `code/eval/calibrate_checklist_thresholds.py` | Lock τ |
| `code/eval/score_checklist_vlm.py` | V batch run |
| `code/eval/run_score_checklist_vlm.sh` | V batch entry |
| `code/eval/import_human_checklist.py` | CSV→human/_consensus |
| `code/eval/compute_agreement.py` | κ / reports |
| `eval_human_consistency_analysis_protocol.md` | Agreement analysis protocol |

---

## 16. Decision Log (This Experiment)

| Date | Decision |
|------|----------|
| 2026-07-16 | Primary comparison is **human checklist**, not asking humans to estimate 3D volume |
| 2026-07-16 | Automatic side two tracks: **M** (mesh→binary) + **V** (VLM on **same** human sheet) |
| 2026-07-16 | Legacy `score_vlm_v1` six axes **do not replace** Checklist_V; appendix OK |
| 2026-07-16 | Gold = 3-person item majority; main paper item consensus then AND for axis pass |
| 2026-07-16 | τ locked in pilot only; holdout/main evaluated once |
| 2026-07-16 | evaluation dashboard continuous v0.1 vs this checklist **metric separation** |
| 2026-07-17 | **v3**: tighten discrimination; I4=fit in main pass, I5=region auxiliary; A2=shape residual mappable; default τ tightened; V paused until human review |

---

## 17. Revision History

| Date | Content |
|------|---------|
| 2026-07-18 | **v5.1**: human Inter all items mandatory; cancel system intent item hiding; `intent_human`; scheme v5.1; frontend 8080 |
| 2026-07-18 | **v5.0**: I1/A1/A2/A3 ordered severity; Q anchors; mesh dual-threshold mapping |
| 2026-07-16 | v1: H + Mesh mapping draft |
| 2026-07-16 | **v2**: add **VLM same checklist**; H↔M↔V, persistence, stats, engineering checklist, legacy v1 boundary |
| 2026-07-16 | **v2.1**: consensus rules, spreadsheet minimum, sample pool, limitations, paper presentation, cost, decision log |
| 2026-07-16 | Engineering start: `checklist_common` + `select_consistency_split` + `export_mesh_checklist`; smoke100 `_split.json` (guide20/pilot100/holdout50) |
| 2026-07-16 | Prep complete: V batch / human import / lock τ / agreement / preview HTML / runbook; blocked=mesh_v3 sync, hiring, gateway env |
| 2026-07-17 | **v3.0**: checklist/τ/mapping reset (see §3–4); v2 human and V results archived void; frontend 8080 follows v3 |
| 2026-07-17 | **Establish construct validity principles** + **v4.0**: ordered Ic, Ir/A5 in construct, remove A0, decision tree, main paper correlation/stratification |
| 2026-07-17 | **v4.1 analysis protocol**: fixed S formulas, U/null, gold, three parallel blind-spot blocks; Ic↔Ir / I1↔A5 / I0 / A2 boundaries pinned |
