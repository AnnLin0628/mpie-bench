# Anat / Inter Analysis Protocol (Peer to Checklist)

> **Status: 2026-07-18 · v5.1** · Peer to checklist protocol `checklist_anat_inter_v5`  
> Principles: [`eval_construct_validity_principle.md`](eval_construct_validity_principle.md)  
> Checklist items: [`eval_human_consistency_anat_inter.md`](eval_human_consistency_anat_inter.md)  
> Implementation: `code/eval/checklist_common.py` (`SCHEME_VERSION=v5.1`) · frontend `annot_frontend` 8080  
> v4.1 binary checklist archived; **v5 motivation**: coarse binary splits make H/V converge and absorb mesh continuous variance → three-level severity.  
> **v5.1**: human **Inter all items mandatory**; system/keywords/`contact_intent` **must not** hide items; stratification uses `intent_human`.

This document answers: **how S is composed, how U/null enter analysis, how intent/blind spots are reported, how gold labels come from multiple annotators, what the main paper compares, and how mesh→per-item mapping works**.  
Without these spelled out, no matter how clean the checklist is, it cannot support a formal experiment.

---

## 0. Hard Rules for Human Annotation (v5.1 · Mandatory)

| Rule | Description |
|------|-------------|
| **Inter all items mandatory** | Annotators must answer `I0, I1, Ic, I3, Ir` (only when `Ic=0` does Ir→`null`, as a logical dependency) |
| **No system-prior item hiding** | split / keywords / mesh-derived intent are **reference only**; the frontend must not show "not applicable for this intent" and skip Ic/I3/Ir |
| **`intent_human`** | Annotator selects `required\|forbidden\|unspecified` from the edit instruction; persisted as `intent_human` / `intent_used` |
| **`intent_system`** | System guess stored separately as `intent_system` (legacy name `intent_shown`); **analysis stratification and S families use `intent_human`** |
| **Checklist priors ≠ ground truth** | Checklist items are inspired by mesh measurability, but human answers are the standard; M mapping must not be inverted to serve as human gold |
| **Overall + item-level both mandatory** | `Q_inter`/`Q_anat` (preference anchors) and item-level scores (construct operationalization) run in parallel; neither alone is sufficient |

On the automatic side (M mapping / optional V gating), intent may still decide which fields enter `S_*` formulas; **no such gating during human collection**.

---

## 0b. What the Main Paper Compares (Plain Language)

| Priority | Compare | Meaning |
|----------|---------|---------|
| **P0** | `corr(Q*_inter, S_inter_mesh)` and `corr(Q*_anat, S_anat_mesh)` | Whether mesh continuous scores track human preference |
| **P0** | Same for V Q/S; expect V inflated or weaker correlation | VLM control |
| **P1** | Ordered items H\*↔M (I1/Ic/A1…) stratified by `intent_human` | Whether "mesh-defined items" are accurate for humans |
| **P2** | Checklist κ(H,V) | Appendix; high κ does not prove V understands geometry, only that it can fill a 2D answer sheet |

---

## 1. Construct Score Formulas (Fixed)

Unified scale: all components entering averages ∈ **[0, 1]**.  
Ordered items: `I1_norm = I1/2`, `Ic_norm = Ic/2`, `A1_norm=A1/2`, `A2_norm=A2/2`, `A3_norm=A3/2` ({0,1,2} → {0, 0.5, 1}).  
**S aggregated by family using `intent_human` (or automatic-side `contact_intent`)**; in human records, Ic/I3/Ir are **kept as raw answers** even when "not in the family formula average" for appendix / sensitivity.

### 1.0 Ordered Severity Encoding (v5)

| Item | Domain | 2 | 1 | 0 | Mesh continuous anchor |
|------|--------|---|---|---|------------------------|
| I1 | Penetration / blob merge | Clean | Mild | Severe | `P_fuse` (dual threshold) |
| Ic | Contact fit | Fitted | Poor fit | No contact | `P_miss` / `min_surf_dist` |
| A1 | Extra structure | Clean | Mild | Severe | `P_anat_extra` |
| A2 | Body-shape anomaly | Normal | Mild | Severe | `P_anat_resid` |
| A3 | Ownership error | Normal | Mild | Severe | `S_anat_ownership` |

I0 / I3 / Ir / A4 / A5 remain 0/1 (or U).  
Appendix AND pass: ordered items must be **full score 2**; binary items must be 1.

### 1.1 Inter (Split by Intent; No Pooling)

| Family | Symbol | Formula | Core items (any U → sample-level S void) |
|--------|--------|---------|--------------------------------------------|
| required | `S_inter_req` | \(\mathrm{mean}(I0,\ I1/2,\ Ic/2,\ [Ir])\) | I0, I1, Ic; add Ir if Ic≥1 |
| forbidden | `S_inter_forb` | \(\mathrm{mean}(I0,\ I1/2,\ I3)\) | I0, I1, I3 |
| unspecified | `S_inter_unspec` | \(\mathrm{mean}(I0,\ I1/2)\) | I0, I1 |

Where \([Ir]\): included only when `Ic ∈ {1,2}`; when `Ic=0`, Ir must be `null` and is excluded from the mean.

**Denominator varies with Ic / I1 (must be written into the guide):**

| Ic | Actual averaged terms in `S_inter_req` | Theoretical upper bound when I0=1 and I1=2 |
|----|----------------------------------------|---------------------------------------------|
| 0 | `mean(I0, I1/2, 0)` | **2/3 ≈ 0.667** (not 1) |
| ≥1 | `mean(I0, I1/2, Ic/2, Ir)` | 1.0 (requires Ic=2, Ir=1, and I1=2) |

→ "Correct person count + no penetration" **cannot** reach full `S_inter_req`; with no contact, the cap is 2/3.

**Do not** report unstratified "overall Inter correlation / overall Inter κ" as a main conclusion.  
Main-paper tables must at least split rows: `required` | `forbidden` (if any) | `unspecified`.

Persisted fields: `S_inter_H` (score for the sample's family) + `S_inter_family` (`req|forb|unspec`).

### 1.2 Anat

\[
S_{anat} = \mathrm{mean}(A1/2,\ A2/2,\ A3/2,\ A4,\ A5)
\]

Core items: `A1,A2,A3,A4,A5` (any U → sample-level `S_anat` void = `null`).

**Blob merge and effective n (must watch in pilot):**  
The decision tree requires "cannot separate people in blob merge → I1=0, Anat marked U" → that sample gets **`S_anat=null`** and exits Anat correlation analysis. This is reasonable (structure not evaluable), but makes Anat effective n skew toward the "good separation" subset. Pilot reports must give: `n_S_anat`, blob/I1=0 rate, and correlations only on the `S_anat` non-null subset.

### 1.3 What Not to Do

- Do not use "first binarize Ic≥2 then average" as main `S` (that is appendix `Inter_pass`).  
- Do not sum unnormalized Ic raw values with binary items directly (avoids 2 dominating).  
- Do not treat `null` as 0 or as agreement.

### 1.4 Parallel Mesh-Side Definitions

| Symbol | Definition | Use |
|--------|------------|-----|
| `S_inter_M` | Same structure as §1.1, but **excludes Ir** (blind spot) | Human–Mesh correlation (mappable subset) |
| `S_anat_M` | \(\mathrm{mean}(A1..A4)\), **excludes A5** | Human–Mesh correlation (mappable subset) |
| `S_*_mesh` continuous main table | Original evaluation dashboard formulas | Model ranking; construct validity via correlation with human constructs |

---

## 2. Decoupling Ir and Ic (Annotation Rules)

- **Ic**: Only evaluates "whether contact occurred / whether fit is good".  
- **Ir**: Only evaluates whether the contacted body **region** roughly matches contact intent.  
- **Ir is not scored 0 for poor fit** (fit belongs to Ic).  
  - Example: hand reaching waist off by ~10cm → may be `Ic=1, Ir=1`.  
  - Example: very close contact but wrong body part → may be `Ic=2, Ir=0`.  
- `Ic=0` → `Ir=null` (region not evaluated).

---

## 3. I1 ↔ A5 Boundary

| Situation | I1 | A5 |
|-----------|----|----|
| Full-body blob merge, people not separable | **0** | U (do not double-count) |
| People separable + obvious penetration into torso/head | **0** | Score hands independently |
| People separable + mild adhesion / shallow penetration | **1** | Score hands independently |
| People separable + **hand blob only / extra fingers** | **2** (or U if no opinion on penetration) | **0** |
| Do not set I1=0 just because hands are blobby | — | — |

---

## 4. A2 Boundary (Reduce κ Risk)

A2 bucket is wide (deformity / folded misalignment / impossible break). The guide must include boundary examples:

| Score 1 (normal) | Score 0 (fail) |
|------------------|----------------|
| Exaggerated dance/fight pose; joints still read as bendable | Obvious reverse break in lower leg, torso fold misalignment, unreasonable limb stretch/collapse |

Gray zone uncertain → **U**, do not guess.

---

## 5. I0 Bystander Rule

- **Only named roles in the instruction (R# / by name) count toward person count.**  
- Waitstaff handing items, onlookers, unnamed child being led, etc. → **all bystanders, ignore.**  
- Counterexample (I0 should still pass): instruction asks only R1–R2 to hug; background waiter handing a menu → count remains 2.

---

## 6. U / null in Analysis (Fixed)

| Object | Rule |
|--------|------|
| `null` | Logical missing only (e.g. Ic=0→Ir); **excluded** from κ/S denominators. Human side **no longer** produces "not applicable null" from system intent |
| `U` | Cannot judge |
| **Item-level agreement (primary)** | **Krippendorff α** (nominal/ordered: Ic uses ordered), **explicitly includes U**; appendix may report %agree / κ with U removed |
| **Majority-vote gold** | ≥2 annotators same 0/1/(same Ic level) → take that value; 1 U + two same → take the two; two conflict or two U → **drop that item** (gold missing for that item) |
| **Sample-level S** | Computed only when all **core items in that family are non-U and non-missing**; otherwise `S=null` (sample exits correlation analysis) |
| **Forbidden** | Treat U as 0; treat null as agreement; use simple %agree as primary IAA |

---

## 7. Human Gold Label (Anchor) Definition

**Formal setting (main paper): 10 independent annotators; item scores are averaged.**

1. **Item-level gold H\***: For each item, after skipping `U`, take the arithmetic mean over **10 annotators**, then round to the item's legal code (binary→`{0,1}`; ordered→`{0,1,2}`).  
2. **`intent_human` gold**: 10-person majority vote (on conflict, drop that sample from intent layer or expert adjudication).  
3. **Construct gold**: Recompute `S_inter_H` / `S_anat_H` from H\* + intent gold (§1).  
4. **`Q*`**: Mean of 10 annotators' `Q_inter`/`Q_anat`, rounded to 1–5.  
5. **Main paper human–Mesh / human–VLM**: Per item, Spearman ρ(H\*, M) vs ρ(H\*, V); appendix may report Q\* ↔ continuous `S_*_mesh`.  
6. **Legacy compatibility**: Three-person majority vote only for early pilot; no longer the main-paper gold definition.

## 7b. Mesh→Checklist Per-Item Mapping (M · item_map_v6)

M **does not** slice one total score into 1–5; it maps **each checklist item** from mesh\_v3 fields to the same option codes as H/V:

1. Select features per item (e.g. I1: penetration+blob/ownership/residual; Ic: `S_prox`/`P_miss`; A1: ownership/extra structure…).  
2. Standardized ridge regression predicts human item scores.  
3. Quantile cuts to `{0,1}` or `{0,1,2}` (maximize Spearman with H on calibration set).  
4. Weights and cuts written to `_item_map_calib.json`, frozen before evaluation; `export_mesh_checklist.py` exports `mesh_bin/`.

Authoritative implementation: `code/eval/calibrate_mesh_item_map.py` + `checklist_common.map_mesh_to_checklist`. Design notes: `reports/mesh_item_mapping_design.md` in pack.  
6. Consistency (IAA) and validity (H\*↔M) **reported separately**.

---

## 8. Blind Spots and Main-Paper Reporting (Avoid Selective Proof)

Main paper **must report all three blocks in parallel**:

| Block | Content |
|-------|---------|
| **(a)** | Mappable subset: `S_inter_M`↔`S_inter_H` (no Ir), `S_anat_M`↔`S_anat_H` (no A5); and item-level I0/I1/Ic/A1–A4 |
| **(b)** | Full construct human–human: IAA / `S_*_H` stability including Ir, A5 |
| **(c)** | Blind-spot items human-only (and H↔V): Ir, A5; acknowledge M does not cover interaction regions and hands |

Do not report only (a) and claim "Mesh fully captures interaction/anatomy quality".

---

## 9. VLM Inflation Control

- V must use **the same checklist and same S formulas**.  
- Main comparison: within the same intent layer, `mean(S_V) > mean(S_H)` (inflated) and `corr(S_V,S_H) < corr(S_M,S_H)` (or lower κ).  
- Do not use a lenient V checklist excluding Ir/A5 to manufacture false contrast.

---

## 9b. Discrimination Hard Constraints (2026-07-19 · Mandatory)

**Problem**: On closed-source and simple-contact (C0/C1) samples, Anat/Inter continuous scores cluster at the high end; collapsing to checklist further ceiling-effects → inflated human–human κ, construct validity cannot measure "tracking preference".

**Sample-pool hard constraints (consistency split `v4_open5_c23`):**

| Constraint | Target |
|------------|--------|
| Five open-source models | **flux / dreamo / uno / omnigen2 / ace** evenly represented (~20% each); **exclude** bagel (not evaluated), qwen (too poor) |
| Contact density | **C2+C3 only** (exclude overly simple C0/C1) |
| Hard-case slots | **≥ 60%** (`is_hard_slot`; signal=mesh hard_score preferred, else VLM-v1 failure score) |
| Closed-source | **Not sampled this round** (discrimination pool open-source only) |
| H↔M main claims | Only on **`has_mesh=true`** subset (this pool should be 100% with mesh) |

**Human preference anchors (overall scores, parallel to item-level):**

| Field | Encoding | Use |
|-------|----------|-----|
| `Q_inter` | 1–5 (overall interaction quality) | Main paper: `corr(S_inter_mesh, Q_inter)` / `corr(S_inter_H, Q_inter)` |
| `Q_anat` | 1–5 (overall anatomy quality) | Main paper: `corr(S_anat_mesh, Q_anat)` / `corr(S_anat_H, Q_anat)` |

- 1=very poor … 5=very good; glance at image ~3s for overall, then item-level (or items first then overall, but **both mandatory**).  
- Overall scores separate preference when "closed-source all look OK"; item-level explains failure modes.  
- **Do not** use checklist AND alone / only three strong closed-source evenly sampled cases as the sole construct-validity evidence.

---

## 10. Appendix: AND pass (Not Main Paper)

- `Inter_pass`: required → I0=1 ∧ **I1=2** ∧ Ic=2 ∧ Ir=1; forbidden → I0=1 ∧ I1=2 ∧ I3=1; unspec → I0=1 ∧ I1=2  
- `Anat_pass`: **A1=A2=A3=2** and A4=A5=1  
- Product-style summary / sensitivity only; **does not** replace §1 construct scores.

---

## 11. Revision History

| Date | Content |
|------|---------|
| 2026-07-19 | **Gold changed to 10-person item mean then round**; main paper per-item ρ; mesh→item mapping item_map_v6.2 written into paper |
| 2026-07-19 | **split v4_open5_c23**: 5 open-source evenly + C2/C3 only; exclude bagel/qwen and C0/C1 |
| 2026-07-19 | split v3_open_heavy (deprecated): weak models≥80%, single closed-source≤5%, hard≥60% |
| 2026-07-18 | **v5.1**: human Inter all items mandatory; cancel system intent item hiding; `intent_human` stratification; main paper P0=`Q*`↔`S_*_mesh`; frontend 8080 synced |
| 2026-07-18 | **v5.0**: I1/A1/A2/A3 three-level severity; S uses `/2` normalization; mesh dual-threshold mapping; main paper prioritizes ordered items↔continuous mesh + Q_* |
| 2026-07-17 | v4.1: S formulas, U/null, gold, three parallel blind-spot blocks, Ic↔Ir / I1↔A5 / I0 / A2 boundaries written into analysis protocol |
| 2026-07-17 | Explicit Ic=0 caps `S_inter_req` at 2/3; blob merge reducing `S_anat` effective n must be tracked in pilot |
