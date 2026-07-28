# MPIE-Bench Core Principle: Anat / Inter Construct Validity

> **Status: established 2026-07-17** · One of the highest-priority principles in the repository  
> Related experiment: [`eval_human_consistency_anat_inter.md`](eval_human_consistency_anat_inter.md) (protocol v4+)

## 1. What we need to prove

The paper's **Anat / Inter geometric evaluation system** should:

1. **Have clear constructs**: measure "anatomical quality" and "interaction quality," not a vague acceptance checklist;  
2. **Be substantively reasonable**: dimensions should be as mutually exclusive as possible, evaluable on all samples or with strict stratification, and governed by a forced decision tree;  
3. **Align with human preference**: use human judgments (H) as the anchor; the Mesh system (M) should agree / correlate more with H;  
4. **Support a valid contrast**: on the same constructs, VLM evaluation (V) should show **inflated scores** and weaker alignment with H.

This is not an engineering-alignment experiment about "whether three sides fill the same checklist." Alignment is only a means; **construct validity is the goal**.

## 2. Rebuttals we explicitly reject

Under this claim, the following responses **cannot** be used to dismiss methodological criticism:

- "This is a product acceptance checklist; consistency is enough"  
- "Intent gating is reasonable in production, so mixing one κ is fine"  
- "Body parts / hands are excluded from the main pass because mesh cannot map them" — if the paper still claims full interaction/anatomy quality, the claim and operational definition are misaligned  

If body parts / hands cannot yet be measured, the **paper claim must be narrowed** to "geometry-measurable subconstructs," and coverage gaps must be stated explicitly; they cannot be silently dropped.

## 3. Measurement-model requirements (summary)

| Requirement | Meaning |
|------|------|
| Human-preference anchor | H item-level / ordinal construct scores (and optional overall scores) define "ground truth preference" |
| M aligns with H | Main paper: correlation, rank agreement, item-level stratified agreement |
| V is inflated | On the same items/constructs, mean score is higher and agreement with H is worse |
| Contact is ordered | No contact → contact but not touching → touching (forbidden parallel fake independent binary items) |
| Person count once | Count people only in Inter; do not repeat in Anat |
| Decision tree | On blur/clump, penetration, extra limbs, or no contact, downstream items force attribution and prevent double counting |
| Stratified reporting | Report required / forbidden / unspecified separately; null does not enter κ |
| Main-paper metrics | Construct scores and item-level metrics; AND pass rate demoted to appendix |

Details and item wording are in human-consistency protocol **v5**; **S formulas / U·null / gold labels / blind spots / humans must answer all items** are in analysis protocol **v5.1** (same level as item wording):  
[`eval_human_consistency_analysis_protocol.md`](eval_human_consistency_analysis_protocol.md).

## 4. Relation to main-table continuous scores

The main evaluation table continues to report `S_anat_mesh` / `S_inter_mesh` for model ranking; the human-construct experiment answers "do these scores look human?"  
**Both must pass** before the metrics can be called trustworthy: high main-table scores but very low correlation with H → fix the formula, not just the presentation.

## 5. Revision history

| Date | Change |
|------|------|
| 2026-07-17 | Principle established; response to expert critique that "acceptance checklist ≠ measurement instrument"; protocol v4 launched |
| 2026-07-17 | Analysis protocol v4.1: S formulas, U/null, gold labels, three parallel blind-spot blocks (same level as checklist) |
| 2026-07-19 | **Main consistency paper**: 10-annotator item mean as gold label; mesh per-item mapping item_map_v6; per-item ρ(H,M) vs ρ(H,V) |
| 2026-07-19 | **Discrimination sample pool v4_open5_c23**: all 5 open-source models represented + C2/C3 only; exclude C0/C1 / bagel / qwen as main pool |
| 2026-07-17 | **Discrimination sample pool v2**: hard cases ≥50%, weak models ≥60%, Q_inter/Q_anat overall scores; forbid using only three strong closed-source uniform samples to prove construct validity |
| 2026-07-18 | **v5 / v5.1**: ordinal severity; humans must answer all Inter items (system intent cannot hide items); main-paper P0=`Q*`↔continuous `S_*_mesh`, item-level validation that mesh priors look human |
