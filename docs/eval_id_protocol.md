# ID Consistency Evaluation Protocol: Pose Degradation and "Face-Hiding" Anti-Cheat Design

> **Status: Protocol finalized (2026-07-13), implementation not built. Priority: ★ Core protocol — this design directly determines whether ID metrics can serve as RL rewards; when building the ArcFace evaluation environment, follow this strictly; do not run bare ArcFace.**
>
> Background (user 07-13): Some reference images are profile views with few frontal faces; some generated images show people facing away — can ArcFace still evaluate normally?
> Conclusion: Yes, but "face not visible / large profile" must be first-class in the protocol; otherwise the metric has systematic bias and can be RL-gamed.

## 1. ArcFace Pose Tolerance Facts

| Situation | Usability | Notes |
|-----------|-----------|-------|
| Frontal ref × frontal gen | ✅ Normal | Same-person similarity typically 0.6–0.7, imposter ~0.1–0.2 |
| Frontal ref × large profile gen (yaw 60°+, detectable) | ⚠️ Degraded but usable | Same-person similarity drops to ~0.3–0.45, still above imposter, but **cannot use same scale/threshold as frontal** |
| Extreme profile (\|yaw\| ≳ 70°) | ❌ Mostly undetectable | RetinaFace/SCRFD detection layer fails; no embedding |
| Back to camera | ❌ Fully unmeasurable | No face to extract |

Two implications: (a) detection layer is the first gate — no detection means that person cannot enter the ArcFace channel; (b) detectable profile scores are systematically lower and need yaw calibration before aggregation.

## 2. Reference Side: Frontal Recipe Solves Most; Tag Residual Risk

- Final reference images follow [frontal recipe](standard_process.md) (det≥0.62 + \|yaw\|≤38°); ArcFace decay in this yaw range is small → **reference side default OK**.
- Residual risk: Some categories (back_hug, piggyback, etc.) may force relaxed yaw when selecting references.
- **Requirement**: Any reference selected with relaxed frontal-recipe thresholds must be tagged `ref_quality: relaxed` in manifest; for ID eval either drop that scene or report separately — **must not mix into main table**.

## 3. Generated Side: Three-Way Accounting by GT Visibility (Core Rule)

For person p, let GT target face detectability be `gt_visible(p)`, generated image detectability be `gen_visible(p)`:

| Branch | Accounting rule | Rationale |
|--------|-----------------|-----------|
| **① gt_visible ∧ gen_visible** | Normal ArcFace similarity; bucket-calibrate by gen-side yaw before aggregation | Standard channel |
| **② gt_visible ∧ ¬gen_visible** | **Score ID = 0 (floor), never N/A exclusion** | **Anti reward hacking**: this metric may serve as RL reward; if "no face detected = no penalty", optimal strategy is draw everyone facing away and ID is hollow. GT face visible but generated hides face = identity preservation failure |
| **③ ¬gt_visible** (GT itself back/no face, e.g. back_hug person being carried) | Remove from ID denominator; identity consistency degrades to weak hair/clothing/body constraints via **person-ReID or person-crop CLIP/DINO similarity** as auxiliary signal, **report separately, do not merge with ArcFace score** | Fair to model: GT has no visible face, cannot require generated face |

Required diagnostic column: **Face-Visible Rate** = |{p: gt_visible ∧ gen_visible}| / |{p: gt_visible}|. It is itself a failure dimension — many edit models systematically "hide faces" under high contact density; this rate is evidence.

## 4. Yaw Bucket Calibration: Calibrate on In-House Multi-View Data (Paper Section)

Profile score depression is systematic bias; multi-view GT data makes it a correctable term:

- **Calibration set**: Harmony4D / Panoptic / EgoHumans are multi-camera same-person same-frame — same identity from frontal to full profile to back all have GT — unique calibration resource for this project.
- **Procedure**: On calibration set plot ArcFace same/different-person similarity vs yaw → per yaw bucket (e.g. 0–20°/20–40°/40–60°/60°+) get independent decision threshold and normalization coefficient → at eval time estimate gen face yaw, normalize by bucket then aggregate.
- **Byproduct**: Calibration itself can be a paper section ("quantifying failure modes of off-the-shelf ID metrics in multi-person interaction scenes"), ID counterpart to CHI3D/FlickrCI3D contact GT calibration.

## 5. Multi-Person Matching: Third Persons and False Matches (07-13 Expanded Final)

"Third person in target/generated not in reference set" splits into two scenarios with different mechanisms:

### 5.1 Scenario A: GT Target Already Has Third Person (Video bystander / unselected actor)

GT reference↔target mapping **does not rely on cross-image ArcFace matching at eval time**; it inherits data-build artifacts: reference and target same video same scene, Stage 3 person assignment clusters all face crops by ArcFace (or H4D/CHI3D GT identity); reference is one frame in cluster; review dashboard "target↔person binding" manually confirmed. Same-video same-clothing adjacent-frame within-cluster matching (~0.7+) far more reliable than cross-image → **GT mapping is offline fact frozen in manifest, not computed at eval**.

- Unbound persons at review = unbound bystanders, not in ID denominator, skipped at eval.
- **Manifest must record bystander face boxes as `distractor_faces`**, otherwise `gt_visible` precheck wrongly counts bystander faces in expected set.

### 5.2 Scenario B: Model Hallucinates Third Person in Generated Image

At eval, cross-image matching with three safeguards:

1. Detect all N faces in generated image × M reference embeddings → M×N similarity matrix;
2. **Hungarian one-to-one global optimal assignment** (`scipy.optimize.linear_sum_assignment`, maximize sum of matched similarities). Extra faces naturally unmatched, no ID score — **extra people penalized by Count dimension, ID does not also check person count**;
3. **Threshold gate (after matching)**: Each matched pair similarity must ≥ that yaw bucket's imposter upper bound (from §4 multi-view calibration table); below threshold match voided, treated as undetected, falls into §3 branch ② score 0. Blocks "real person back + hallucinated face visible → Hungarian forces false match"; hallucinated face cannot steal score.

**Order must not reverse**: Hungarian first, then threshold gate. Threshold-first then match would discard "poorly drawn but correct person" candidates early.

### 5.3 Hungarian vs Greedy (Why Not "Each Claims Highest")

Greedy (each reference takes max) is unstable under **identity confusion** — exactly this benchmark's key failure mode: ref A sim 0.55/0.50 to gen faces 1/2, ref B 0.60/0.20; greedy both want face1, order-dependent, B may take 0.20. Hungarian compares full assignments (A→1,B→2)=0.75 vs (A→2,B→1)=1.10, deterministically picks latter — globally most reasonable, order-independent. Matrix solve for 2~5 people is microseconds, no implementation burden.

### 5.4 Honest Boundary

Threshold gate cannot perfectly separate "poorly drawn correct person" vs "low-similarity stranger face" (both low similarity); both score low/0, metric direction correct, attribution coarser. Mitigation: **report match-rate (match success) and matched-similarity (mean similarity when matched) separately**; manual audit can distinguish "could not match" vs "matched but not similar".

> Similarity matrix + one-to-one assignment is standard in MultiHuman-Testbench etc.; our increment = yaw-bucket calibrated threshold gate + tie-in to three-way accounting.

## 5.5 Reference-Side Embedding Freeze (07-13 Addendum: Multi-Face Disambiguation and Reproducibility)

Reference images are often half/full-body crops; nearby faces may intrude (face bleed). Build stage has three gates: ingest largest face + position gate (face must be upper-center / near projected head) + dashboard manual review; final references theoretically single-person clean. But at eval **must not re-run "detect→largest face" each time** — when two faces similar size, random wrong pick; detector version drift breaks third-party reproduction. Rules:

1. **One-time freeze at finalize**: Run detection once per final reference, write selected face box + 512-d embedding to manifest; eval reads frozen vectors only, no detection on references. Deterministic, reproducible; third parties unaffected by detector version after benchmark release.
2. **Multi-face disambiguation without guessing**: Reference belongs to actor cluster with other reference images — **pick face with highest cosine to cluster center**. Intruder is another person, embedding won't match cluster center; criterion almost never wrong.
3. **Verifiable**: Record `n_faces_in_ref` at freeze; >1 flagged on separate review page (volume should be tiny).

Generated side unchanged: full face detect then Hungarian matching; multi-face is normal not exceptional.

## 6. One-Sentence Protocol

> ID score only on persons with "GT visible ∧ generated detected" using raw ArcFace similarity (yaw bucket calibration); "GT visible ∧ generated undetected" scores 0 anti-cheat; "GT not visible" excluded with ReID auxiliary reported separately; also report Face-Visible Rate; multi-person matching = GT side offline freeze + generated side Hungarian global assignment + yaw bucket threshold gate (match then gate), extra hallucinated people handled by Count.

## Implementation Checklist (Tick When Building ArcFace Environment)

- [ ] RetinaFace/SCRFD detection + yaw estimation (detection confidence, yaw dual output)
- [ ] **Finalize freeze reference embeddings**: face box + 512-d vector + `n_faces_in_ref` to manifest, multi-face disambiguate by cluster-center cosine; eval reads frozen vectors only
- [ ] Separate review page for references with `n_faces_in_ref > 1`
- [ ] Pre-run GT target visibility once, persist `gt_visible` in manifest (offline one-time)
- [ ] **`distractor_faces` field in manifest** (unbound bystander face boxes in GT, excluded at eval)
- [ ] Backfill `ref_quality: relaxed` in existing manifest (back_hug/piggyback etc. categories audit)
- [ ] Multi-view calibration script: yaw bucket threshold table on H4D/Panoptic/EgoHumans (also for threshold gate)
- [ ] Hungarian matching (`linear_sum_assignment`) → threshold gate → three-way accounting → Face-Visible Rate / match-rate / matched-similarity summary columns
- [ ] person-ReID / crop-CLIP auxiliary channel (¬gt_visible persons only)
