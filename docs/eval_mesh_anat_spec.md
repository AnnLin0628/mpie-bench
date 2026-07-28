# Mesh Anat Spec: leftover / orphan / P_* / S_anat_mesh

> **Status: 2026-07-16** · **Initial v0.1** (main evaluation table Anat = `S_anat_mesh`)  
> **Consistency (human · mesh binary · VLM same checklist)**: [`eval_human_consistency_anat_inter.md`](eval_human_consistency_anat_inter.md)  

> **Protocol entry**: [`eval_protocol_v3.md`](eval_protocol_v3.md) §3  
> **Inter spec counterpart**: [`eval_mesh_inter_spec.md`](eval_mesh_inter_spec.md)  
> **Code**: `code/eval/anat_extended.py` (`structure_overcount_score`, `compose_anat_score`, `explain_residual_proxy`)  
> **Refresh**: `code/eval/refresh_anat_extra.py` (requires re-running Multi-HMR for j2d)  
> **Frontend**: `eval_outputs/pilot subset_v3/` “Mesh Supplement” section  
> **Illustrative example image**: `arm_around_shoulder__000ad2a1de74__T1.png` (§13 numbers are pedagogical fabrications, not on-disk ground truth)

This document is written at the same level of detail as the Inter spec: what problem it solves, step-by-step algorithm, **formula / meaning / design rationale for each metric**, concrete examples, relationship to the main table, and why it does not pass the gate.

---

## 0. Metric Quick Reference (External Narrative)

| Symbol | One-line meaning | Direction | Weight in total? |
|--------|------------------|-----------|------------------|
| `leftover` | Fraction of foreground near people not covered by skeleton | ↓ | No (diagnostic → `P_area`) |
| `orphan` / `max_blob_frac` | Share of leftover that is a large blob not touching skeleton | ↓ | No (diagnostic → `P_blob`) |
| `P_area` | Area penalty when leftover exceeds baseline | ↓ | Via `P_extra` |
| `P_blob` | Penalty for disconnected large blobs (+ multi-blob) | ↓ | Via `P_extra` |
| **`P_extra`** | Total extra-structure penalty = max(area, blob) | ↓ | **×0.40** |
| **`P_resid`** | Penalty when skeleton bbox misaligns with image person | ↓ | **×0.30** |
| **`P_struct`** | Penalty for absurd two-person scale / limb ownership, etc. | ↓ | **×0.15** |
| **`P_detect`** | Penalty for missed detections (keep &lt; expected count) | ↓ | **×0.15** |
| **`S_anat_mesh`** | `Σ w s` (`s=1−P`) | ↑ | **Scheme ③ main table Anat** |

**Four-penalty division of labor**:

| Penalty | What it checks | What it does not check |
|---------|----------------|------------------------|
| `P_extra` | **Extra-topology structure** such as extra hands/feet, floating fragments | Penetration, skin-tone bleeding |
| `P_resid` | Whether HMR fit **covers** the image person | Whether there is an extra hand |
| `P_struct` | Whether two-person **relative scale / limb ownership** is absurd | Area of floating fake hands |
| `P_detect` | **Whether enough people** are present (count switch) | Anatomical detail |

---

## 1. Problem Statement

VLM Anat saturates on pilot subset (≈0.98–0.99). Early Anat used SMPL-X **joint-angle / bone-length priors**; all models also ≈0.99 — **the prior cannot separate good from bad**.

Root cause:

> SMPL-X **fixes hand/foot topology per person**. Extra hands/feet/heads/half-bodies **are not in parameter space**; no matter how “legal” the joint angles, they cannot be detected.

Therefore mesh Anat asks the image instead:

> **Do these N fitted skeletons explain how much “human mass” appears in the image? Whatever they cannot explain is suspicious extra structure.**

| Error type | Who handles it | Why |
|------------|----------------|-----|
| Extra hands/feet, fragmented torso, limb hallucinations | **Anat leftover/orphan → P_extra** | Extra-topology structure |
| Wrong person / ownership confusion | **P_struct** (ownership) + Inter | Contact band |
| Pathological fusion, should-touch-but-doesn't | **Inter** | See Inter spec |
| Skin-tone bleeding, clothing texture | **VQA** (scheme ②) | Not skeleton topology; mesh ignores |

**Explicitly not done**: per-part counting of hands/feet/heads (combinatorial explosion); “count only YOLO torsos” (five hands still fit in two person boxes).

---

## 2. On-Board Status (Clarify Externally First)

| | Scheme ② main table Anat | Scheme ③ main table Anat (evaluation dashboard) |
|--|--------------------------|----------------------------------|
| Signal | VLM error checklist | leftover → orphan → P_extra → **S_anat_mesh** |
| pilot subset | Saturates ≈0.98 | Replaced in Overall; P_extra ≈0.25 for all models, discrimination still limited |
| Abandon? | — | **No** — continue iterating |

---

## 3. End-to-End Pipeline

```text
Generated image + Multi-HMR (shares keep with Inter)
        │
        ├─ j2d for keep (must map from 896 canvas back to original image)
        │
        ├─ ① Extract human foreground fg
        ├─ ② Draw coarse skeleton + torso explained
        ├─ ③ leftover = (fg ∧ ROI − explained) / fg_roi
        ├─ ④ orphan = disconnected large-enough leftover blobs / fg_roi
        ├─ ⑤ P_extra = max(P_area, P_blob)
        ├─ ⑥ Aux: P_resid / P_struct / P_detect
        └─ ⑦ S_anat_mesh = Σ w_i s_i
```

**Prompt involvement**: Same as Inter — mainly **R# → top-k person count**.  
Contact-intent keywords **do not** participate in Anat.

---

## 4. Prerequisites: Coordinates and Resolution

### 4.1 j2d Coordinate System (Required)

Multi-HMR `j2d` lives on an **896×896 contain+pad canvas**, not original image pixels.  
Must call `j2d_canvas_to_original(j2d, (W,H), 896)` before drawing skeleton.  
If skipped: skeleton draws empty → leftover all wrong → previously `P_extra≡0`.

### 4.2 Working Resolution

Metrics computed on thumbnails with **long edge ≤768** (avoid dilate OOM on large images).  
Joint coordinates scaled by the same `scale`. Field: `work_scale`.

---

## 5. Step 1: Human Foreground `fg`

Function: `_foreground_mask(rgb)`.

| | |
|--|--|
| **Meaning** | Which pixels “look like person, not background” |
| **Design rationale** | No segmentation network — low-cost heuristic; provides “human mass” denominator for leftover |
| **Summary** | Estimate background brightness on four edges; `fg_luma` (luma diff) ∪ (`fg_sat`∧`fg_mid`); avoids old whole-image Otsu mislabeling |
| **Limitations** | Floral clothing, strong shadows, same color as background can fail |

---

## 6. Step 2: Explained Region `explained` (Skeleton)

Function: `_draw_explained_skeleton(j2ds_px, h, w)` (main path).

| | |
|--|--|
| **Meaning** | Pixels that N kept people “claim to occupy” |
| **Design rationale** | SMPL cannot detect extra-topology structure → use 2D skeleton + torso as “explainer”; prefer skeleton over mesh projection (more stable, less memory) |
| **Summary** | Coarse limb segments + shoulder-hip torso quadrilateral + light dilation |

| Symbol | Meaning |
|--------|---------|
| `explained` | bool mask; True = explained by skeleton/torso |
| `j2ds_px` | Joints mapped back to working resolution |

---

## 7. `leftover` — Unexplained Area

### 7.1 Formula

```text
ROI           = skeleton bbox expanded outward by pad, pad ≈ 0.22 × max(h,w) (min 24px)
fg_roi        = fg ∧ ROI
leftover_mask = fg_roi ∧ ¬explained
leftover      = area(leftover_mask) / area(fg_roi)     # ∈ [0,1]
```

Persisted: `anat_leftover_frac`.

### 7.2 Meaning

> Near the people, what fraction of foreground is **not** covered by these N skeletons?

Clothing, hair, body thickness, **and** extra hands all contribute to leftover.

### 7.3 Design Rationale

- Denominator uses `fg_roi` not whole image: distant clutter does not interfere.  
- **Absolute value is not “60% wrong”**: stick-figure skeleton cannot cover clothing; GT leftover is also often ≈0.6.  
- Use: cross-model relative comparison + map to `P_area` (penalize only above baseline).

| Symbol | Meaning | Good/bad |
|--------|---------|----------|
| `ROI` | Look only near people | — |
| `fg_roi` | Human foreground inside ROI | — |
| `leftover_mask` | Foreground pixels skeleton does not cover | — |
| **`leftover`** | Uncovered fraction | **↓ better** |

---

## 8. `orphan` / `max_blob_frac` — Disconnected Fragments

### 8.1 Formula

```text
explained_touch = dilate(explained, touch_r)   # touch_r ≈ 2.2% of edge length, clamped 4–16
For each connected component C in leftover_mask:
  frac = area(C) / area(fg_roi)
  if frac < 0.025          → ignore (dust)
  if C ∩ explained_touch ≠ ∅ → ignore (attached to body: clothing/flesh/uncovered torso)
  else count as orphan

orphan           = Σ area(qualified blobs) / area(fg_roi)
max_blob_frac    = max(frac of each qualified blob)
n_leftover_blobs = count of qualified blobs
```

Persisted: `anat_orphan_frac`, `anat_n_leftover_blobs`; intermediate `max_blob_frac`.

### 8.2 Meaning

> In leftover, **floating, large-enough blobs not touching skeleton** ≈ extra hands/feet drifting nearby.

`orphan ⊆ leftover`. Extra limbs attached to the body often **enter leftover only, not orphan**.

### 8.3 Design Rationale

- leftover is high for everyone → need sharper signal for “fake limbs”.  
- “Does not touch `explained_touch`” separates: clothing edge (attached) vs floating hand (disconnected).  
- `0.025` threshold: filter 1–2% noise blobs.  
- **Known weakness**: real extra hands often attach at waist/sleeve → miss orphan → `P_blob` does not separate on pilot subset.

| Symbol | Meaning | Good/bad |
|--------|---------|----------|
| **`orphan`** | Total share of disconnected extra structure | **↓ better** |
| `max_blob_frac` | Largest disconnected blob share (feeds `P_blob`) | ↓ better |
| `n_leftover_blobs` | Count of disconnected blobs (multi-blob bonus) | More = auxiliary signal |

---

## 9. `P_area` / `P_blob` / `P_extra` — Extra-Structure Penalties

### 9.1 Formula

```text
base, span = 0.48, 0.32
P_area = clip( (leftover - 0.48) / 0.32 , 0, 1 ) × 0.4

P_blob = clip( (max_blob_frac - 0.025) / 0.055 , 0, 1 )
if n_leftover_blobs ≥ 2:
  P_blob = max(P_blob, min(0.65, 0.28 × n_leftover_blobs))

P_extra = max(P_area, P_blob)          # ∈ [0,1], ↓ better
S_overcount = 1 - P_extra              # internal good score, ↑ better
```

Persisted: `P_anat_extra`; intermediates may include `P_extra_area` / `P_extra_blob`.

### 9.2 Meaning

| Symbol | Meaning |
|------|---------|
| `P_area` | “Is unexplained area abnormally high?” — weak signal, amplitude suppressed overall |
| `P_blob` | “Is there a large-enough floating fake limb blob?” — main overcount catcher |
| **`P_extra`** | Max of both paths: heavy penalty if disconnected blob; otherwise only light penalty for high leftover |

### 9.3 Design Rationale

| Constant | Why set this way |
|----------|------------------|
| `base=0.48` | Clean images often have leftover ~0.5; below this `P_area=0`, avoid penalizing everyone |
| `span=0.32` | leftover 0.48→0.80 fills area branch (then ×0.4) |
| `×0.4` | Area branch caps at 0.4 — **clothing/hair alone cannot max out P_extra** |
| `0.025` / `0.055` | Fragments &lt;2.5% ignored; ~8% → full penalty — limb scale |
| `max(·,·)` | Obvious floating fake hand → `P_blob` dominates; else fall back to weak `P_area` |
| Multi-blob bonus | ≥2 disconnected blobs → penalize at least `min(0.65, 0.28×n)` |

**Narrative**: `P_extra` = “suspicious extra structure the skeleton cannot explain”; weight 0.40, main Anat-mesh penalty.

---

## 10. `P_resid` — Fit-to-Image (Residual)

### 10.1 Formula

Function: `explain_residual_proxy`.

```text
bbox      = bounding box of all keep j2d (pad ≈ 0.08× edge length)
For luma / otsu / sat foreground cues, compute IoU(bbox, fg_cue) each
iou       = max(three IoUs)
S_residual = clip( (iou - 0.06) / 0.28 , 0, 1 )
if bbox covers >85% of image: S_residual *= 0.7   # box too large, unreliable

P_resid = 1 - S_residual                   # ∈ [0,1], ↓ better
```

Persisted: `P_anat_resid` / `S_anat_residual`.

### 10.2 Meaning

> **Do person and skeleton align?**  
> Good overlap between joint bbox and “person-like” image region → small residual; bbox drift / person offset → large residual.

**Does not check** extra hands or penetration; only “does this fit cover the person in the image?”

### 10.3 Design Rationale

- After SMPL joint-angle prior saturates, use **image alignment** as second signal.  
- Max over multiple cues: reduce single-style kills (dark/bright/low saturation).  
- Weight 0.30: secondary to `P_extra`; Anat should drop when fit collapses.

---

## 11. `P_struct` — Inter-Person Structure (Scale / Ownership, etc.)

### 11.1 Formula

```text
For each available branch compute good score S ∈ [0,1], then:
P_struct = max( 1-S_scale, 1-S_ownership, 1-S_part_mesh, 1-S_abhuman, … )
if P_struct < 0.02: P_struct = 0     # noise zeroed
```

Main branches (`anat_extended.py`):

| Branch | Good-score function | What it asks |
|--------|---------------------|--------------|
| **Scale** `S_scale` | `cross_person_scale_score` | Do two adult meshes have absurd height/upper-arm ratio? (height ratio ≤1.25 full score, ≥1.55 zero) |
| **Ownership** `S_ownership` | `ownership_confusion_score` | In contact band, are A's mesh points closer to B's limb? (arm seems on wrong person) |
| part_mesh / AbHuman | optional | If present, enter `max` |

Persisted: `P_anat_struct`; diagnostics show `S_anat_scale` / `S_anat_ownership`.

### 11.2 Meaning

> **Is two-person relative structure chaotic?**  
> Not “one extra hand”, but “absurd proportions” or “this limb seems on the wrong person”.

### 11.3 Design Rationale

- `P_extra` catches extra-topology area; contact-band **ownership confusion** needs separate 3D point/limb distance check.  
- Take `max`: any structural problem bad enough raises penalty.  
- Weight 0.15: auxiliary signal, avoid drowning main overcount.

---

## 12. `P_detect` — Missed Detection

### 12.1 Formula

```text
n_expected = count of unique R# in prompt (e.g. R1,R2 → 2)
n_keep     = Multi-HMR top-k kept persons (k = n_expected)
under_detect = (n_keep < n_expected) and not recon_fail

P_detect = 1  if under_detect else 0
```

Persisted: `P_anat_detect` / `under_detect`.

### 12.2 Meaning

> **Are there enough people?** Expected 2 but only 1 kept → leftover / Inter unreliable afterward; record full missed-detection first.

### 12.3 Design Rationale

- Simple hard switch, no soft score.  
- Related to Count axis but does not replace main-table Count; here only makes Anat-mesh drop together on missed persons.  
- Weight 0.15.

**Reconstruction failure** `recon_fail`: entire `S_anat_mesh = 0` (harsher than `P_detect` alone).

---

## 13. Composing `S_anat_mesh`

### 13.1 Formula (Additive Main Formula)

```text
s_extra  = 1 − P_extra
s_resid  = 1 − P_resid
s_struct = 1 − P_struct
s_detect = 1 − P_detect

S_anat_mesh = clip(
  0.40·s_extra + 0.30·s_resid + 0.15·s_struct + 0.15·s_detect,
  0, 1
)
```

Equivalent (code): `1 − 0.40·P_extra − 0.30·P_resid − 0.15·P_struct − 0.15·P_detect`.

| Weight | Default | Meaning |
|--------|---------|---------|
| `w_extra` | 0.40 | **Must catch** extra structure |
| `w_resid` | 0.30 | Fit alignment |
| `w_struct` | 0.15 | Scale/ownership, etc. |
| `w_detect` | 0.15 | Missed persons |

**Higher is better.** Additive main formula, same philosophy as Inter (`Σ w s`; diagnostics still persist `P_*`).

### 13.2 Weight Reallocation When Missing

| Case | Behavior |
|------|----------|
| overcount unavailable (no person / `no_j2d`) | `P_extra` = 0; its weight split to resid/struct |
| residual unavailable | its weight split to extra/struct |
| `recon_fail` | `S_anat_mesh=0` |

### 13.3 Relationship Diagram

```text
Image + Multi-HMR j2d (keep)
        │
        ├─→ (separate path) Inter_mesh     ← unrelated to leftover
        │
        └─→ fg + explained
               │
               ├─ leftover ──→ P_area ──┐
               │                        ├─→ s_extra=1−P_extra ─×0.40─┐
               └─ orphan  ──→ P_blob ──┘                              │
                                                                      │
        Joint bbox vs foreground IoU ──→ s_resid ─×0.30────────────────┤
        Scale / ownership / … ─────────→ s_struct ─×0.15────────────────┤
        keep < R# ─────────────────────→ s_detect ─×0.15────────────────┤
                                                                      ▼
                                           S_anat_mesh = Σ w s
```

---

## 14. Concrete Example (Pedagogical): `arm_around_shoulder__000ad2a1de74__T1.png`

> **Pixel counts and scores below are fabricated** for causal explanation; do not treat as on-disk ground truth for this image.

### 14.1 What the Scene Checks

Two people shoulder-to-shoulder embrace (zebra dress dark skin + white dress red hair). Messy hands at waist: multiple dark-skinned hands, fused with red-haired hand; left waist often has a **seemingly floating** hand.

| | Expected (R#=2) |
|--|-----------------|
| keep | 2 skeletons |
| Extra-topology risk | Extra hand flesh at waist → leftover; floating hand at left waist → orphan |

### 14.2 Fabricated Intermediates

```text
|fg_roi|              = 12000
Skeleton+torso covers fg =  4800
leftover pixels         =  7200
  └─ attached clothing/hair + fused hands =  6300   # touches explained_touch → not orphan
  └─ floating hand at left waist          =   900   # qualified orphan

leftover      = 7200/12000 = 0.60
max_blob_frac =  900/12000 = 0.075
orphan        = 0.075
n_blobs       = 1
```

### 14.3 Map Penalties and Compose

```text
P_area = clip((0.60-0.48)/0.32,0,1)×0.4 = 0.15
P_blob = clip((0.075-0.025)/0.055,0,1)   = 0.91
P_extra = max(0.15, 0.91)                = 0.91

# Aux (fabricated): fit OK, ownership slightly messy, count OK
P_resid  = 0.18
P_struct = 0.05
P_detect = 0

S_anat_mesh = 0.40×(1−0.91) + 0.30×(1−0.18) + 0.15×(1−0.05) + 0.15×1
            ≈ 0.40×0.09 + 0.30×0.82 + 0.15×0.95 + 0.15
            ≈ 0.57
# Equivalent: 1 − 0.40×0.91 − 0.30×0.18 − 0.15×0.05 ≈ 0.57
```

### 14.4 Contrast: Fake Hand Attached, Misses orphan

Same leftover=0.60, but `max_blob_frac=0`:

```text
P_extra = 0.15
S_anat_mesh = 0.40×0.85 + 0.30×0.82 + 0.15×0.95 + 0.15×1 ≈ 0.88
```

| | Floating hand counts as orphan | All fake hands attached |
|--|--------------------------------|-------------------------|
| leftover | 0.60 | 0.60 |
| P_extra | **0.91** | **0.15** |
| S_anat_mesh | **≈0.57** | **≈0.88** |

→ Shows current pipeline: **separation relies on disconnected blobs**; attached messy hands leave only weak `P_area` — one root cause of not passing the gate.

### 14.5 Pure Formula Mini-Example (No Image)

No orphan, other P=0, `leftover=0.60`:

```text
P_extra = 0.15 → S_anat_mesh = 1 - 0.40×0.15 = 0.94
```

When `max_blob_frac=0.08`: `P_blob=1` → `P_extra=1` → `S=0.60` (ignore other P).

---

## 15. Why pilot subset Does Not Pass the Gate (Record)

| Phenomenon | Meaning |
|------------|---------|
| All models `P_extra`≈0.24–0.30 | Extra-structure signal barely separates |
| leftover ≈0.59–0.68 for all | Limited skeleton explanatory power; still clustered after baseline calibration |
| orphan mean ≈0.02 | Real extra hands often attached → miss orphan; or fg fusion |
| Flux S_anat_mesh > GT | Conflicts with “GT should be higher” |

→ **Appendix iteration**; direction kept (unified leftover, no per-part counting).  
Possible improvements: better foreground/human parsing, mesh projection silhouette, other cues for attached extra limbs, fusion with VLM Anat, etc. (TBD).

---

## 16. Persisted Fields

| Field | Meaning |
|-------|---------|
| `anat_leftover_frac` | leftover |
| `anat_orphan_frac` | orphan |
| `anat_n_leftover_blobs` | disconnected blob count |
| `P_anat_extra` | P_extra |
| `P_anat_resid` / `P_anat_struct` / `P_anat_detect` | other penalties |
| `S_anat_overcount` | `1 - P_extra` (internal good score) |
| `S_anat_residual` / `S_anat_scale` / `S_anat_ownership` | auxiliary branch good scores |
| `S_anat_mesh` | final mesh anatomy score |
| `anat_overcount_note` | e.g. `unexplained_via_skeleton_torso` / `no_j2d` |
| `anat_scene.structure_overcount` | step-by-step intermediates (incl. `max_blob_frac`, `P_extra_area`, etc.) |

---

## 17. Division of Labor with Inter (For Explanation)

| | Inter | Anat (mesh) |
|--|-------|-------------|
| Question | Are inter-person geometric relations correct? | Are there too many / chaotic limb structures in the image? |
| Main signal | Penetration volume, surface distance | Foreground − skeleton → P_extra |
| Read contact intent? | Yes (keywords, TBD optimization) | No |
| Main table | **Replaced** Inter | **Replaced** Anat (= S_anat_mesh) |

---

## 18. Related Files

| Path | Role |
|------|------|
| `anat_extended.py` | leftover/orphan/P_extra/P_resid/P_struct/compose |
| `refresh_anat_extra.py` | refresh Anat only (re-run HMR) |
| `mesh_metrics.py` | aggregate into sample json |
| [`eval_protocol_v3.md`](eval_protocol_v3.md) | scheme ③ overview + pilot subset runs |
