# Mesh Inter Spec: Full Pipeline for S_inter_mesh

> **Status: 2026-07-16** · **Initial v0.1** (matches code; main evaluation table Inter = `S_inter_mesh`)  
> **Consistency (human · mesh binary · VLM same checklist)**: [`eval_human_consistency_anat_inter.md`](eval_human_consistency_anat_inter.md)  

> **Protocol entry**: [`eval_protocol_v3.md`](eval_protocol_v3.md) §2  
> **Code**: `code/eval/mesh_metrics.py` (`compose_inter_score`, `prompt_contact_intent`, `_ramp_high_good`)  
> **Calibration**: `code/eval/rescore_mesh_inter.py` (`calibrate_from_gt`)  
> **Main table**: `eval_outputs/pilot subset_v3/` **Inter** column = this document's `S_inter_mesh`

This document makes every symbol, every step, and when the prompt participates explicit. After reading, you should be able to explain the rationale independently.

---

## 1. Problem Statement

VLM “does it look like interaction?” tends to score everyone high. Mesh Inter instead asks:

> Use 3D human meshes to check: **Is there pathological penetration?** And per **this edit prompt**, **should there be contact?** — should hug but too far, should not touch but glued together, both penalized.

Design principles:

1. **Additive scoring (main formula)**: Map geometry to unit good scores `s_*∈[0,1]`, then weighted sum by intent `S=Σ w_i s_i`. Diagnostics still use `P_*=1−s_*`; numerically equivalent to `1−Σ w P`.  
2. **Intent from prompt text**, not dataset category `cat` (category names often disagree with caption).  
3. **Score only people named in prompt** (`R1/R2/…`), avoid bystander interference in penetration/distance.  
4. **Do not reward “standing apart” itself**: less penetration can beat GT — known bias; narrative is diagnosing unreasonable interaction, not contact reward.

---

## 2. End-to-End Pipeline (Prompt Involved in Two Places Only)

```text
Edit prompt + generated image
        │
        ├─【Prompt involvement ①】Count R# → n_expected
        │         │
        │         ▼
        │   Multi-HMR detects multiple people → take top-k by score (k=n_expected)
        │         │
        │         ▼
        │   Compute mesh geometry on kept people (prompt not read here)
        │         · pen_volume_m3
        │         · min_surf_dist
        │         │
        ├─【Prompt involvement ②】Keywords → contact_intent
        │         │
        │         ▼
        │   Select which s_* to weight by intent, compose S_inter_mesh
        ▼
```

| Stage | Read prompt? | Role |
|-------|--------------|------|
| Multi-HMR forward | No | Image only |
| Penetration volume / surface distance | No | Kept mesh only |
| Select keep count | **Yes** | `n_expected = \|{Rk}\|` |
| Compose score, select additive terms | **Yes** | `required` / `forbidden` / `unspecified` |

**Anat leftover pipeline does not read contact intent**; prompt affects Anat mainly indirectly via “how many people to score”.

---

## 3. Prompt Involvement ①: Person Selection (R#)

### 3.1 Rules

```text
n_expected = |{ Rk | Rk appears in prompt }|     # at least 1
keep = top k detections by Multi-HMR score, k = n_expected
```

Fields:

| Field | Meaning |
|-------|---------|
| `n_detected_raw` | Raw detection count (may include bystanders) |
| `n_humans` | Count after keep |
| `person_select` | `top_k_by_score_vs_prompt_R` |
| `keep_idx` | Kept detection indices |

### 3.2 Example

Prompt:

> Keep **R1** and **R2**. **R1** puts an **arm around** **R2**'s shoulder.

→ R1, R2 appear → `n_expected=2` → keep top-2 scored detections before computing Inter.

If 4 people detected (incl. bystanders), bystanders do not enter `pen_volume` / `min_surf_dist`.

---

## 4. Prompt Involvement ②: Contact Intent

### 4.1 Three Values

Function: `prompt_contact_intent(prompt) -> str`

| Return | Meaning | What gets weighted at compose time |
|--------|---------|-------------------------------------|
| `required` | Prompt asserts body contact | `0.55 s_pen + 0.45 s_prox` |
| `forbidden` | Prompt asserts no contact | `0.55 s_pen + 0.45 s_clear` |
| `unspecified` | No explicit contact/forbidden language | **Only** `s_pen` |

Decision logic (summary):

1. Regex counts “contact-class” hits `n_c` and “forbidden-contact-class” hits `n_f`.  
2. If `n_f>0` and `n_c==0` → `forbidden`.  
3. If `n_c≥1` → `required` (when both contact and forbidden words appear: implementation checks forbidden only when no contact words).  
4. Else → `unspecified`.

Contact-class examples (incomplete, see code `_CONTACT_RES`):  
`hug`, `embrace`, `hold hands`, `arm around`, `around … shoulder`, `wrestle`, `grapple`, `carry`, `kiss`, …

Forbidden-class examples (`_FORBIDDEN_RES`):  
`no contact`, `not touching`, `standing apart`, `keep distance`, …

**Deliberately not using `cat`**: upstream taxonomy often contradicts caption.

### 4.2 Same Geometry, Different Prompt (Shows “Intent Only Affects Weighting”)

Same image, same mesh (`pen_volume`, `min_surf_dist` unchanged):

| Prompt gist | intent | Compose |
|-------------|--------|---------|
| R1 arm around R2 | required | `0.55 s_pen + 0.45 s_prox` |
| R1 and R2 standing apart, no contact | forbidden | `0.55 s_pen + 0.45 s_clear` |
| R1 and R2 face each other (no contact words) | unspecified | `s_pen` |

Whether distance is penalized is entirely prompt-driven; geometric measurement unchanged.

---

## 5. Geometric Quantities (Prompt Not Read)

Computed on **kept people** (usually 2; notation assumes pair).

### 5.1 `pen_volume_m3` (Penetration Volume, m³)

**Intuition**: How much of one person's body “grows into” the other.

**Procedure (summary)**:

1. For kept mesh pair A, B.  
2. Use `trimesh.contains` to see how many A vertices lie inside B → ratio × volume scale → penetration volume contribution; swap A↔B and combine.  
3. Scalar `pen_volume_m3`.  
4. If volume method fails → fall back to `pen_vert_ratio` (fraction of vertices inside other), different `ok/bad` thresholds later, but **same P_fuse form**.

**Larger is worse** (more severe fusion).

### 5.2 `min_surf_dist` (Minimum Surface Distance, m)

**Intuition**: How close are the two “skins”?

**Procedure**: Minimum distance between mesh surfaces.  
- Very small ≈ touching or already penetrating  
- Very large ≈ far apart  

**For required**: smaller is better (should contact).  
**For forbidden**: smaller is worse (should not touch).

---

## 6. Helper `ramp` / `_ramp_high_good`

Code name: `_ramp_high_good(x, lo, hi)`. In protocol and explanation, abbreviated **`ramp(x; lo, hi)`**.

### 6.1 Definition

Map raw quantity `x` where **smaller is better** to **good score** \(s \in [0,1]\):

```text
If x is NaN:     s = 0
If x ≤ lo:       s = 1          # still acceptable
If x ≥ hi:       s = 0          # fully bad
If lo < x < hi:  s = 1 - (x - lo) / (hi - lo)   # linear decay
```

```text
s
1 |----\
  |     \
0 |------\____
     lo   hi     → x (larger is worse)
```

### 6.2 Why ramp Instead of Hard Threshold

Hard threshold (`<ok→full, else 0`) jumps near boundary. Linear transition is smoother and suits GT percentile calibration of `lo/hi`.

### 6.3 Relationship to Penalty

For “smaller is better” quantities:

```text
P = 1 - s = 1 - ramp(x; lo, hi)
```

| s (good score) | P (penalty) |
|----------------|-------------|
| 1 | 0 |
| 0 | 1 |
| 0.5 | 0.5 |

---

## 7. Calibration Thresholds: `vol_ok`, `vol_bad`, `d_good`, `d_fail`

Source: `rescore_mesh_inter.calibrate_from_gt`, statistics on GT subset mesh judgments, written to `_calibration.json` for all-model rescore.

### 7.1 Volume Thresholds (Penetration)

Collect all available `pen_volume_m3` on GT → list `vols`:

| Symbol | Formula (code) | Meaning |
|--------|----------------|---------|
| **`vol_ok`** | `max(0.01, percentile(vols, 55%))` | “Most GT penetration below this” → **acceptable upper bound** (ramp lo) |
| **`vol_bad`** | `max(vol_ok×1.8, percentile(vols, 95%)×1.15)` | “Few GT this bad” → **fully bad start** (ramp hi) |

Default when no volume samples: `vol_ok=0.05`, `vol_bad=0.15`.

### 7.2 Distance Thresholds (Contact Proximity)

Collect only GT `min_surf_dist` where **`contact_intent==required`** and distance valid → `dists`:

| Symbol | Formula (code) | Meaning |
|--------|----------------|---------|
| **`d_good`** | `clip(percentile(dists, 80%), 0.03, 0.12)` | “When contact required, most GT close enough” → distance lo |
| **`d_fail`** | `clip(max(d_good+0.15, percentile(dists,98%)+0.15), …, 0.60)` | “Beyond this = failed contact” → distance hi |

Default when no distance samples: `d_good=0.05`, `d_fail=0.40` (meters).

### 7.3 Why GT Percentiles

Reconstruction always has slight false penetration; absolute m³ thresholds unstable. **Real GT distribution** anchors “normal band” vs “pathological band”.

---

## 8. Three Penalties P_* (Letter by Letter)

Compute good scores first, then `P = 1 - s` (unless noted).

### 8.1 `P_fuse` — Pathological Fusion (Almost Always On)

```text
S_pen  = ramp(pen_volume_m3; lo=vol_ok, hi=vol_bad)
P_fuse = 1 - S_pen
```

| Symbol | Meaning |
|--------|---------|
| `pen_volume_m3` | Penetration volume |
| `vol_ok` | Acceptable penetration upper bound |
| `vol_bad` | Fully bad penetration start |
| `S_pen` | Penetration term “good score” (internal) |
| `P_fuse` | Penalty of **fuse** (fusion penalty), ∈[0,1], larger is worse |

If using `pen_vert_ratio` fallback: `ok = tau_pen×0.35`, `bad = tau_pen`, same form.

**Why needed**: Mesh embedded in other body = interaction geometry failure, independent of “should hug”; `unspecified` also penalized.

---

### 8.2 `P_miss` — Should Contact but Too Far (Only `required`)

```text
S_prox = ramp(min_surf_dist; lo=d_good, hi=d_fail)
P_miss = 1 - S_prox
```

| Symbol | Meaning |
|--------|---------|
| `min_surf_dist` | Minimum surface distance (meters) |
| `d_good` | “Close enough” upper bound |
| `d_fail` | “Too far” start |
| `S_prox` | Proximity term good score (internal) |
| `P_miss` | Penalty of **miss**ing contact, ∈[0,1] |

`forbidden` / `unspecified`: **not computed** (set 0, excluded from compose).

**Why needed**: Prompt requires hug / arm around but people far apart → interaction failure.

---

### 8.3 `P_unwanted` — Should Not Contact but Too Close (Only `forbidden`)

```text
# S_prox = ramp(distance): larger when closer (for required = “close enough” good score)
# forbidden: same quantity as “too close” penalty:
P_near     = S_prox
P_unwanted = max(P_near, P_fuse) = max(S_prox, P_fuse)
```

| Symbol | Meaning |
|--------|---------|
| `S_prox` / `P_near` | Larger when closer; under forbidden = “too close” penalty |
| `P_unwanted` | Penalty of **unwanted** contact |
| `max(S_prox, P_fuse)` | Too close or already fused — take worse |

**Note**: Do not write `1 - S_prox` — that is required's `P_miss` (penalize too far).  
Old implementation wrongly used `1-S_prox`, penalizing correct standing apart; fixed 2026-07-16 to `max(S_prox, P_fuse)`.

`required` / `unspecified`: **not computed**.

**Why max**: Distance alone may miss penetration; penetration alone may miss “very close but contains not exploding”. Both mean “should not touch” — merge with max, not double-weight with fuse.

---

## 9. Composed Score `S_inter_mesh`

### 9.1 Good Scores and Weights

| Symbol | Definition | Meaning |
|------|------|------|
| `s_pen` | `= S_pen = 1 − P_fuse` | Low penetration is better |
| `s_prox` | `= S_prox = 1 − P_miss` | Under required, closer is better |
| `s_clear` | `= 1 − P_unwanted = 1 − max(s_prox, P_fuse)` | Under forbidden, separation/non-fusion is better |
| `w_pen` / `w_fuse` | **0.55** | Penetration weight |
| `w_prox` / `w_miss` | **0.45** | Required proximity weight |
| `w_clear` / `w_unwanted` | **0.45** | Forbidden separation weight |

`0.55+0.45=1`: when both terms are 0, total score is 0 — clean scale.  
Penetration slightly heavier than distance: pathological interpenetration usually worse than “slightly too far”.

### 9.2 Branch by intent (Main Formula = Additive)

**A. `required`**

```text
S_inter_mesh = clip(0.55·s_pen + 0.45·s_prox, 0, 1)
```

Equivalent (code): `1 − 0.55·P_fuse − 0.45·P_miss`.

**B. `forbidden`**

```text
S_inter_mesh = clip(0.55·s_pen + 0.45·s_clear, 0, 1)
```

Equivalent: `1 − 0.55·P_fuse − 0.45·P_unwanted`.

**C. `unspecified`**

```text
S_inter_mesh = clip(s_pen, 0, 1)    # penetration only
```

No “must be close” or “must be apart” — prompt did not say.

### 9.3 Relationship to Penalty Form

```text
S = Σ w_i s_i  =  1 − Σ w_i P_i    (when Σ w = 1 and s_i = 1 − P_i)
```

| Form | Role |
|------|------|
| `Σ w s` | **Main formula / paper and protocol body** |
| `1 − Σ w P` | Code persistence and diagnostic expansion (`P_fuse` / `P_miss` / `P_unwanted`) |
| `clip(...,0,1)` | Numerical guard; normal path stays in [0,1] by construction |

### 9.4 `under_detect`

If kept count below expected (missed detection):

```text
S_inter_mesh ← S_inter_mesh × 0.5
```

Interaction geometry unreliable when people incomplete — down-weight.

### 9.5 Why Weighted Additive, Not Naive `mean(s_pen, s_prox)`

- `unspecified` has no meaningful distance term; main formula reduces to single `s_pen`.  
- Equal-weight mean lets “bad fusion but OK distance” wash each other out; weighted sum pulls total down by bad term weight.  
- Additive narrative clearer: each term is “how good”, then compose by intent.
---

## 10. Numeric Walkthrough (required)

Calibration: `vol_ok=0.05`, `vol_bad=0.15`, `d_good=0.05`, `d_fail=0.40`.  
Prompt: `R1` arm around `R2` → `required`.

| Raw | ramp good score | Penalty | Weighted penalty |
|-----|-----------------|---------|------------------|
| `pen_volume=0.05` | S_pen=1 | P_fuse=0 | 0.55×0=0 |
| `pen_volume=0.10` | S_pen=0.5 | P_fuse=0.5 | 0.275 |
| `pen_volume=0.15` | S_pen=0 | P_fuse=1 | 0.55 |
| `min_surf_dist=0.05` | S_prox=1 | P_miss=0 | 0 |
| `min_surf_dist=0.225` | S_prox=0.5 | P_miss=0.5 | 0.225 |
| `min_surf_dist=0.40` | S_prox=0 | P_miss=1 | 0.45 |

If simultaneously `P_fuse=0.5` and `P_miss=0.5` (i.e. `s_pen=s_prox=0.5`):

```text
S_inter_mesh = 0.55×0.5 + 0.45×0.5 = 0.50
# Equivalent: 1 − 0.275 − 0.225 = 0.50
```

---

## 11. Full Example (Both Prompt Involvements)

**Prompt:**

> Keep R1 and R2. R1 puts an arm around R2's shoulder. Preserve identities.

**① Person selection**

- Parse R1, R2 → `n_expected=2`  
- Multi-HMR detects 3 → keep top-2  

**② Geometry (prompt not read)**

- For these 2: `pen_volume_m3=0.08`, `min_surf_dist=0.12`  

**③ Intent**

- Hit `arm around` → `contact_intent=required`  

**④ Penalties**

- `S_pen = ramp(0.08; 0.05, 0.15) = 1 - (0.08-0.05)/(0.15-0.05) = 0.70`  
  → `P_fuse = 0.30`  
- `S_prox = ramp(0.12; 0.05, 0.40) = 1 - (0.12-0.05)/(0.40-0.05) ≈ 0.80`  
  → `P_miss ≈ 0.20`  

**⑤ Compose**

```text
s_pen = 0.70,  s_prox ≈ 0.80
S_inter_mesh = 0.55×0.70 + 0.45×0.80 = 0.745
# Equivalent: 1 − 0.55×0.30 − 0.45×0.20 = 0.745
```

If prompt changed to “standing apart, no contact”, same geometry takes `forbidden` branch with `s_clear` (`1−P_unwanted`) instead of `s_prox` — score generally different.

---

## 12. Persisted Fields (Judgment json)

| Field | Meaning |
|-------|---------|
| `contact_intent` | required / forbidden / unspecified |
| `pen_volume_m3` / `pen_vert_ratio` | Raw penetration quantities |
| `min_surf_dist` | Surface distance |
| `P_fuse` / `P_miss` / `P_unwanted` | Three penalties |
| `S_prox` / `S_pen` | Internal good scores (debug) |
| `vol_ok` / `vol_bad` / `d_good` / `d_fail` | Calibration used for this sample |
| `w_fuse` / `w_miss` / `w_unwanted` | Weights |
| `S_inter_mesh` | **Final interaction score** |
| `inter_regime` | e.g. `prompt_contact_required` |
| `n_expected` / `n_humans` / `under_detect` | Selection and missed detection |

Calibration file: `$PACK/judgments/mesh_v3/_calibration.json`.

---

## 13. Known Limitations (Must Mention When Explaining)

1. **Generated image Inter can exceed GT**: less penetration ≠ edit semantics more correct.  
2. **top-k ≠ identity alignment**: not guaranteed keep[0] is R1.  
3. **Penetration volume is approximate** (contains + scale), not CAD boolean.  
4. **Contact intent via prompt keyword regex (weak heuristic)** — **TBD optimization (recorded, 07-16)**  
   - Current: `prompt_contact_intent()` scans contact/forbidden word lists → required/forbidden/unspecified.  
   - Issues: missed recall, false recall, EN/ZH variants, compound sentences; word lists cannot be semantically precise. pilot subset `forbidden=0` is one example.  
   - **Future direction (discuss with team first)**: per-sample gold `contact_intent` field preferred; or small model/VLM 3-way classifier; keywords as fallback only.  
   - **Current strategy**: geometry pipeline usable; externally state intent is heuristic, not precise NLU.  
5. **Single-person keep**: when two-person geometry inapplicable, score semantics weaken (relies on under_detect, etc.).

---

## 14. Related Files

| Path | Role |
|------|------|
| `code/eval/mesh_metrics.py` | geometry + intent + compose |
| `code/eval/score_mesh_v3.py` | inference write json |
| `code/eval/rescore_mesh_inter.py` | GT calibration + offline Inter rescore |
| [`eval_protocol_v3.md`](eval_protocol_v3.md) | scheme ③ overview + pilot subset runs |
