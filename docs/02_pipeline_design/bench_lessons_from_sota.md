# Lessons from Top Benches · MPIE Improvements and Follow-Up Work

> **Status: 2026-07-17** · Benchmarks compared: Qwen-Image-Bench, ImgEdit / GEdit, Boogu Arena (and Boogu's self-critique of ImgEdit)  
> Goal: turn "what to learn" into executable items, not narrative only. Overall progress: [`../PROGRESS.md`](../PROGRESS.md).

---

## 0. One-line positioning (pin down before borrowing)

| Bench | Gap filled | Source of discrimination |
|---|---|---|
| Qwen-Image-Bench | Frontier T2I models saturate old leaderboards | Real-world Fidelity + Creative Generation |
| ImgEdit / GEdit | No unified standard for general instruction editing | Multi-task / real user requests + LLM/specialized judge |
| **MPIE-Bench** | **Anatomy/geometry of multi-person contact editing** | **C0–C3 mesh Anat/Inter + anti face-hiding ID** |

Borrowing principle: **learn engineering and narrative for "becoming the default reference," not competing on text rendering / generic edit Overall.**

---

## 1. What top benches do well (transferable principles)

### 1.1 Measure failure modes that are not yet saturated
- Qwen: not "can it draw a cat," but world fidelity / creativity / contact-anatomy ceiling.
- ImgEdit/GEdit: instruction editing and real requests, not CLIP reconstruction.
- **MPIE already aligned**: VLM Anat/Inter saturated → switch to Multi-HMR; no six-axis Overall.

### 1.2 Layered taxonomy + attributable diagnosis
- Qwen: 5 pillars → 23 sub-skills → 56 rubrics; one image, many axes, clear "where it fails."
- **MPIE already has**: six axes + 14 categories × C0–C3.  
- **Gap**: main paper/figures still lean on means; **degradation curves and per-category breakdown** are not yet default "must-report" views.

### 1.3 Judges anchored to humans, with rubricization
- Qwen: 80 professional annotators → Q-Judger; fine-grained 0/1/2/N/A.
- ImgEdit: specialized judge + multi-dimensional scoring.
- **MPIE already designed**: H / Mesh-checklist / VLM-checklist three-way κ.  
- **Gap**: engineering is ready but **humans not labeled, τ not locked, κ table empty** — one of the largest unfinished credibility items for the paper.

### 1.4 Prompt / sample factory bound to capability units
- Qwen: each prompt deliberately covers ≥3–4 facets; short/long and EN/ZH tiers.
- GEdit: real edit requests from Reddit, etc.
- **MPIE already has**: English edit prompts + `(R#)` anchors + scene-level split.  
- **Gap**: contact intent is still **keyword heuristics** (already in limitations); should move to gold labels / classifier.

### 1.5 Frozen reproducibility pack → network effect
- Qwen: prompts + judge reproducible offline; leaderboard after training freeze to reduce leakage.
- **MPIE protocol already written**: reference embedding finalize freeze, API/weight snapshot dates.  
- **Gap**: manifest freeze, third-party one-command eval subpack, version pinning — **not yet delivered**.

### 1.6 Honest boundaries (increase citation trust)
- Boogu proactively says ImgEdit does not fully align with human preference and In-Context coverage is insufficient.
- **MPIE should hold**: Inter>GT does not mean better semantics; mesh is a proxy for out-of-topology structure; photorealistic domain only. State clearly in main text; do not let a single Arena-style ELO dominate.

### 1.7 What to borrow cautiously (explicitly not doing)
| Practice | Reason |
|---|---|
| Build "MPIE Arena" as main table | Boogu Arena is weak; lacks third-party anchoring |
| Copy ImgEdit's huge edit subtask sprawl | Dilutes contact-geometry uniqueness |
| Copy Qwen's 56 aesthetic rubrics | Not differentiated; HPSv2 is enough for Qual |
| Bare VLM Overall / weighted total score | Anat/Inter saturation proven; dilutes failures |

---

## 2. MPIE gap checklist (learn → change)

| # | Learned | Current state | Action | Priority |
|---|---|---|---|---|
| A | New-axis narrative must be unavoidable | Abstract has it; related work/main figure not hard enough | related-work table + teaser failures + fixed slogan | P0 paper |
| B | Difficulty stratification is discrimination evidence | C0–C3 protocol exists; full table not out | main figure = degradation curves; mean secondary | P0 experiment |
| C | Human–machine consistency is a first-class contribution | scripts ready, humans not run | recruit → pilot lock τ → fill κ table | P0 credibility |
| D | Frozen pack drives community cross-reporting | protocol exists, pack incomplete | finalize embedding + one-command eval + version pin | P0 release |
| E | Intent/label gold standardization | contact intent = keywords | gold label or classifier into `_split`/manifest | P1 metrics |
| F | Anat must separate models | `P_extra` still clustered | strengthen leftover / foreground; VLM appendix comparison | P1 metrics |
| G | Model lineup aligned with "2026 edit frontier" | zoo missing FireRed; FLUX.2/Step1X candidates | add 1–2 multi-ref editors after eval | P1 baselines |
| H | Leakage and time declaration | paper TODO | API/weight snapshot dates, no-training statement in appendix | P1 paper |
| I | Restraint on self-built preference leaderboard | no Arena (good) | keep; at most appendix preference spot-check | — |

---

## 3. Work that can be added now (sorted by "do immediately")

> Aligned with `PROGRESS.md` blockers: mesh_v3 sync, recruiting, gateway env, open-source full image generation.

### P0 — push this week (blocks main-paper credibility / main table)

1. **Run human consistency end-to-end (H↔M↔V)**  
   - Runbook: `eval_human_consistency_runbook.md`  
   - Actions: sync `mesh_v3` → configure gateway and run V → recruit 3 annotators for pilot100 → lock τ → fill κ table in `05_experiments`  
   - Output: human-consistency section in paper can be filled with numbers; prove "mesh beats saturated VLM"

2. **Open-source 7 models full 2500 generation + scoring**  
   - Smoke green → `run_opensource_full_8gpu.sh all`  
   - Then: mesh / ArcFace / HPSv2 / VLM Count+Instr → replace smoke in main table  
   - Output: `2500×10` main table + **C0–C3 degradation-curve main figure**

3. **Release-grade frozen-pack skeleton**  
   - [ ] finalize: reference face boxes + ArcFace embeddings in manifest (`eval_id_protocol.md` §5.5)  
   - [ ] pin: Multi-HMR ckpt hash, ArcFace/HPSv2 versions, judge model name, API snapshot dates  
   - [ ] minimal `README_eval.md`: env vars + one command for six axes  
   - Output: third-party reproducibility (prerequisite for Qwen/ImgEdit network effect)

4. **Paper narrative reinforcement (low compute, high payoff)**  
   - related: ImgEdit / GEdit / MultiHuman / ASAP comparison — **who tests interaction, anatomy, contact geometry**  
   - intro/teaser: 2–3 "VLM high score but fused limbs" failure cases  
   - fixed slogan: *Existing edit benches miss contact-time anatomy.*  
   - **No** Overall; main figure emphasizes C0–C3 slope

### P1 — metric and lineup strengthening (parallel with main table)

5. **Contact-intent gold labels**  
   - Manually label `required|forbidden|unspecified` on pilot subset / consistency split first  
   - Replace keyword heuristics; write to pack fields and freeze  
   - Matches limitations TODO

6. **Anat discrimination iteration**  
   - For cross-model `P_extra` clustering: better human foreground, close-body extra-limb case set  
   - Goal: model ranking on smoke/full aligns better with human "extra limb" perception

7. **Open-source zoo additions (multi-reference editing)**  
   | Candidate | Rationale | Notes |
   |---|---|---|
   | **FireRed-Image-Edit-1.1** | Multi-element fusion + identity; beats Qwen-Edit-2511 on leaderboard | **Highest-priority new integration** |
   | FLUX.2-dev / klein | Official multi-ref; compare with Kontext | VRAM; dev non-commercial license |
   | Step1X-Edit | Already on candidate list | Lower priority than FireRed |
   | Boogu-Edit | Strong editing but **single reference only** | Not in main table until multi-ref release |

8. **Leakage / time-freeze statement**  
   - Appendix: test set must not enter training after release; model weight/API dates  
   - Align with Qwen "evaluate after data freeze" credibility wording

### P2 — if capacity allows

9. Rebalance categories (fight/restrain etc. underrepresented in C3) — only if main curves are challenged for class imbalance  
10. Appendix: small blind preference study (not Arena main leaderboard)  
11. contact-intent classifier to replace manual gold labels (scale)

---

## 4. Suggested two-week rhythm (usable as schedule)

| Window | Focus | Definition of done |
|---|---|---|
| D1–D3 | mesh sync + open-source full run + recruiting | generation started; pilot annotation started |
| D4–D7 | V batch + human pilot + lock τ | κ table fillable v1; consistency section no longer TODO |
| D1–D14 parallel | full scoring + C0–C3 figure + related/teaser | smoke main table replaceable by full draft |
| interleaved | FireRed runner skeleton; embedding finalize | zoo update; frozen-pack checklist half done |

---

## 5. Links to existing docs

| Topic | Authoritative doc |
|---|---|
| Main-table definition | `eval_protocol_v3.md` |
| Human consistency | `eval_human_consistency_anat_inter.md` + `eval_human_consistency_runbook.md` |
| Model list | `eval_model_zoo.md` (add FireRed candidate; mark Boogu "single-ref, watch") |
| ID freeze | `eval_id_protocol.md` §5.5 |
| Overall progress | `../PROGRESS.md` |

---

## 6. Change log

- **2026-07-17**: Initial version. Sources: Qwen-Image-Bench / ImgEdit_O / Arena used by Boogu-Image-0.1, plus ImgEdit, GEdit, Qwen-Image-Bench design principles compared with MPIE status.
