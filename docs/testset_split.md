# Test Set Split Protocol (Benchmark Split)

> **Status: Finalized (2026-07-14); sample pool verified (2026-07-15).** Test set first — build benchmark to evaluate closed/open models; training-set human curation deferred.
> Frontend live: per-scene dashboard "Move to test set" button + `/testset` overview (quota progress aggregated in real time).
> **Actual assignment: 2500 target images / 405 scenes** (same definition as `_test_split`; +150 over design quota of 2350 total; class balance may still be uneven).

## 1. Core Principles

1. **Split unit = scene/video, never frames**. Adjacent frames in the same video are near-duplicates; frame-level split = leakage. Multi-cam data (H4D/CHI3D/Panoptic): all views in a sequence count as one unit.
2. **License determines membership**: AVA / K700 (restricted tier, pixels not redistributable) **training only, never in public benchmark**; public test set only from distributable sources — CC0 Pexels (primary), Harmony4D / CHI3D / Panoptic / EgoHumans (academic licenses per terms; worst case release ID + frame index + reconstruction script).
3. **Test set size 2~3k target images**; this phase draws from **Harmony4D + CC0 + CHI3D**.
4. Test set **100% human-reviewed** (dashboard delete/star/bind clean before assignment); training set tolerates noise.
5. Same-person leakage scan: CC0 professional models recur across videos; before finalize, ArcFace actor clusters scanned cross train/test — same person moved entirely to training side.
6. H4D sequences assigned to test **excluded from yaw bin calibration** (calibration/test sequences disjoint — avoids "threshold tuned on test data" criticism).

## 2. Stratified Quotas (14 classes × contact density, total 2350)

Contact density is **four levels C0–C3** (from 2026-07-18): former "large-area/load-bearing" (old C3) and "entangle/grapple" (old C4) merged into **C3 high contact** — both clearly above C0–C2, but contact area hard to rank strictly. High-contact bucket total quota 1380 (59%), intentionally over-weighted — that is this benchmark's differentiator.

| Density | Category | Quota | Suggested sources |
|---|---|---|---|
| C1 light/transient contact | face_to_face_talk | 100 | CC0 (no-contact control; includes CHI3D Posing) |
| C1 | handshake | 120 | CC0 + CHI3D Handshake |
| C1 | high_five | 100 | CC0 |
| C2 sustained point/line contact | hand_hold | 140 | CC0 + CHI3D HoldingHands |
| C2 | arm_around_shoulder | 150 | CC0 |
| C2 | dance | 160 | CC0 + H4D ballroom |
| C3 high contact (close body/load ∪ grapple/combat) | hug | 240 | CC0 + CHI3D Hug |
| C3 | piggyback | 180 | CC0 |
| C3 | carry_lift | 200 | CC0 |
| C3 | dance_lift | 180 | CC0 |
| C3 | wrestle_grapple | 260 | CC0 + H4D grappling + CHI3D Grab |
| C3 | fight_combat | 260 | CC0 + H4D karate/mma + CHI3D Hit/Kick/Push |
| C3 | restrain_pin | 60 | CC0 stock only 67; gap filled from wrestle |
| 3+ people | other_multi_person | 200 | CC0 3+ scenes + Panoptic; tests Count dimension |

**Cross-source category mapping** (auto-counted on `/testset` page):
- H4D scenes: ballroom* → dance; grappling* → wrestle_grapple; karate*/mma* → fight_combat.
- CHI3D actions: Grab→wrestle_grapple, Handshake→handshake, Hit/Kick/Push→fight_combat, HoldingHands→hand_hold, Hug→hug, Posing→face_to_face_talk.
- Panoptic → other_multi_person.

## 3. Selection Criteria (check per item on dashboard)

1. Each actor has at least 1 ★ clean frontal reference (prefer fewer scenes with relaxed refs; if selected, keep `ref_quality: relaxed` tag).
2. Target action structure clear, both people visible, contact region obvious; prefer high resolution/clarity (must pass ArcFace/mesh formal metrics).
3. Avoid same model repeated within a class (Pexels models recur across videos).
4. **≤20 target images counted per scene** — for burst shots at one moment, keep best only (delete extras with ✕ before assignment); forces scene diversity (~120+ scenes for full set).
5. 3+ person scenes 15~20%.
6. CHI3D: **assign whole actor pairs to test** (s02/s03/s04 three pairs, all different people): splitting one pair half train / half test causes same-person leakage.

## 4. Mechanism (Frontend → Data Flow)

- Dashboard per-scene card "**Move to test set**" button (purple): anchor written to that class's `board_state.json` `tset`, team-shared, click again to move back; if any member in a scene merge chain is marked → entire final scene goes to test.
- **H4D unified to same scene dashboard** (07-14, `/cc0scene/harmony4d`, replaces old `/h4dreview`): `h4d_to_board.py` converts `final/harmony4d` to dashboard format (filenames rewritten to `_f`/`_r` convention, hard-link zero-copy), `board_name_map.json` records dashboard name → final original name; dashboard ✕ delete is bookkeeping on final — applied back to `final/` via mapping table. Test assignment same mechanism as other sources (`tset`); scene names (ballroom/grappling/karate/mma*) are anchors; quota class from §2 mapping auto-counted.
- `/testset` overview: quota progress bars + per-class assigned scene cards (ref→target thumbnails); portal/category pages separate train vs test counts (train prep count = total − test).
- Dashboard "Export scene map" JSON includes per-scene `split: "test" | "train"`; `finalize_from_scene_map.py` writes manifest `split` field accordingly, same timing as reference embedding freeze (eval_id_protocol.md §5.5).

## 5. Future Work (Not This Phase — Placeholders)

- Val/Dev set (~500 for checkpoint / RL reward tuning) carved from training side after training starts; video-level disjoint from test.
- OOD track: hold out one full source (likely Panoptic or EgoHumans) from training; separate cross-domain generalization score.
- EgoHumans can supplement test when available (same license path).

## 6. 07-15 Assignment Notes

- Total **2500** hits upper bound of "2~3k" target; next priority is **rebalancing across classes** (fill `restrain_pin`, supplement 3+ from Panoptic if needed), not growing total count.
- Stage 7 / eval / paper distribution stats use **2500 target images** as N; scripts must use same smerge merge logic as `_test_split` (see `generate_prompts_full.get_test_scenes`).
