# Standard Dataset Processing Pipeline & Standard Output (Finalized 2026-07-10)

> Distilled from Harmony4D + CC0 practice rounds. **All data sources (CHI3D / future additions) follow this process.**

## Standard Output (Final Form)

```
final/<dataset>/
  refs/           # exactly 1 human-curated clean reference image per actor per scene
  targets/        # all qualified target images for that scene
  manifest.json   # scenes: [{scene, actors:{id: ref_fn}, targets:[...], bindings}]
```

- Three organizing principles: **scene category → one clean reference per actor → target↔actor binding map** (a target may include only a subset of actors in the scene; AB/AC/BC/ABC subsets are expressed via bindings).
- **Unified final form (2026-07-10)**: training/eval uses a single **(reference image, text, target image)** triple pool — **no per-interaction-category directories**. Category is optional metadata on targets; DB results are category-agnostic; the core is aligning each target with its reference and text.
- **Cross-category scene merge = physical move** (07-10 revision; prior "bookkeeping then assemble" was rejected — user required merged results to appear live and remain editable): dashboard "Merge scene" input `hug 12` → server `/cc0xmove` (`xmove.py`) moves actor groups + image files + human review state into the target category, writes structured merge into target `scene_merges.json` and target scene card; both dashboards regenerate immediately; all subsequent edits happen natively in the target category and export with its scene map. Moved members leave `moved_to` placeholders in source `ref_clusters` to keep aid stable. Note: **if the target category already exported a scene map, re-export after merge**; legacy 3-field bookkeeping records auto-migrate on dashboard open.
- **Dual-machine storage**: dev machine `$MPIE_ROOT/data/final/<dataset>/` (VLM rewrite + transfer to eval cluster) + eval machine `~/mpie_data/final/<dataset>/` (model testing). Dev machine tar+md5 transfer to eval machine.

## Standard Process (Six Steps)

| Step | Task | Location | Tools |
|---|---|---|---|
| 1 Ingest | Raw assets on disk (video / multi-cam / stills) | eval machine `raw_video/` or `datasets/` | `01_ingest/` |
| 2 Extract | Target = interaction-peak frame; ref candidates via **frontal-face recipe** (below); persist to DB | eval machine `manifests/mpie_<src>*.db` | `02~06` stages |
| 3 Cluster | GT identity (H4D/CHI3D) used directly; no GT → GPU ArcFace clustering → `ref_clusters.json` | eval machine | `cluster_refs_category.py` |
| 4 Review | Package to dev machine → scene dashboard human review: drop bad images, 1 ref per actor, target bindings, scene merge (optional VLM scene-fingerprint pre-merge) | dev machine port 8080 `/cc0scene/<cat>` | `package_cc0_review.sh` → `ingest_review_pkg.sh` |
| 5 Apply | Export scene map JSON → archive verbatim to `manifests/scene_maps/` → apply to DB | eval machine | `common/apply_scene_map.py` |
| 6 Finalize | Produce `final/<dataset>/` in standard form; one copy on each machine | dev + eval | `10_manifest_build/build_samples.py` |

## Frontal Reference Candidate Recipe (Finalized 2026-07-10 from CHI3D s03 three-round iteration, user sign-off; all datasets use this extraction)

1. **Candidate frames**: frames where the two people are farthest apart (GT joints or detection boxes for spacing); candidates from all camera views;
2. **Face detection gate** (insightface detection only, not clustering): det_score ≥ 0.62;
3. **Frontal pose**: |yaw| ≤ 38° (from insightface pose directly — do not infer head orientation from skeleton joints; CHI3D lesson: joints_25 is not BODY_25 order, inference gave all back views);
4. **Anti face-swap**: detected face must lie near that person's projected joint / detection-box head location. Sparse two-person scenes: crop-box ratio (≤0.30×height, ≤0.35×width); **crowded scenes (3+ people in frame, e.g. Panoptic) must use face scale: face-center distance to head projection ≤ 2× face height** — box-ratio rules let neighbor faces through (07-10 smoke test leak);
5. **Full head top**: crop top margin ≥ 0.32× body height (joint highest point is often neck; 0.15 crops off the head);
6. **Ranking**: sort by face pixel height descending, take TOP12 (dashboard actor-group cap); at most 2 per action/clip to spread candidates.

Reference implementation: `code/pipeline/01_ingest/chi3d_refs_frontal.py`.

## Two-Layer Storage for Human Review Ground Truth

1. **Verbatim archive**: dashboard-exported scene map / delete list JSON permanently under `manifests/` (human_review/, scene_maps/); replayable anytime;
2. **DB apply**: identity merge, ref deletion, is_primary star, human_bindings table — in `build_samples.py`, human data overrides auto-derived fields.

## Data Source Roles in This Pipeline

| Source | What it adds | Drop rate | License |
|---|---|---|---|
| Harmony4D (✅ final) | C3 high-contact hard interaction + full GT | Low (GT-driven) | MIT, publishable |
| CC0 Pexels (in review) | Actor/scene diversity + publishable pixels | **High** (noisy web footage) | ≈CC0, publishable |
| CHI3D (pending) | Contact-region GT + C1–C3 full action spectrum | Expected low (lab capture) | No redistribution; training/calibration only |
| FlickrCI3D Sig (❌ deprecated for pairing) | Metric generalization only (different people per image — cannot build ref↔target pairs; 07-10 manual verification) | — | No redistribution |

CC0 high drop rate → volume of hard interaction samples from CHI3D (low drop); real appearance diversity gap needs new sources (CC0 candidate pool expansion is the ready lever: 4313 candidates, only 559 downloaded). Per-dataset plans in dataset archives.
