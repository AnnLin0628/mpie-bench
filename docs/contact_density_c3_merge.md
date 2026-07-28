# Contact-Density Tiers: C3∪C4 → C3 (2026-07-18)

## Decision

Contact density changes from five tiers **C0–C4** to four tiers **C0–C3**.

| New tier | Meaning | Former tier |
|---|---|---|
| C0 | No contact / near distance without touch | C0 |
| C1 | Light / transient (handshake, high-five, etc.) | C1 |
| C2 | Sustained point/line or light torso contact (hand-holding, arm-around-shoulder, dance holds, etc.) | C2 |
| **C3** | **High contact**: close-body load-bearing ∪ grappling/combat | **Former C3 ∪ former C4** |

## Why merge

- Former C3 (hug / piggyback / carry) and former C4 (fight / wrestle) are hard to rank strictly by "contact area."
- Both are clearly above C0–C2; forcing two tiers created a fake "difficulty ladder" (mesh curves "rebounded" on former C4).
- Fine-grained failure modes are now broken down by **board_cat / interaction_type** (hug vs fight), not by density tier.

## Frozen counts (public test set 2500)

C0=513, C1=449, C2=942, **C3=596** (formerly 336+260).

## Synced locations (summary)

- Data: `data/manifests/prompt_distribution/targets_{tagged,natural}.csv`, `summary.json`
- Eval aggregation: `code/eval/aggregate_mesh_v3.py` → evaluation dashboard `full2500_v3` (density table + multi-model C0→C3 curves)
- Protocol: `testset_split.md`, `eval_protocol_v3.md`, `caption_client.py`, `analyze_prompt_natural.py`, `harmony4d_ingest.density_band`
- Frontend quota: `app.py` `_QUOTA`
- Full progress: the e2e evaluation guide
- Manuscript text: former "five-tier C0–C4" wording updated to four tiers

If historical docs still mention "former C4," read it as **merged into C3**.

## When reading curves

Main results use **C0→C3** and **Δ(C3−C0)**; within C3, hug / fight, etc. should still be split by `board_cat`. Do not treat density tiers as a strict difficulty ladder.
