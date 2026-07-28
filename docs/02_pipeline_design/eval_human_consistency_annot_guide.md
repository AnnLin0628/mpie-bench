# Anat / Inter Annotation Guide (v5.1)

> Synced with `judgments/human_consistency/GUIDELINES.md` in the pack and frontend 8080.  
> Authoritative details: [`eval_human_consistency_analysis_protocol.md`](eval_human_consistency_analysis_protocol.md) §0 / §1.

- Principle: `eval_construct_validity_principle.md`  
- UI: http://127.0.0.1:8080/

## Sample Pool Notes (required reading)

This split is **`v4_open5_c23`**: **all 5 open-source models represented** (flux / dreamo / uno / omnigen2 / ace), contact density **C2+C3 only**, hard cases ≥60%; excludes bagel/qwen; closed-source models are not sampled in this round.  
Please **score carefully**, using the full 1–5 and 2/1/0 scales; do not default to perfect scores.

## Contact Intent (required · judge yourself)

The system-displayed intent is **only a keyword guess and is unreliable**. Choose based on the edit instruction:

| Option | Meaning |
|------|------|
| required | The instruction requires contact (hug, handshake, etc.) |
| forbidden | The instruction forbids touching / entanglement |
| unspecified | Neither explicitly required nor forbidden |

**Do not skip the Inter items below because of the system guess.**

## Overall Preference (required · use full 1–5)

| ID | Anchor (summary) |
|----|------|
| **Q_inter** | 1=blur/clump / interaction fully failed … 3=partly valid with obvious issues … 5=touching + no pathological penetration + correct body region |
| **Q_anat** | 1=extra limbs / severe deformity … 3=moderate structural issues … 5=clean + clear ownership + normal proportions |

## Item-Level Coding

| Code | Meaning |
|----|------|
| I1 / A1 / A2 / A3 = 2/1/0 | **Severity**: clean / mild / severe |
| Ic = 2/1/0 | touching / contact but not touching / no contact |
| I0 / I3 / Ir / A4 / A5 = 1/0 | Binary: pass / fail |
| U | Cannot judge |
| null | **Only** when Ic=0, Ir is auto-skipped (not "system says not applicable") |

## Inter (all items required)

You must answer: **I0, I1, Ic, I3**; if Ic≥1 you must also answer **Ir**.

- **Ic** judges touch only; **Ir** judges body region only (do not fail region because of poor touch)  
- **I1**: blur/clump or obvious penetration=0; mild adhesion=1; clean=2; hand blur only → I1=2, hand blur goes to A5=0  
- **I3**: no unwanted entanglement; when the instruction requires a hug, "no extra wrong entanglement" can still score 1  

## Anat

- **A1/A2/A3** use three severity levels; mild issues score 1  
- Blur/clump → I1=0, mark Anat as U  

## Ten-Person Gold Label (main paper)

Official: `H*` = **mean score across 10 annotators** per item, then rounded to a legal code. Early three-person majority vote is pilot-only compatibility.

## (Legacy) Three-Person Gold Label

Independent annotation → item-level majority vote + `intent_human` majority vote → then compute S; Q uses median.  
Do not look at mesh / VLM scores.
