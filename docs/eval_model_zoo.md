# Evaluation Model List (Model Zoo)

Task form: **multiple reference images + edit instruction → target image**. Pure T2I models without reference-image input cannot run this task directly.

Paper / main-table lineup: **3 closed-source + 7 open-source = 10**.

Order and names match the MPIE-Bench paper experiments table.

## Closed-source (main table 3)

| # | Model | Vendor | On-disk `model_id` |
|---|---|---|---|
| 1 | GPT-Image-2 | OpenAI | `gpt-image-2` |
| 2 | Gemini-3-Pro-Image | Google | `gemini-3-pro-image` |
| 3 | Seedream-5-Pro | ByteDance | `seedream-5-pro` |

Integration: [`code/eval/closedsource/README.md`](../code/eval/closedsource/README.md).  
Optional public artifacts (these three only): [BASELINES.md](BASELINES.md).

## Open-source (main table 7)

| # | Paper name | `model_id` | Suggested conda | Main weights | runner |
|---|---|---|---|---|---|
| 1 | FLUX.1-Kontext | `flux1-kontext-dev` | `mpie_edit` | FLUX.1 Kontext | `run_kontext.py` |
| 2 | DreamO | `dreamo` | `dreamo` | DreamO LoRA + BEN2 + Turbo + FLUX.1-dev | `run_dreamo.py` |
| 3 | OmniGen2 | `omnigen2` | `omnigen2` | OmniGen2 | `run_omnigen2.py` |
| 4 | UNO | `uno` | `uno` | FLUX.1-dev + UNO LoRA + CLIP/T5 | `run_uno.py` |
| 5 | ACE++ | `ace` | `ace` | FLUX.1-Fill + ACE-Plus subject LoRA | `run_ace.py` |
| 6 | BAGEL | `bagel` | `bagel` | BAGEL-7B-MoT | `run_bagel.py` |
| 7 | FireRed-Image-Edit | `firered` | `firered` | FireRed-Image-Edit-1.1 | `run_firered.py` |

Entry script: `code/eval/run_opensource_full_8gpu.sh` (`all` = these 7).  
Generation notes: [eval_opensource.md](eval_opensource.md).
