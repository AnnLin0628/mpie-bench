# Evaluation Model List (Model Zoo)

Task form: **multiple reference images + edit instruction → target image**. Pure T2I models without reference-image input cannot run this task directly.

Current main table: **3 closed-source + 7 open-source = 10** (image generation only, for baseline leaderboard).

## Closed-source (main table 3)

| # | Model | Vendor | On-disk `model_id` |
|---|---|---|---|
| 1 | gpt-image-2 | OpenAI | `gpt-image-2` |
| 2 | Gemini (nano-banana-pro) | Google | `gemini-3-pro-image` |
| 3 | seedream-5-pro | ByteDance | `seedream-5-pro` |

Integration notes: [`code/eval/closedsource/README.md`](../../code/eval/closedsource/README.md).

### Closed-source candidates

| Model | Notes |
|---|---|
| nano-banana-2 / lite | Same vendor, different tier |
| qwen-image-2 | Can be compared with open-source Qwen-Image-Edit |

## Open-source (main table 7)

| model_id | Suggested conda | Main weights | runner |
|---|---|---|---|
| `flux1-kontext-dev` | `mpie_edit` | FLUX.1 Kontext | `run_kontext.py` |
| `qwen-image-edit-2511` | `mpie_edit` | Qwen-Image-Edit | `run_qwen_edit.py` |
| `omnigen2` | `omnigen2` | OmniGen2 | `run_omnigen2.py` |
| `uno` | `uno` | FLUX.1-dev + UNO LoRA + CLIP/T5 | `run_uno.py` |
| `ace` | `ace` | FLUX.1-Fill + ACE-Plus subject LoRA | `run_ace.py` |
| `bagel` | `bagel` | BAGEL-7B-MoT | `run_bagel.py` |
| `dreamo` | `dreamo` | DreamO LoRA + BEN2 + Turbo + FLUX.1-dev | `run_dreamo.py` |

Entry script: `code/eval/run_opensource_full_8gpu.sh`  
Image-generation notes and pitfalls: [eval_opensource.md](eval_opensource.md)

### Open-source candidates

| Model | Notes |
|---|---|
| Step1X-Edit | Depends on schedule |
| FLUX.2-dev | Re-evaluate when more VRAM is available |
