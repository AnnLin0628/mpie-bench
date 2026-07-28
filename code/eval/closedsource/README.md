# Closed-Source API Evaluation

Uses the same pack layout as open-source: results are written to `$MPIE_TEST_PACK/outputs/<model_id>/` so scoring can run side by side.

| model_id (on disk) | API | Protocol |
|---|---|---|
| `gpt-image-2` | OpenAI-compatible gateway | `POST /v1/images/edits` (multipart `image[]`) |
| `gemini-3-pro-image` | Gemini / OpenAI-compatible gateway | `v1beta generateContent` or `POST /v1/chat/completions` |
| `seedream-5-pro` | Vendor image API | See environment variables below |

## Environment

The public repository **does not ship** gateway URLs or API keys. Export them yourself:

```bash
export AI_GATEWAY_KEY=sk-...                 # or GPT_IMAGE_KEY / AI_GATEWAY_KEY / ARK_API_KEY
export ARK_API_KEY=...               # Volcano ARK (optional)

# gpt-image-2
export GPT_IMAGE_GATEWAY_URL=https://<your-gateway>/v1
# or unified:
# export AI_GATEWAY_URL=https://<your-gateway>/v1

# gemini-3-pro-image
export GEMINI_V1BETA_URL=https://<your-gateway>/v1beta
export GEMINI_PROTOCOL=v1beta        # or chat
export GEMINI_IMAGE_MODEL=gemini-3-pro-image

# seedream-5-pro (example: BytePlus SEA)
export SEEDREAM_URL=https://ark.ap-southeast.bytepluses.com/api/v3/images/generations
export SEEDREAM_MODEL=doubao-seedream-5-0-pro
export SEEDREAM_KEY=$SEEDREAM_KEY   # or fallback via ARK_API_KEY / AI_GATEWAY_KEY
```

> MPIE requires reference images: gpt uses `/images/edits`; gemini uses `v1beta/...:generateContent` (with `inline_data`); seedream passes `image` data URIs in the payload.

Dependency: `requests`.

## Commands

```bash
export MPIE_TEST_PACK=/path/to/mpie-bench/data/testset

# Batch entry
bash code/eval/run_full_closed.sh
bash code/eval/run_full_closed.sh --limit 2

# Single model
cd code/eval/closedsource
python run_closed.py --model gpt-image-2 --pack "$MPIE_TEST_PACK"
python run_closed.py --model gemini-3-pro-image --pack "$MPIE_TEST_PACK"
```

Once open-source and closed-source output directories are aligned, the same aggregation/judge scripts can score both.
