# FormRecap LoRA Classifier

Fine-tuned LoRA classifiers for form abandonment detection. Trained on Modal, served on both Modal (logprob-calibrated confidence) and Cloudflare Workers AI (edge).

**[Live demo](https://labs.formrecap.com)**

## Why

FormRecap tracks form interaction events: focus, blur, input, scroll, exit. When a user abandons a form, those events tell a story. A single "abandoned" label loses the signal. This classifier turns event traces into actionable abandonment reasons so recovery flows can be targeted.

Six classes:

| Code | Class | Description |
|---|---|---|
| 1 | `validation_error` | User hit a field error they couldn't resolve |
| 2 | `distraction` | User task-switched away |
| 3 | `comparison_shopping` | Browsing, not committing |
| 4 | `accidental_exit` | Closed tab / back button by mistake |
| 5 | `bot` | Automated non-human interaction |
| 6 | `committed_leave` | Intentionally chose not to complete |

## Results

Evaluated on 52 hand-labeled real test examples.

| System | Macro-F1 | 95% CI | ECE |
|---|---|---|---|
| Zero-shot Gemma 2B | 0.063 | [0.040, 0.089] | 0.755 |
| Zero-shot Mistral 7B | 0.095 | [0.063, 0.128] | 0.645 |
| **Gemma 2B Full LoRA** | **0.916** | [0.813, 0.981] | 0.056 |
| Mistral 7B CF LoRA | 0.760 | [0.648, 0.852] | 0.071 |

Calibration headline: temperature scaling on logprobs drops ECE from 0.145 (verbalized) to 0.056 (calibrated) on the Gemma 2B adapter.

### Per-class F1 (Gemma 2B Full LoRA)

| Class | F1 |
|---|---|
| validation_error | 0.957 |
| distraction | 1.000 |
| comparison_shopping | 0.900 |
| accidental_exit | 1.000 |
| bot | 0.889 |
| committed_leave | 0.750 |

## Architecture

Train once, deploy twice. The same LoRA adapter runs on both Modal (for calibration-critical paths with full logprobs) and Cloudflare Workers AI (for edge latency with verbalized confidence only).

CF Workers AI BYO-LoRA does not expose logprobs. This constraint drives the dual deployment:

- **Cloudflare Workers AI** — Mistral 7B + LoRA at the edge, sub-200ms p50 from Australia
- **Modal vLLM** — Gemma 2B + LoRA with logprob extraction + temperature scaling

## Trained Adapters (HuggingFace Hub)

| Adapter | Base Model | F1 | Link |
|---|---|---|---|
| Gemma 2B Full LoRA | google/gemma-2b-it | 0.916 | [HF](https://huggingface.co/jasonm4130/formrecap-gemma-2b-lora) |
| Gemma 2B CF LoRA | google/gemma-2b-it | — | [HF](https://huggingface.co/jasonm4130/formrecap-gemma-2b-cf-lora) |
| Mistral 7B CF LoRA | mistralai/Mistral-7B-Instruct-v0.2 | 0.760 | [HF](https://huggingface.co/jasonm4130/formrecap-mistral-7b-cf-lora) |
| Llama 3.2 3B LoRA | meta-llama/Llama-3.2-3B-Instruct | — | [HF](https://huggingface.co/jasonm4130/formrecap-llama-3.2-3b-lora) |

## Tech Stack

| Component | Technology |
|---|---|
| Training | HuggingFace PEFT + TRL, QLoRA NF4, Modal L4 GPU |
| Edge serving | Cloudflare Workers AI (BYO-LoRA) |
| Calibrated serving | Modal vLLM (logprobs + temperature scaling) |
| Data generation | Claude Sonnet |
| Protection | Turnstile, rate limiting, HMAC tokens, daily budget, kill-switch |
| Package manager | uv |

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env  # fill in keys

# Generate synthetic training data
uv run python -m formrecap_lora.data.generate --count 1100

# Dedupe + split
uv run python -m formrecap_lora.data.assemble --skip-semantic-dedupe

# Upload to Modal volume + train
uv run python scripts/upload_data.py
uv run modal run training/modal_app.py::run_train \
    --run-id my-run \
    --train-file data/train.jsonl \
    --val-file data/val.jsonl \
    --base-model google/gemma-2b-it

# Download adapter from HF Hub (or use your own)
uv run python scripts/download_adapter.py --model gemma-2b

# Evaluate
uv run python -m formrecap_lora.eval.runner --run-id my-run
```

## Project Structure

```
src/formrecap_lora/
  data/         # preprocessor, generator, dedupe, splits
  eval/         # metrics, calibration, baselines, runner, judge
  serving/      # inference endpoints
training/
  config.py     # hyperparameters, shared constants
  modal_app.py  # training + HF predictor (Modal)
  vllm_serve.py # vLLM inference server (Modal)
serving/cf/
  worker/       # Cloudflare Worker (TypeScript)
  pages/        # Demo site (static HTML)
scripts/        # data upload, adapter download, HF push
```

## License

Apache-2.0
