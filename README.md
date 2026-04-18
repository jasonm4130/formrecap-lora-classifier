# formrecap-lora-classifier

Fine-tuned Llama 3.2 3B classifier for form abandonment detection.

## Why

FormRecap tracks form interaction events: focus, blur, input, scroll, exit. When a user abandons a form, those events tell a story. The question is which story.

I built this because off-the-shelf LLMs handle this classification task inconsistently, and verbalized confidence scores from large models are poorly calibrated on structured behavioural data. A small fine-tuned model running at the edge is faster, cheaper, and more consistent.

The model classifies abandonment events into six reasons:

| Class | Description |
|---|---|
| `validation_error` | User hit a field they couldn't complete |
| `distraction` | Rapid unfocus, no scroll, short session |
| `comparison_shopping` | Multiple tab switches, long dwell before exit |
| `accidental_exit` | Immediate return signal or quick re-engagement |
| `bot` | Non-human interaction pattern |
| `committed_leave` | Deliberate exit after meaningful engagement |

Different abandonment reasons require different recovery strategies. A single "abandoned" label loses that signal entirely.

## How It Works

### Training pipeline

1. Generate 1100 synthetic training examples with Claude Sonnet (per-call form context randomisation, temperature=1.0, in-run dedup rejection to avoid the duplicate collapse problem)
2. Fine-tune with QLoRA on Modal (NF4 quantisation, LoRA r=16, DoRA, Unsloth)
3. Evaluate against 52 hand-crafted real test examples

One training run: ~30 minutes, ~$0.40 on a single L4 GPU.

### Why LoRA over full fine-tune

Llama 3.2 3B has the strongest fine-tuning delta of the sub-4B models. LoRA gets most of that delta at a fraction of the compute. Full fine-tune would cost 20x more and produce a model that can't be served via Cloudflare's BYO-LoRA path.

### Why dual deployment

Cloudflare Workers AI supports BYO-LoRA adapters but does not expose logprobs. Modal does.

Logprobs matter here. Verbalized confidence from an LLM ("confidence: 0.85") is poorly calibrated. The correct approach is to extract the probability of the class digit token directly from the model's output distribution, then apply temperature scaling to minimise Expected Calibration Error (ECE).

So the deployment split is:

- **Cloudflare Workers AI** → edge inference, sub-200ms p50 from Australia, verbalized confidence only
- **Modal** → calibrated inference via logprobs + temperature scaling, used for high-stakes recovery paths

One trained adapter. Two deployment targets. The CF constraint is what made the dual-deploy architecture necessary, not a design preference.

### Calibration approach (Modal path)

Each of the six class labels maps to a single digit token (1-6) in Llama's vocabulary. The logprob of that token is a cleaner confidence signal than anything the model says in text. Temperature scaling fits one scalar `T` on the validation set to minimise ECE. The calibrated confidence is what gets sent to the FormRecap recovery pipeline.

## Results

Full baseline comparison (majority class, zero-shot, 5-shot, Claude Haiku, this model) is documented in [`docs/blog-draft.md`](docs/blog-draft.md). Metrics include macro-F1, per-class F1, and ECE before and after calibration. Final numbers pending training run completion.

## Tech Stack

| Component | Technology |
|---|---|
| Model | Llama 3.2 3B |
| Fine-tuning | Hugging Face PEFT/TRL, Unsloth, QLoRA NF4 |
| Training infra | Modal (L4 GPU) |
| Edge serving | Cloudflare Workers AI (BYO-LoRA) |
| Calibrated serving | Modal (logprobs + temperature scaling) |
| Data generation | Claude Sonnet |
| Package manager | uv |

## Quickstart

```bash
# Install dependencies
uv sync --extra dev

# Configure environment
cp .env.example .env  # fill in Modal token, CF API key, Anthropic key

# Generate synthetic training data
python -m formrecap_lora.data.generate --count 1100

# Train (runs on Modal, ~30 min, ~$0.40)
modal run training/modal_app.py::train --run-id baseline

# Evaluate against real test set
python -m formrecap_lora.eval.runner --run-id baseline
```

With 1Password CLI:

```bash
op run --env-file .env.op -- modal run training/modal_app.py::train --run-id baseline
```

## Project Structure

```
src/formrecap_lora/
  data/       # synthetic data generation
  training/   # Modal training app, LoRA config
  eval/       # evaluation runner, calibration
  serving/    # Modal + CF inference endpoints
training/
  modal_app.py
docs/
  blog-draft.md   # full write-up with architecture detail and results
```

## Protection Stack

The serving endpoints include Turnstile verification, rate limiting, HMAC request tokens, a daily inference budget cap, and a KV kill switch. Weekend project, not unprotected endpoint.

## Status

Active development. Training complete. Calibration and CF deployment in progress. Live demo at [lab.formrecap.com](https://lab.formrecap.com) when serving is stable.

## License

Apache-2.0
