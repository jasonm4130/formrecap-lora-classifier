# FormRecap LoRA Classifier

## Project Overview

LoRA fine-tuned classifiers for form abandonment reasons (6 classes). Trained on Modal, deployed dually to Modal (logprobs/calibration) and Cloudflare Workers AI (edge).

## Tech Stack

- **Python 3.11** via `uv` (NOT pip, NOT conda)
- **Modal** for GPU training (L4) and inference
- **PEFT + TRL** for LoRA fine-tuning
- **Anthropic SDK** for data generation + LLM-as-judge
- **pytest** for testing, **ruff** for linting/formatting
- **TypeScript + Cloudflare Workers** for edge serving
- **Terraform** (cloudflare provider v5.x) for CF infrastructure
- **1Password CLI (`op`)** for all secrets — NEVER use plaintext .env

## Conventions

### Secrets

Always use `.env.op` with `op://` references. Run commands with:
```bash
op run --env-file .env.op -- <command>
```
Never store secrets in `.env` files or paste them in conversation.

### Python

- Source layout: `src/formrecap_lora/` with subpackages `data/`, `training/`, `eval/`, `serving/`
- Tests in `tests/` — run with `uv run pytest tests/ -v`
- Lint/format: `uv run ruff check --fix . && uv run ruff format .`
- Type hints on public APIs, no docstrings on obvious functions
- Use `click` for CLIs, `rich` for console output

### Training

- Config lives in `training/config.py` — single source of truth
- Modal app in `training/modal_app.py`
- Data on Modal Volume `formrecap-lora`, uploaded via `scripts/upload_data.py`

### Testing

- TDD for pure functions (preprocessor, dedupe, metrics, calibration, splits)
- No mocking of external APIs in unit tests — those are integration tests
- Run full suite: `uv run pytest tests/ -v`

### Git

- Commit after each logical unit of work
- Don't commit data files (`.jsonl`, `.safetensors`) — they live in Modal Volume or are gitignored
- Don't commit `.env` files — only `.env.op` and `.env.example`

## Architecture Decisions

- **Dual deployment:** Modal (logprobs + calibration) + CF Workers AI (edge, verbalized confidence only). CF does NOT expose logprobs — confirmed in day-1 verification.
- **Exact-only dedupe:** No OpenAI dependency. Semantic dedupe dropped — exact hash on event string is sufficient for synthetic data.
- **6-class taxonomy:** validation_error, distraction, comparison_shopping, accidental_exit, bot, committed_leave
- **Training target format:** Leading digit (1-6) on first line, JSON on second line. Digit is a single token in Llama vocabulary for clean logprob extraction.
