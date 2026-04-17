# FormRecap LoRA Classifier

Fine-tuned Llama 3.2 3B LoRA for form abandonment classification. Trained on Modal, served dually on Modal (calibrated) and Cloudflare Workers AI (edge).

See [`docs/`](docs/) for the blog post draft and results.

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env  # fill in keys
python -m formrecap_lora.data.generate --count 800
modal run training/modal_app.py::train --run-id baseline
python -m formrecap_lora.eval.runner --run-id baseline
```

## Architecture

See [design spec](./docs/DESIGN.md).

## License

Apache-2.0
