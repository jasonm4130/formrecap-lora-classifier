"""Push a local LoRA adapter directory to HuggingFace Hub with a model card."""

import json
from pathlib import Path

import click
from huggingface_hub import HfApi
from rich.console import Console

console = Console()


def _build_model_card(
    repo_id: str,
    base_model: str,
    adapter_dir: Path,
    metrics: dict | None = None,
) -> str:
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text())
    r = adapter_config.get("r", "?")
    alpha = adapter_config.get("lora_alpha", "?")
    use_dora = adapter_config.get("use_dora", False)

    metrics_section = ""
    if metrics:
        f1 = metrics.get("macro_f1", "N/A")
        ece = metrics.get("ece", "N/A")
        metrics_section = f"""
## Evaluation (52 hand-labeled real test examples)

| Metric | Value |
|---|---|
| Macro-F1 | {f1:.3f} |
| ECE (logprob) | {ece:.3f} |
"""

    return f"""---
library_name: peft
base_model: {base_model}
tags:
  - lora
  - form-abandonment
  - classification
  - formrecap
license: apache-2.0
---

# {repo_id.split("/")[-1]}

LoRA adapter fine-tuned on synthetic form abandonment event traces for 6-class classification.

## Base Model

[{base_model}](https://huggingface.co/{base_model})

## Task

Classifies form interaction event traces into one of six abandonment reasons:

| Code | Class | Description |
|---|---|---|
| 1 | validation_error | User hit a field error they couldn't resolve |
| 2 | distraction | User task-switched away |
| 3 | comparison_shopping | Browsing, not committing |
| 4 | accidental_exit | Closed tab / back button by mistake |
| 5 | bot | Automated non-human interaction |
| 6 | committed_leave | Intentionally chose not to complete |

## Training

- **Method:** QLoRA (NF4 4-bit) + LoRA (r={r}, alpha={alpha}, DoRA={use_dora})
- **Data:** 884 synthetic examples generated with Claude Sonnet, stratified 80/10/10 split
- **Hardware:** Modal L4 GPU, ~30 minutes, ~$0.40
- **Framework:** HuggingFace PEFT + TRL
{metrics_section}
## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("{base_model}")
model = PeftModel.from_pretrained(base, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

## Source

[jasonm4130/formrecap-lora-classifier](https://github.com/jasonm4130/formrecap-lora-classifier)
"""


@click.command()
@click.option("--adapter-dir", required=True, type=click.Path(exists=True))
@click.option("--repo-id", required=True, help="e.g. jasonm4130/formrecap-gemma-2b-lora")
@click.option("--base-model", required=True, help="e.g. google/gemma-2b-it")
@click.option(
    "--results-file", default=None, type=click.Path(exists=True), help="JSON results file"
)
@click.option("--results-key", default=None, help="Name key to match in results JSON")
def main(
    adapter_dir: str,
    repo_id: str,
    base_model: str,
    results_file: str | None,
    results_key: str | None,
):
    adapter_path = Path(adapter_dir)
    api = HfApi()

    # Load metrics if provided
    metrics = None
    if results_file and results_key:
        data = json.loads(Path(results_file).read_text())
        for r in data.get("results", []):
            if r["name"] == results_key:
                metrics = r
                break

    # Create repo
    console.print(f"Creating repo [cyan]{repo_id}[/cyan]...")
    api.create_repo(repo_id, exist_ok=True, repo_type="model")

    # Write model card
    card_text = _build_model_card(repo_id, base_model, adapter_path, metrics)
    readme_path = adapter_path / "README.md"
    readme_path.write_text(card_text)

    # Upload all files
    console.print(f"Uploading {adapter_path} -> {repo_id}...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(adapter_path),
        commit_message=f"Upload LoRA adapter from {adapter_path.name}",
    )

    console.print(f"[green]Done: https://huggingface.co/{repo_id}[/green]")


if __name__ == "__main__":
    main()
