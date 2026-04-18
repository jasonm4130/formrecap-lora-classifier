"""Download a trained LoRA adapter from HuggingFace Hub to local disk."""

import click
from huggingface_hub import snapshot_download
from rich.console import Console

console = Console()

HF_REPOS = {
    "llama-3b": "jasonm4130/formrecap-llama-3.2-3b-lora",
    "gemma-2b": "jasonm4130/formrecap-gemma-2b-lora",
    "gemma-2b-cf": "jasonm4130/formrecap-gemma-2b-cf-lora",
    "mistral-7b-cf": "jasonm4130/formrecap-mistral-7b-cf-lora",
}


@click.command()
@click.option(
    "--model",
    type=click.Choice(list(HF_REPOS.keys())),
    required=True,
    help="Which adapter to download",
)
@click.option("--local-dir", default=None, help="Local directory (default: adapter-{model})")
def main(model: str, local_dir: str | None):
    repo_id = HF_REPOS[model]
    out = local_dir or f"adapter-{model}"
    console.print(f"Downloading [cyan]{repo_id}[/cyan] -> {out}/")
    snapshot_download(repo_id=repo_id, local_dir=out)
    console.print(f"[green]Done. Adapter saved to {out}/[/green]")


if __name__ == "__main__":
    main()
