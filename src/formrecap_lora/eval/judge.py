"""LLM-as-judge for reason-quality evaluation."""

import json
import random
from pathlib import Path

import click
from anthropic import Anthropic
from rich.console import Console
from rich.table import Table

console = Console()

RUBRIC = """You are evaluating the quality of free-text reasoning in a form-abandonment classifier's output.

For each example, score 0-5 on each of four criteria:
1. Specificity — does the reason cite concrete events from the input?
2. Groundedness — is the reasoning supported by the event trace (no hallucinated events)?
3. Class fit — does the reason justify the predicted class?
4. Concision — is the reason appropriately brief without losing substance?

Respond with ONLY a JSON object: {"specificity": int, "groundedness": int, "class_fit": int, "concision": int}
"""


def score_one(client: Anthropic, events: str, predicted_class: str, reason: str) -> dict:
    content = (
        f"Event trace: {events}\n"
        f"Predicted class: {predicted_class}\n"
        f"Reason: {reason}\n"
        "Score now."
    )
    r = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        system=RUBRIC,
        messages=[{"role": "user", "content": content}],
    )
    return json.loads(r.content[0].text.strip())


@click.command()
@click.option("--predictions", required=True, help="JSONL of {events, predicted_class, reason}")
@click.option("--sample-size", default=50)
@click.option("--seed", default=42)
@click.option("--output", default="docs/results/judge_scores.json")
def main(predictions: str, sample_size: int, seed: int, output: str):
    client = Anthropic()
    all_preds = [json.loads(l) for l in Path(predictions).read_text().splitlines() if l.strip()]
    random.seed(seed)
    sampled = random.sample(all_preds, min(sample_size, len(all_preds)))

    scores: list[dict] = []
    for pred in sampled:
        try:
            s = score_one(client, pred["events"], pred["predicted_class"], pred["reason"])
            scores.append(s)
        except Exception as e:
            console.print(f"[yellow]skip: {e}[/yellow]")

    # Aggregate
    criteria = ["specificity", "groundedness", "class_fit", "concision"]
    agg = {c: sum(s[c] for s in scores) / len(scores) for c in criteria}
    total_agg = sum(agg.values()) / 4

    table = Table(title="Reason Quality (LLM-as-judge)")
    table.add_column("Criterion")
    table.add_column("Mean score (0-5)", justify="right")
    for c, v in agg.items():
        table.add_row(c, f"{v:.2f}")
    table.add_row("[bold]Overall[/bold]", f"[bold]{total_agg:.2f}[/bold]")
    console.print(table)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(
        json.dumps({"sampled": len(scores), "aggregate": agg, "overall": total_agg}, indent=2)
    )


if __name__ == "__main__":
    main()
