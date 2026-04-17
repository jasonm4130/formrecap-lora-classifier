"""Build the final JSONL datasets in Llama 3.2 chat template format."""

import json
from datetime import date
from pathlib import Path

import click
from rich.console import Console

from .dedupe import dedupe_exact
from .primers import CLASS_NAMES
from .splits import split_stratified

console = Console()

SYSTEM_CONTENT = (
    "You analyse form interaction event sequences and classify the likely abandonment reason. "
    "Respond with a digit class code 1-6 on the first line, then a JSON object on the second line "
    "with class, reason, and confidence fields. "
    "Classes: 1=validation_error, 2=distraction, 3=comparison_shopping, "
    "4=accidental_exit, 5=bot, 6=committed_leave."
)


def to_chat_record(rec: dict) -> dict:
    class_name = CLASS_NAMES[rec["code"]]
    assistant = f'{rec["code"]}\n{{"class": "{class_name}", "reason": {json.dumps(rec["reason"])}, "confidence": {rec["confidence"]}}}'
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": f"Events: {rec['events']}"},
            {"role": "assistant", "content": assistant},
        ]
    }


@click.command()
@click.option("--raw", default="data/synthetic/raw.jsonl")
@click.option("--output-dir", default="data/synthetic")
@click.option("--seed", default=42)
def main(raw: str, output_dir: str, seed: int):
    raw_path = Path(raw)
    records = [json.loads(l) for l in raw_path.read_text().splitlines() if l.strip()]
    console.print(f"Loaded {len(records)} raw records")

    before = len(records)
    records = dedupe_exact(records)
    console.print(f"After exact dedupe: {len(records)} (removed {before - len(records)})")

    train, val, test = split_stratified(records, val_frac=0.1, test_frac=0.1, seed=seed)
    console.print(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    ts = date.today().isoformat()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in [(f"train-{ts}", train), (f"val-{ts}", val), (f"test-{ts}", test)]:
        path = out / f"{name}.jsonl"
        with path.open("w") as f:
            for rec in data:
                f.write(json.dumps(to_chat_record(rec)) + "\n")
        console.print(f"  Wrote {path}")


if __name__ == "__main__":
    main()
