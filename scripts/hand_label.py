# scripts/hand_label.py
"""Interactive CLI for hand-labeling real form event traces for the test set."""

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

from formrecap_lora.data.preprocessor import normalize_events
from formrecap_lora.data.primers import CLASS_NAMES

console = Console()


@click.command()
@click.option("--output", default="data/real/test.jsonl")
def main(output: str):
    console.print("[bold]Hand-label real test examples.[/bold] Enter event traces one per session.")
    console.print("Paste JSON event array (one paste, end with Ctrl-D), or type 'quit' to finish.\n")
    for code, name in CLASS_NAMES.items():
        console.print(f"  {code} = {name}")
    console.print()

    records: list[dict] = []
    if Path(output).exists():
        records = [json.loads(l) for l in Path(output).read_text().splitlines() if l.strip()]
        console.print(f"[dim]Loaded {len(records)} existing records from {output}[/dim]\n")

    while True:
        console.print(f"[cyan]--- Example #{len(records) + 1} ---[/cyan]")
        console.print("Paste event JSON array (or 'quit'):")
        buf: list[str] = []
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                if line.strip() == "quit":
                    raise KeyboardInterrupt
                buf.append(line)
        except KeyboardInterrupt:
            break

        raw = "".join(buf).strip()
        if not raw or raw == "quit":
            break
        try:
            events = json.loads(raw)
            normalized = normalize_events(events)
        except Exception as e:
            console.print(f"[red]Parse failed: {e}[/red]")
            continue
        console.print(f"Normalized: [yellow]{normalized}[/yellow]")

        code = IntPrompt.ask("Class code (1-6)", choices=["1", "2", "3", "4", "5", "6"])
        reason = Prompt.ask("Reason (1-2 sentences)")
        confidence = float(Prompt.ask("Confidence 0.0-1.0", default="0.75"))
        records.append({
            "events": normalized,
            "code": int(code),
            "reason": reason,
            "confidence": confidence,
        })
        console.print(f"[green]Recorded. Total: {len(records)}[/green]\n")

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    console.print(f"[bold green]Saved {len(records)} records to {out}[/bold green]")


if __name__ == "__main__":
    main()
