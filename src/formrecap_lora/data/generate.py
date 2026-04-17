"""Generate synthetic training examples via Claude Sonnet 4.5."""

import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

import click
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

from .primers import CLASS_NAMES, PRIMERS, ClassCode, Primer, get_primers_for_class

load_dotenv()
console = Console()

SYSTEM_PROMPT = """You generate synthetic form interaction event traces for training a classifier.

Format:
- Events are written as terse tokens separated by commas.
- Event types: focus, input, blur, scroll, exit, submit.
- `input:field(xN)` means N consecutive keystrokes in `field`.
- `blur:field(reason)` includes validation reason if any.
- `scroll:page(Nms)` includes idle duration in ms.

You output ONE new training example as JSON with keys: events, code, reason, confidence.

Constraints:
- Vary field names (email, name, phone, password, address, company, plan, role, message, etc.)
- Vary event sequence length (3-20 events).
- Vary realism: include edge cases.
- Match the example's class semantics exactly.
- `confidence` should be in [0.4, 0.95] and reflect how ambiguous the trace is.
- The `reason` must cite specific events in the trace (not generic).
- Do not copy the primer verbatim. Generate a NEW example.

Output pure JSON only, no markdown, no commentary."""


def _few_shot_block(code: ClassCode, n_primers: int = 2) -> list[dict]:
    primers = get_primers_for_class(code)
    chosen = random.sample(primers, min(n_primers, len(primers)))
    blocks: list[dict] = []
    for p in chosen:
        blocks.append({"role": "user", "content": f"Generate a new example for class: {CLASS_NAMES[code]} (code {code})"})
        blocks.append({
            "role": "assistant",
            "content": json.dumps({
                "events": p["events"],
                "code": p["code"],
                "reason": p["reason"],
                "confidence": p["confidence"],
            }),
        })
    return blocks


def generate_one(client: Anthropic, code: ClassCode) -> dict | None:
    messages = _few_shot_block(code, n_primers=2)
    messages.append({
        "role": "user",
        "content": f"Generate a new example for class: {CLASS_NAMES[code]} (code {code})",
    })
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        raw = response.content[0].text.strip()
        obj = json.loads(raw)
        # Validate schema
        assert obj["code"] == code
        assert isinstance(obj["events"], str) and obj["events"]
        assert isinstance(obj["reason"], str) and obj["reason"]
        assert 0.0 <= float(obj["confidence"]) <= 1.0
        return obj
    except Exception as e:
        console.print(f"[yellow]skip (code={code}): {e}[/yellow]")
        return None


# Class distribution target: intentionally imbalanced toward plausible real-world signal
TARGET_DISTRIBUTION: dict[ClassCode, float] = {
    1: 0.25,  # validation_error
    2: 0.20,  # distraction
    3: 0.20,  # comparison_shopping
    4: 0.15,  # accidental_exit
    5: 0.05,  # bot
    6: 0.15,  # committed_leave
}


def plan_counts(total: int) -> dict[ClassCode, int]:
    counts = {code: int(total * frac) for code, frac in TARGET_DISTRIBUTION.items()}
    # Put remainder into class 1
    counts[1] += total - sum(counts.values())
    return counts


@click.command()
@click.option("--count", default=1000, help="Total examples to generate")
@click.option("--output", default="data/synthetic/raw.jsonl", help="Output JSONL path")
@click.option("--seed", default=42, help="Random seed")
def main(count: int, output: str, seed: int):
    random.seed(seed)
    client = Anthropic()
    counts = plan_counts(count)
    console.print(f"Plan: {dict(counts)}")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w") as f, Progress() as progress:
        task = progress.add_task("Generating", total=count)
        for code, n in counts.items():
            for _ in range(n):
                obj = generate_one(client, code)
                if obj is not None:
                    f.write(json.dumps(obj) + "\n")
                    written += 1
                progress.advance(task)
                time.sleep(0.05)  # light rate limit
    console.print(f"[green]Wrote {written}/{count} examples to {out_path}[/green]")


if __name__ == "__main__":
    main()
