"""Unified eval runner using vLLM servers on Modal.

Evaluates multiple models in one run: zero-shot baseline + LoRA for each.
Uses OpenAI-compatible API from vLLM for fast inference with logprobs.
"""

import json
import math
from pathlib import Path

import click
import numpy as np
from openai import OpenAI
from rich.console import Console
from rich.table import Table

from formrecap_lora.eval.calibration import (
    apply_temperature,
    calibrate_and_persist,
    fit_temperature,
)
from formrecap_lora.eval.metrics import (
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    macro_f1,
    per_class_f1,
)

console = Console()
N_CLASSES = 6

SYSTEM_CONTENT = (
    "You analyse form interaction event sequences and classify the likely abandonment reason. "
    "Respond with a digit class code 1-6 on the first line, then a JSON object on the second line "
    "with class, reason, and confidence fields. "
    "Classes: 1=validation_error, 2=distraction, 3=comparison_shopping, "
    "4=accidental_exit, 5=bot, 6=committed_leave."
)

# Models that don't support system role — merge into user turn
NO_SYSTEM_ROLE = {"google/gemma-2b-it", "google/gemma-7b-it", "mistralai/Mistral-7B-Instruct-v0.2"}


def build_messages(events: str, base_model: str) -> list[dict]:
    if base_model in NO_SYSTEM_ROLE:
        return [{"role": "user", "content": f"{SYSTEM_CONTENT}\n\n Events: {events}"}]
    return [
        {"role": "system", "content": SYSTEM_CONTENT},
        {"role": "user", "content": f"Events: {events}"},
    ]


def load_jsonl_chat(path: str) -> list[dict]:
    records = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        user = next(m for m in msgs if m["role"] == "user")["content"]
        events = user.replace("Events: ", "", 1)
        assistant = next(m for m in msgs if m["role"] == "assistant")["content"]
        first_line, _, rest = assistant.partition("\n")
        code = int(first_line.strip())
        obj = json.loads(rest)
        records.append(
            {
                "events": events,
                "code": code,
                "reason": obj.get("reason", ""),
                "confidence": float(obj.get("confidence", 0.5)),
            }
        )
    return records


def load_real_test(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _process_one(
    client: OpenAI,
    model_name: str,
    rec: dict,
    base_model: str,
    idx: int,
    total: int,
) -> dict:
    """Process a single record. Returns a result dict."""
    messages = build_messages(rec["events"], base_model)
    empty = {
        "pred": None,
        "verbalized_conf": 0.5,
        "logprob_conf": 0.0,
        "per_class": {k: -100.0 for k in range(1, 7)},
    }
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=64,
            temperature=0,
            logprobs=True,
            top_logprobs=10,
        )
    except Exception as e:
        console.print(f"[yellow]  Call {idx + 1}/{total} failed: {e}[/yellow]")
        return empty

    text = response.choices[0].message.content or ""
    logprobs_data = response.choices[0].logprobs

    result = dict(empty)

    if logprobs_data and logprobs_data.content:
        # Find the first digit token (Mistral prepends a space token)
        digit_token = None
        for tok in logprobs_data.content:
            if tok.token.strip() and tok.token.strip()[0].isdigit():
                digit_token = tok
                break
        if digit_token is None:
            digit_token = logprobs_data.content[0]

        token_str = digit_token.token.strip()
        try:
            code = int(token_str)
            assert 1 <= code <= 6
            result["pred"] = code
        except Exception:
            pass

        result["logprob_conf"] = math.exp(digit_token.logprob)
        top_lp = {lp.token.strip(): lp.logprob for lp in digit_token.top_logprobs}
        result["per_class"] = {k: top_lp.get(str(k), -100.0) for k in range(1, 7)}

    # Fallback: extract class from text if logprob extraction missed it
    if result["pred"] is None and text.strip():
        first_char = text.strip()[0]
        try:
            code = int(first_char)
            if 1 <= code <= 6:
                result["pred"] = code
        except ValueError:
            pass

    try:
        _, _, rest = text.strip().partition("\n")
        obj = json.loads(rest.strip())
        result["verbalized_conf"] = float(obj.get("confidence", 0.5))
    except Exception:
        pass

    return result


def predict_batch(
    client: OpenAI,
    model_name: str,
    records: list[dict],
    base_model: str,
    max_workers: int = 8,
) -> dict:
    """Run predictions in parallel. Returns preds, confidences, logprobs."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(records)
    total = len(records)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one, client, model_name, rec, base_model, i, total): i
            for i, rec in enumerate(records)
        }
        for future in as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()
            done += 1
            if done % 10 == 0:
                console.print(f"  [{done}/{total}]")

    preds = [r["pred"] for r in results]
    verbalized_conf = [r["verbalized_conf"] for r in results]
    logprob_conf = [r["logprob_conf"] for r in results]
    per_class_logprobs = [r["per_class"] for r in results]

    return {
        "preds": preds,
        "verbalized_confidences": verbalized_conf,
        "logprob_confidences": logprob_conf,
        "per_class_logprobs": per_class_logprobs,
    }


def summarise(name: str, preds: list, labels: list, confidences: list | None = None) -> dict:
    present_classes = list(range(1, N_CLASSES + 1))
    valid_pairs = [
        (p, l, c)
        for p, l, c in zip(preds, labels, confidences or [0.5] * len(preds))
        if p is not None
    ]
    if not valid_pairs:
        return {"name": name, "macro_f1": 0.0, "macro_f1_ci": [0.0, 0.0]}
    p = [pp for pp, _, _ in valid_pairs]
    ll = [ll for _, ll, _ in valid_pairs]
    c = [cc for _, _, cc in valid_pairs]
    mf1 = macro_f1(ll, p)
    pcf1 = per_class_f1(ll, p, classes=present_classes)
    ece = expected_calibration_error(ll, p, c) if confidences else None
    brier = brier_score(ll, p, c) if confidences else None
    ci_low, ci_high = bootstrap_ci(macro_f1, ll, p, n_iterations=1000)
    return {
        "name": name,
        "macro_f1": mf1,
        "macro_f1_ci": [ci_low, ci_high],
        "per_class_f1": pcf1,
        "ece": ece,
        "brier": brier,
        "n_valid": len(valid_pairs),
        "n_total": len(preds),
    }


MODEL_CONFIGS = [
    {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "run_id": "baseline-3b",
        "label": "Llama 3.2 3B",
        "server_url": None,  # filled at runtime from Modal
    },
    {
        "base_model": "google/gemma-2b-it",
        "run_id": "gemma-2b",
        "label": "Gemma 2B",
        "server_url": None,
    },
    {
        "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
        "run_id": "mistral-7b",
        "label": "Mistral 7B",
        "server_url": None,
    },
]


@click.command()
@click.option("--val-file", default="data/synthetic/val-2026-04-18.jsonl")
@click.option("--test-real", default="data/real/test.jsonl")
@click.option("--output-dir", default="docs/results")
@click.option("--server-url", required=True, help="vLLM server URL")
@click.option("--base-model", required=True, help="Base model name")
@click.option("--run-id", required=True, help="LoRA adapter run ID")
@click.option("--label", default="", help="Display label for the model")
def main(
    val_file: str,
    test_real: str,
    output_dir: str,
    server_url: str,
    base_model: str,
    run_id: str,
    label: str,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    label = label or base_model.split("/")[-1]
    configs = [
        {"base_model": base_model, "run_id": run_id, "label": label, "server_url": server_url}
    ]

    console.print("[bold]Loading data...[/bold]")
    val = load_jsonl_chat(val_file)
    test_real_records = load_real_test(test_real)
    labels = [r["code"] for r in test_real_records]

    all_results: list[dict] = []

    for cfg in configs:
        base_model = cfg["base_model"]
        run_id = cfg["run_id"]
        label = cfg["label"]
        url = cfg["server_url"]
        client = OpenAI(base_url=f"{url}/v1", api_key="not-needed")

        console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
        console.print(f"[bold]{label}[/bold] ({base_model})")
        console.print(f"[bold cyan]{'=' * 60}[/bold cyan]")

        # Zero-shot baseline (use base model name)
        console.print(f"  [cyan]Zero-shot {label}...[/cyan]")
        zs = predict_batch(client, base_model, test_real_records, base_model)
        all_results.append(
            summarise(f"Zero-shot {label}", zs["preds"], labels, zs["logprob_confidences"])
        )

        # LoRA
        console.print(f"  [cyan]LoRA {label} (test set)...[/cyan]")
        lora = predict_batch(client, run_id, test_real_records, base_model)
        all_results.append(
            summarise(
                f"LoRA {label} (verbalized)", lora["preds"], labels, lora["verbalized_confidences"]
            )
        )
        all_results.append(
            summarise(
                f"LoRA {label} (logprob raw)", lora["preds"], labels, lora["logprob_confidences"]
            )
        )

        # Temperature calibration on val set
        console.print("  [cyan]Calibration on val set...[/cyan]")
        val_records = [
            {
                "events": v["events"],
                "code": v["code"],
                "reason": v["reason"],
                "confidence": v["confidence"],
            }
            for v in val
        ]
        val_lora = predict_batch(client, run_id, val_records, base_model)
        val_logits = np.array(
            [[per[c] for c in range(1, 7)] for per in val_lora["per_class_logprobs"]]
        )
        val_labels_idx = np.array([v["code"] - 1 for v in val])
        T = fit_temperature(val_logits, val_labels_idx)
        console.print(f"  [green]Fitted T = {T:.3f}[/green]")
        calibrate_and_persist(val_logits, val_labels_idx, str(out / f"calibration-{run_id}.json"))

        # Apply temperature to test set
        test_logits = np.array(
            [[per[c] for c in range(1, 7)] for per in lora["per_class_logprobs"]]
        )
        test_probs = apply_temperature(test_logits, T)
        test_preds_temp = [int(np.argmax(p) + 1) for p in test_probs]
        test_conf_temp = [float(np.max(p)) for p in test_probs]
        all_results.append(
            summarise(f"LoRA {label} (calibrated)", test_preds_temp, labels, test_conf_temp)
        )

    # Render combined table
    table = Table(title="Combined Results — All Models")
    table.add_column("System")
    table.add_column("Macro-F1", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Brier", justify="right")
    for r in all_results:
        f1 = f"{r['macro_f1']:.3f}"
        ci = f"[{r['macro_f1_ci'][0]:.2f}, {r['macro_f1_ci'][1]:.2f}]"
        ece = f"{r['ece']:.3f}" if r.get("ece") is not None else "-"
        brier_val = f"{r['brier']:.3f}" if r.get("brier") is not None else "-"
        table.add_row(r["name"], f1, ci, ece, brier_val)
    console.print(table)

    # Persist
    results_path = out / "results-all-models.json"
    results_path.write_text(json.dumps({"results": all_results}, indent=2))
    console.print(f"[green]Wrote {results_path}[/green]")


if __name__ == "__main__":
    main()
