"""Run all baselines + our fine-tune, compute metrics, emit results table."""

import json
import math
from pathlib import Path

import click
import modal
import numpy as np
from rich.console import Console
from rich.table import Table

from formrecap_lora.eval.baselines import (
    claude_haiku_baseline,
    few_shot_llama_via_cf,
    majority_class_baseline,
    zero_shot_llama_via_cf,
)
from formrecap_lora.eval.calibration import (
    apply_temperature,
    calibrate_and_persist,
    fit_temperature,
)
from formrecap_lora.eval.metrics import (
    bootstrap_ci,
    brier_score,
    calibration_buckets,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    per_class_f1,
)

console = Console()

N_CLASSES = 6


def load_jsonl_chat(path: str) -> list[dict]:
    """Load a chat-template JSONL and return simplified records (events, code, reason, confidence)."""
    records: list[dict] = []
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
    """Real test set is the flat schema from hand_label.py."""
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def our_lora_via_modal(
    run_id: str, test_records: list[dict], base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
) -> dict:
    """Call Modal Predictor class for each record. Model stays loaded between calls."""
    predictor = modal.Cls.from_name("formrecap-lora", "Predictor")(
        run_id=run_id, base_model=base_model
    )
    preds: list[int | None] = []
    verbalized_conf: list[float] = []
    logprob_conf: list[float] = []
    first_token_logprobs_by_class: list[dict[int, float]] = []
    for rec in test_records:
        result = predictor.predict_with_logprobs.remote(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You analyse form interaction event sequences and classify the likely "
                        "abandonment reason. Respond with a digit class code 1-6 on the first line, "
                        "then a JSON object on the second line with class, reason, and confidence fields. "
                        "Classes: 1=validation_error, 2=distraction, 3=comparison_shopping, "
                        "4=accidental_exit, 5=bot, 6=committed_leave."
                    ),
                },
                {"role": "user", "content": f"Events: {rec['events']}"},
            ],
        )
        first_token_str = result["first_token"]["token"].strip()
        try:
            code = int(first_token_str)
            assert 1 <= code <= 6
            preds.append(code)
        except Exception:
            preds.append(None)
        # verbalized confidence
        try:
            _, _, rest = result["text"].strip().partition("\n")
            obj = json.loads(rest.strip())
            verbalized_conf.append(float(obj.get("confidence", 0.5)))
        except Exception:
            verbalized_conf.append(0.5)
        logprob_conf.append(math.exp(result["first_token"]["logprob"]))
        # Extract per-class first-token logprob for temp scaling
        candidates = {c["token"].strip(): c["logprob"] for c in result["top_candidates"]}
        per_class = {k: candidates.get(str(k), -100.0) for k in range(1, 7)}
        first_token_logprobs_by_class.append(per_class)
    return {
        "preds": preds,
        "verbalized_confidences": verbalized_conf,
        "logprob_confidences": logprob_conf,
        "per_class_logprobs": first_token_logprobs_by_class,
    }


def summarise(name: str, preds: list, labels: list, confidences: list | None = None) -> dict:
    present_classes = list(range(1, N_CLASSES + 1))
    # Filter Nones
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


@click.command()
@click.option("--run-id", required=True)
@click.option("--train-file", default="data/synthetic/train-2026-04-18.jsonl")
@click.option("--val-file", default="data/synthetic/val-2026-04-18.jsonl")
@click.option("--test-real", default="data/real/test.jsonl")
@click.option("--output-dir", default="docs/results")
@click.option("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
@click.option("--skip-baselines", is_flag=True, help="Skip baselines, only run LoRA eval")
def main(
    run_id: str,
    train_file: str,
    val_file: str,
    test_real: str,
    output_dir: str,
    base_model: str,
    skip_baselines: bool,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Loading data...[/bold]")
    train = load_jsonl_chat(train_file)
    val = load_jsonl_chat(val_file)
    test_real_records = load_real_test(test_real)

    labels = [r["code"] for r in test_real_records]

    results: list[dict] = []

    if not skip_baselines:
        # 1. Majority
        console.print("[cyan]1/5 Majority class baseline[/cyan]")
        maj = majority_class_baseline(train, test_real_records)
        results.append(summarise("Majority class", maj["preds"], labels, maj["confidences"]))

        # 2. Zero-shot Llama 3.2 3B
        console.print("[cyan]2/5 Zero-shot Llama 3.2 3B[/cyan]")
        zs = zero_shot_llama_via_cf(test_real_records)
        results.append(summarise("Zero-shot Llama 3.2 3B", zs["preds"], labels, zs["confidences"]))

        # 3. 5-shot Llama
        console.print("[cyan]3/5 5-shot Llama 3.2 3B[/cyan]")
        fs = few_shot_llama_via_cf(test_real_records, n_shots=5)
        results.append(summarise("5-shot Llama 3.2 3B", fs["preds"], labels, fs["confidences"]))

        # 4. Claude Haiku
        console.print("[cyan]4/5 Claude Haiku 4.5[/cyan]")
        ch = claude_haiku_baseline(test_real_records)
        results.append(summarise("Claude Haiku 4.5", ch["preds"], labels, ch["confidences"]))

    # Zero-shot baseline for this base model (via Modal, no adapter)
    model_short = base_model.split("/")[-1]
    console.print(f"[cyan]Zero-shot {model_short} via Modal[/cyan]")
    zs_modal = our_lora_via_modal("none", test_real_records, base_model=base_model)
    results.append(
        summarise(
            f"Zero-shot {model_short}", zs_modal["preds"], labels, zs_modal["logprob_confidences"]
        )
    )

    # Our LoRA on Modal
    console.print(f"[cyan]LoRA {model_short} via Modal (with logprobs)[/cyan]")
    ours = our_lora_via_modal(run_id, test_real_records, base_model=base_model)

    # Verbalized
    r_verbal = summarise(
        "Our LoRA (verbalized confidence)", ours["preds"], labels, ours["verbalized_confidences"]
    )
    results.append(r_verbal)

    # Logprob raw
    r_lp = summarise(
        "Our LoRA (logprob confidence, raw)", ours["preds"], labels, ours["logprob_confidences"]
    )
    results.append(r_lp)

    # Fit temperature on val set
    console.print("[cyan]Fitting temperature scalar on val...[/cyan]")
    val_ours = our_lora_via_modal(
        run_id,
        [
            {
                "events": v["events"],
                "code": v["code"],
                "reason": v["reason"],
                "confidence": v["confidence"],
            }
            for v in val
        ],
        base_model=base_model,
    )
    val_logits = np.array([[per[c] for c in range(1, 7)] for per in val_ours["per_class_logprobs"]])
    val_labels_idx = np.array([v["code"] - 1 for v in val])  # 0-indexed
    T = fit_temperature(val_logits, val_labels_idx)
    console.print(f"[green]Fitted T = {T:.3f}[/green]")
    calibrate_and_persist(val_logits, val_labels_idx, str(out / "calibration.json"))

    # Apply temperature to test-set per-class logits
    test_logits = np.array([[per[c] for c in range(1, 7)] for per in ours["per_class_logprobs"]])
    test_probs = apply_temperature(test_logits, T)
    test_preds_temp = [int(np.argmax(p) + 1) for p in test_probs]
    test_conf_temp = [float(np.max(p)) for p in test_probs]
    r_temp = summarise(
        "Our LoRA (logprob + temperature scaled)", test_preds_temp, labels, test_conf_temp
    )
    results.append(r_temp)

    # Save detailed predictions for charts
    classes = list(range(1, N_CLASSES + 1))
    valid_preds = [p for p in ours["preds"] if p is not None]
    valid_labels = [l for p, l in zip(ours["preds"], labels) if p is not None]
    valid_verbalized = [
        c for p, c in zip(ours["preds"], ours["verbalized_confidences"]) if p is not None
    ]
    valid_logprob = [c for p, c in zip(ours["preds"], ours["logprob_confidences"]) if p is not None]
    valid_calibrated = [c for p, c in zip(ours["preds"], test_conf_temp) if p is not None]

    detailed = {
        "run_id": run_id,
        "base_model": base_model,
        "temperature": T,
        "predictions": [
            {
                "true_label": int(l),
                "predicted_label": int(p),
                "verbalized_confidence": round(float(vc), 4),
                "logprob_confidence": round(float(lc), 4),
                "calibrated_confidence": round(float(cc), 4),
            }
            for p, l, vc, lc, cc in zip(
                valid_preds, valid_labels, valid_verbalized, valid_logprob, valid_calibrated
            )
        ],
        "confusion_matrix": {
            "classes": classes,
            "matrix": confusion_matrix(valid_labels, valid_preds, classes),
        },
        "calibration_buckets": {
            "logprob_raw": calibration_buckets(valid_labels, valid_preds, valid_logprob),
            "calibrated": calibration_buckets(valid_labels, valid_preds, valid_calibrated),
            "verbalized": calibration_buckets(valid_labels, valid_preds, valid_verbalized),
        },
    }
    detailed_path = out / f"detailed-{run_id}.json"
    detailed_path.write_text(json.dumps(detailed, indent=2))
    console.print(f"[green]Wrote detailed data to {detailed_path}[/green]")

    # Render table
    table = Table(title=f"Results for run {run_id}")
    table.add_column("System")
    table.add_column("Macro-F1", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("ECE", justify="right")
    table.add_column("Brier", justify="right")
    for r in results:
        f1 = f"{r['macro_f1']:.3f}"
        ci = f"[{r['macro_f1_ci'][0]:.2f}, {r['macro_f1_ci'][1]:.2f}]"
        ece = f"{r['ece']:.3f}" if r.get("ece") is not None else "-"
        brier_val = f"{r['brier']:.3f}" if r.get("brier") is not None else "-"
        table.add_row(r["name"], f1, ci, ece, brier_val)
    console.print(table)

    # Persist
    (out / f"results-{run_id}.json").write_text(
        json.dumps({"run_id": run_id, "temperature": T, "results": results}, indent=2)
    )
    console.print(f"[green]Wrote {out / f'results-{run_id}.json'}[/green]")


if __name__ == "__main__":
    main()
