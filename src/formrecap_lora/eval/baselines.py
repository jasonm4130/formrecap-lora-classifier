"""Baseline predictors. Each returns predictions + (optional) confidences for the test set."""

import json
import os
import random
from collections import Counter

import httpx
from anthropic import Anthropic

from formrecap_lora.data.primers import CLASS_NAMES, PRIMERS


def majority_class_baseline(train_records: list[dict], test_records: list[dict]) -> dict:
    counts = Counter(r["code"] for r in train_records)
    top = counts.most_common(1)[0][0]
    preds = [top] * len(test_records)
    confidences = [counts[top] / sum(counts.values())] * len(test_records)
    return {"preds": preds, "confidences": confidences}


SYSTEM_PROMPT = (
    "You analyse form interaction event sequences and classify the likely abandonment reason. "
    "Respond with a digit class code 1-6 on the first line, then a JSON object on the second line "
    "with class, reason, and confidence fields. "
    "Classes: 1=validation_error, 2=distraction, 3=comparison_shopping, "
    "4=accidental_exit, 5=bot, 6=committed_leave."
)


def _parse_response(text: str) -> tuple[int | None, float]:
    """Extract class code (first integer 1-6) + confidence from model output."""
    import re

    first_line, _, rest = text.strip().partition("\n")
    m = re.search(r"([1-6])", first_line)
    code = int(m.group(1)) if m else None
    conf = 0.5
    try:
        obj = json.loads(rest.strip())
        conf = float(obj.get("confidence", 0.5))
    except Exception:
        pass
    return code, conf


def _events_to_user_message(events: str) -> str:
    return f"Events: {events}"


def zero_shot_llama_via_cf(
    test_records: list[dict], model: str = "@cf/meta/llama-3.2-3b-instruct"
) -> dict:
    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    preds: list[int | None] = []
    confidences: list[float] = []
    for rec in test_records:
        r = httpx.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _events_to_user_message(rec["events"])},
                ],
                "max_tokens": 200,
            },
            timeout=60,
        )
        text = r.json().get("result", {}).get("response", "")
        code, conf = _parse_response(text)
        preds.append(code)
        confidences.append(conf)
    return {"preds": preds, "confidences": confidences}


def few_shot_llama_via_cf(
    test_records: list[dict],
    n_shots: int = 5,
    model: str = "@cf/meta/llama-3.2-3b-instruct",
) -> dict:
    shots = PRIMERS[:n_shots]
    example_messages: list[dict] = []
    for p in shots:
        example_messages.append(
            {"role": "user", "content": _events_to_user_message(p["events"])}
        )
        example_messages.append(
            {
                "role": "assistant",
                "content": f'{p["code"]}\n{{"class": "{CLASS_NAMES[p["code"]]}", "reason": {json.dumps(p["reason"])}, "confidence": {p["confidence"]}}}',
            }
        )

    account = os.environ["CLOUDFLARE_ACCOUNT_ID"]
    token = os.environ["CLOUDFLARE_API_TOKEN"]
    preds: list[int | None] = []
    confidences: list[float] = []
    for rec in test_records:
        r = httpx.post(
            f"https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/{model}",
            headers={"Authorization": f"Bearer {token}"},
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *example_messages,
                    {"role": "user", "content": _events_to_user_message(rec["events"])},
                ],
                "max_tokens": 200,
            },
            timeout=60,
        )
        text = r.json().get("result", {}).get("response", "")
        code, conf = _parse_response(text)
        preds.append(code)
        confidences.append(conf)
    return {"preds": preds, "confidences": confidences}


def claude_haiku_baseline(test_records: list[dict]) -> dict:
    client = Anthropic()
    preds: list[int | None] = []
    confidences: list[float] = []
    for rec in test_records:
        r = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _events_to_user_message(rec["events"])}],
        )
        text = r.content[0].text
        code, conf = _parse_response(text)
        preds.append(code)
        confidences.append(conf)
    return {"preds": preds, "confidences": confidences}
