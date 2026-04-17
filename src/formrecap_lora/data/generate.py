"""Generate synthetic training examples via Claude Sonnet 4.5."""

import json
import random
import time
from pathlib import Path

import click
from anthropic import Anthropic
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress

from .primers import CLASS_NAMES, ClassCode, get_primers_for_class

load_dotenv()
console = Console()

SYSTEM_PROMPT = """You generate synthetic form interaction event traces for training a form abandonment classifier.

## Format

Events are terse tokens separated by ", ":
- `focus:field` — user clicked/tabbed into a field
- `input:field(xN)` — N consecutive keystrokes in that field
- `blur:field` — user left the field (no validation issue)
- `blur:field(reason)` — user left the field and validation failed
- `scroll:page(Nms)` — user scrolled/idled for N milliseconds
- `exit` — user left the page
- `submit` — user submitted the form

## Output

ONE JSON object with keys: events, code, reason, confidence.

## CRITICAL: Diversity Rules

You MUST vary each example. Do NOT repeat patterns from the examples shown.

1. **Field names**: Pick from a WIDE variety. Not just email/password/name — use fields like:
   - Registration: username, display_name, bio, avatar, referral_code, terms_checkbox
   - Checkout: card_number, card_expiry, cvv, billing_address, shipping_address, coupon_code, gift_message
   - Contact: subject, department, priority, attachment, callback_number
   - Application: resume, cover_letter, salary_expectation, start_date, work_rights, references
   - Profile: timezone, language, notification_prefs, two_factor_setup
   - Business: abn, company_name, industry, employee_count, revenue_range, website_url

2. **Form contexts**: Think about DIFFERENT form types — signup, checkout, job application, insurance quote, loan application, contact form, survey, booking form, support ticket, account settings.

3. **Sequence length**: Vary between 2 and 25 events. Short traces AND long traces.

4. **Timing**: scroll durations should range from 500ms to 300000ms (5 minutes). Use realistic values.

5. **Validation reasons**: Use varied, realistic reasons — not just "invalid_format". Examples: required, too_short, too_long, invalid_format, taken, mismatch, invalid_date, invalid_luhn, out_of_range, unsupported_country, expired, blacklisted, profanity, invalid_abn, weak_password, missing_special, missing_uppercase, disposable_email.

6. **Confidence**: Should genuinely reflect ambiguity. Ambiguous traces (could be class X or Y) should have confidence 0.45-0.65. Clear-cut cases 0.80-0.95.

7. **The `reason` field**: Must cite SPECIFIC events from the trace. Not generic descriptions.

8. **NEVER produce the same event trace as a previous example.** Each trace must be structurally unique."""

# Form contexts to inject variety into the generation prompt
FORM_CONTEXTS = [
    "a user signup form (name, email, password, username)",
    "an e-commerce checkout (billing, shipping, payment, coupon)",
    "a job application form (personal info, resume, cover letter, references)",
    "a contact/support form (name, email, subject, message, priority)",
    "an insurance quote form (personal details, vehicle/property info, coverage options)",
    "a loan/mortgage application (income, employment, assets, liabilities)",
    "a booking/reservation form (dates, guests, room type, special requests)",
    "a business registration form (company name, ABN, industry, employee count)",
    "a medical/patient intake form (name, DOB, insurance, medical history, allergies)",
    "an event registration form (name, email, ticket type, dietary requirements)",
    "an account settings/profile update form (display name, bio, timezone, notifications)",
    "a survey or feedback form (ratings, comments, NPS score)",
]

# Per-class behavioral hints to guide generation
CLASS_HINTS: dict[ClassCode, str] = {
    1: (
        "The user hits a validation error they cannot resolve. Key signals: "
        "repeated focus/blur cycles on the SAME field with validation failures, "
        "increasing keystroke counts on retries, no forward progress past the error point. "
        "The user is TRYING to complete the form but a specific field blocks them."
    ),
    2: (
        "The user gets distracted and task-switches away. Key signals: "
        "normal steady field progression followed by a LONG idle gap (30-300+ seconds), "
        "partial form completion (some fields filled correctly), no validation errors. "
        "The idle gap is the primary signal — it should be much longer than normal inter-field pauses."
    ),
    3: (
        "The user is browsing/evaluating, not committing. Key signals: "
        "high scroll-to-input ratio, focus/blur on read-only info fields WITHOUT typing, "
        "looking at pricing/features/terms/shipping/reviews sections. "
        "Minimal or zero actual field input. They're gathering information, not filling a form."
    ),
    4: (
        "The user accidentally navigated away. Key signals: "
        "steady input velocity across multiple fields, NO errors, NO idle time, "
        "abrupt termination mid-flow (sometimes mid-keystroke). "
        "The trace shows someone actively engaged who was interrupted by a back button, "
        "accidental link click, or tab close."
    ),
    5: (
        "Automated non-human interaction. Key signals: "
        "NO focus events (direct input injection), no blur events, no scroll events, "
        "fields filled in strict DOM order, input bursts with unrealistic keystroke counts, "
        "ends with submit (not exit). The trace should look mechanically impossible for a human."
    ),
    6: (
        "The user intentionally decides not to complete the form. Key signals: "
        "browsed the form, maybe focused a few fields without typing, "
        "possibly scrolled to read terms/pricing/requirements. "
        "Short session with deliberate departure — they evaluated and decided no. "
        "Different from distraction (no long idle) and comparison_shopping (less systematic browsing)."
    ),
}


def _few_shot_block(code: ClassCode, n_primers: int = 2) -> list[dict]:
    primers = get_primers_for_class(code)
    chosen = random.sample(primers, min(n_primers, len(primers)))
    blocks: list[dict] = []
    for p in chosen:
        blocks.append(
            {
                "role": "user",
                "content": f"Generate a new example for class: {CLASS_NAMES[code]} (code {code})",
            }
        )
        blocks.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "events": p["events"],
                        "code": p["code"],
                        "reason": p["reason"],
                        "confidence": p["confidence"],
                    }
                ),
            }
        )
    return blocks


def generate_one(
    client: Anthropic,
    code: ClassCode,
    seen_events: set[str],
    batch_index: int,
) -> dict | None:
    # Randomize the form context for each call to increase variety
    form_context = random.choice(FORM_CONTEXTS)
    seq_len_hint = random.choice(["short (3-6 events)", "medium (7-12 events)", "long (13-25 events)"])

    messages = _few_shot_block(code, n_primers=2)
    messages.append(
        {
            "role": "user",
            "content": (
                f"Generate a new example for class: {CLASS_NAMES[code]} (code {code}).\n\n"
                f"Context: This is {form_context}.\n"
                f"Target sequence length: {seq_len_hint}.\n"
                f"Behavioral hint: {CLASS_HINTS[code]}\n\n"
                f"This is example #{batch_index} — make it DIFFERENT from the primers above. "
                f"Use different field names, different sequence structure, different timing."
            ),
        }
    )
    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            temperature=1.0,
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
        # Reject exact duplicates within this run
        if obj["events"] in seen_events:
            return None
        seen_events.add(obj["events"])
        return obj
    except Exception as e:
        console.print(f"[yellow]skip (code={code}): {e}[/yellow]")
        return None


# Class distribution target: intentionally imbalanced toward plausible real-world traffic
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
@click.option("--max-retries", default=3, help="Max retries per example on duplicate")
def main(count: int, output: str, seed: int, max_retries: int):
    random.seed(seed)
    client = Anthropic()
    counts = plan_counts(count)
    console.print(f"Plan: {dict(counts)}")
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_events: set[str] = set()
    written = 0
    skipped_dupes = 0

    with out_path.open("w") as f, Progress() as progress:
        task = progress.add_task("Generating", total=count)
        for code, n in counts.items():
            batch_idx = 0
            for _ in range(n):
                batch_idx += 1
                obj = None
                for retry in range(max_retries):
                    obj = generate_one(client, code, seen_events, batch_idx)
                    if obj is not None:
                        break
                    skipped_dupes += 1
                if obj is not None:
                    f.write(json.dumps(obj) + "\n")
                    written += 1
                progress.advance(task)
                time.sleep(0.05)

    console.print(f"[green]Wrote {written}/{count} unique examples to {out_path}[/green]")
    if skipped_dupes:
        console.print(f"[yellow]Skipped {skipped_dupes} duplicate retries[/yellow]")


if __name__ == "__main__":
    main()
