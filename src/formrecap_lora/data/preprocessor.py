"""Normalise verbose form event JSON to terse token format for LLM input."""

from typing import TypedDict, NotRequired, Literal


class EventTrace(TypedDict):
    type: Literal["focus", "blur", "input", "scroll", "exit", "submit"]
    field: NotRequired[str]
    ts: int  # milliseconds since session start
    validation: NotRequired[str]
    duration_ms: NotRequired[int]


ALLOWED_TYPES = {"focus", "blur", "input", "scroll", "exit", "submit"}


def _format_event(event: dict) -> str:
    etype = event["type"]
    if etype not in ALLOWED_TYPES:
        raise ValueError(f"Unknown event type: {etype}")
    # ts must exist for every event
    _ = event["ts"]

    if etype == "exit":
        return "exit"
    if etype == "submit":
        return "submit"

    field = event.get("field", "page")
    if etype == "blur" and "validation" in event:
        return f"blur:{field}({event['validation']})"
    if etype == "scroll" and "duration_ms" in event:
        return f"scroll:{field}({event['duration_ms']}ms)"
    return f"{etype}:{field}"


def _compress_consecutive_inputs(tokens: list[str]) -> list[str]:
    """Replace runs of `input:<field>` with `input:<field>(xN)` when N > 1."""
    if not tokens:
        return tokens
    compressed: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("input:") and "(" not in tok:
            run_end = i
            while run_end + 1 < len(tokens) and tokens[run_end + 1] == tok:
                run_end += 1
            count = run_end - i + 1
            if count > 1:
                compressed.append(f"{tok}(x{count})")
            else:
                compressed.append(tok)
            i = run_end + 1
        else:
            compressed.append(tok)
            i += 1
    return compressed


def normalize_events(events: list[dict]) -> str:
    """Convert a list of event dicts into the terse token format used for model input."""
    if not events:
        return ""
    tokens = [_format_event(e) for e in events]
    tokens = _compress_consecutive_inputs(tokens)
    return ", ".join(tokens)
