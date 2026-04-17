"""Deduplication: exact match on event strings + cosine similarity utility."""

import math


def dedupe_exact(records: list[dict]) -> list[dict]:
    """Remove records with identical `events` strings. Keeps first occurrence."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in records:
        key = r["events"]
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def is_near_duplicate(e1: list[float], e2: list[float], threshold: float = 0.95) -> bool:
    """Check if two embedding vectors are near-duplicates by cosine similarity."""
    return _cosine(e1, e2) >= threshold
