"""Stratified train/val/test splitter."""

import random
from collections import defaultdict


def split_stratified(
    records: list[dict],
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified split by `code`. Returns (train, val, test)."""
    rng = random.Random(seed)
    by_class: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        by_class[r["code"]].append(r)

    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for code, items in sorted(by_class.items()):
        shuffled = items[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_val = int(n * val_frac)
        n_test = int(n * test_frac)
        test.extend(shuffled[:n_test])
        val.extend(shuffled[n_test : n_test + n_val])
        train.extend(shuffled[n_test + n_val :])
    # Shuffle within each split for training noise
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
