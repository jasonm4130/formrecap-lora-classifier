"""Classification + calibration metrics."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_f1(y_true: list[int], y_pred: list[int], classes: list[int]) -> dict[int, float]:
    scores = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return {c: float(s) for c, s in zip(classes, scores)}


def expected_calibration_error(
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """Equal-width binning ECE."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    conf_arr = np.array(confidences)
    correct = (y_true_arr == y_pred_arr).astype(float)
    n = len(y_true_arr)
    if n == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        bucket_size = mask.sum()
        if bucket_size == 0:
            continue
        acc = correct[mask].mean()
        avg_conf = conf_arr[mask].mean()
        ece += (bucket_size / n) * abs(acc - avg_conf)
    return float(ece)


def brier_score(y_true: list[int], y_pred: list[int], confidences: list[float]) -> float:
    """Brier for classifier: squared error of confidence in correctness vs binary correctness."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    conf_arr = np.array(confidences)
    correct = (y_true_arr == y_pred_arr).astype(float)
    return float(np.mean((conf_arr - correct) ** 2))


def bootstrap_ci(
    metric_fn: Callable,
    *args,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap a metric's confidence interval. args are passed as parallel arrays."""
    rng = np.random.default_rng(seed)
    length = len(args[0])
    scores: list[float] = []
    for _ in range(n_iterations):
        idx = rng.integers(0, length, size=length)
        resampled = tuple([list(np.array(a)[idx]) for a in args])
        scores.append(metric_fn(*resampled))
    lower = float(np.percentile(scores, (1 - confidence) / 2 * 100))
    upper = float(np.percentile(scores, (1 + confidence) / 2 * 100))
    return lower, upper


def confusion_matrix(y_true: list[int], y_pred: list[int], classes: list[int]) -> list[list[int]]:
    """Return a len(classes) x len(classes) confusion matrix as nested lists.
    Rows = true class, columns = predicted class."""
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def calibration_buckets(
    y_true: list[int],
    y_pred: list[int],
    confidences: list[float],
    n_bins: int = 10,
) -> list[dict]:
    """Return per-bin calibration data for reliability diagrams."""
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    conf_arr = np.array(confidences)
    correct = (y_true_arr == y_pred_arr).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    buckets = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        midpoint = (lo + hi) / 2
        if i == n_bins - 1:
            mask = (conf_arr >= lo) & (conf_arr <= hi)
        else:
            mask = (conf_arr >= lo) & (conf_arr < hi)
        count = int(mask.sum())
        if count == 0:
            buckets.append(
                {
                    "bin_midpoint": round(float(midpoint), 2),
                    "count": 0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                }
            )
        else:
            buckets.append(
                {
                    "bin_midpoint": round(float(midpoint), 2),
                    "count": count,
                    "avg_confidence": round(float(conf_arr[mask].mean()), 4),
                    "accuracy": round(float(correct[mask].mean()), 4),
                }
            )
    return buckets
