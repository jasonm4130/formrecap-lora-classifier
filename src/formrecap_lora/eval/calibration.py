"""Post-hoc temperature scaling calibration."""

import numpy as np
from scipy.optimize import minimize_scalar


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Apply temperature scaling: softmax(logits / T). logits shape (N, C)."""
    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)  # numeric stability
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(T: float, logits: np.ndarray, labels: np.ndarray) -> float:
    probs = apply_temperature(logits, T)
    eps = 1e-12
    return float(-np.mean(np.log(probs[np.arange(len(labels)), labels] + eps)))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit temperature scalar by minimising NLL on held-out set. logits shape (N, C), labels (N,)."""
    res = minimize_scalar(
        _nll,
        bounds=(0.5, 20.0),
        method="bounded",
        args=(logits, labels),
        options={"xatol": 1e-3},
    )
    return float(res.x)


def calibrate_and_persist(logits: np.ndarray, labels: np.ndarray, out_path: str) -> dict:
    """Fit + write to JSON file."""
    import json
    from pathlib import Path

    T = fit_temperature(logits, labels)
    result = {"temperature": T, "fit_size": int(len(labels))}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result
