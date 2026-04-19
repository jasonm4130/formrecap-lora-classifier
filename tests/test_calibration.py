import numpy as np

from formrecap_lora.eval.calibration import apply_temperature, fit_temperature


def test_apply_temperature_identity():
    logits = np.array([[1.0, 2.0, 3.0]])
    probs = apply_temperature(logits, T=1.0)
    expected = np.exp([1, 2, 3])
    expected = expected / expected.sum()
    assert np.allclose(probs[0], expected)


def test_apply_temperature_softens():
    logits = np.array([[10.0, 1.0]])
    probs_low_t = apply_temperature(logits, T=1.0)
    probs_high_t = apply_temperature(logits, T=10.0)
    assert probs_high_t[0, 0] < probs_low_t[0, 0]


def test_fit_temperature_on_perfectly_calibrated_returns_near_1():
    np.random.seed(0)
    n_classes = 3
    n = 500
    true_labels = np.random.randint(0, n_classes, size=n)
    logits = np.zeros((n, n_classes))
    for i, y in enumerate(true_labels):
        logits[i, y] = 2.0
    T = fit_temperature(logits, true_labels)
    assert 0.5 < T < 3.0


def test_fit_temperature_on_overconfident_returns_greater_than_1():
    np.random.seed(0)
    n_classes = 3
    n = 200
    true_labels = np.random.randint(0, n_classes, size=n)
    pred_labels = true_labels.copy()
    flip_mask = np.random.rand(n) < 0.5
    pred_labels[flip_mask] = (pred_labels[flip_mask] + 1) % n_classes
    logits = np.zeros((n, n_classes))
    for i, p in enumerate(pred_labels):
        logits[i, p] = 10.0
    T = fit_temperature(logits, true_labels)
    assert T > 1.5
