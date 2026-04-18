import numpy as np
import pytest

from formrecap_lora.eval.metrics import (
    bootstrap_ci,
    brier_score,
    expected_calibration_error,
    macro_f1,
    per_class_f1,
)


def test_macro_f1_perfect():
    y_true = [1, 2, 3, 1, 2, 3]
    y_pred = [1, 2, 3, 1, 2, 3]
    assert macro_f1(y_true, y_pred) == pytest.approx(1.0)


def test_macro_f1_all_wrong():
    y_true = [1, 2, 3]
    y_pred = [2, 3, 1]
    assert macro_f1(y_true, y_pred) == pytest.approx(0.0)


def test_per_class_f1_returns_dict():
    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 2]
    result = per_class_f1(y_true, y_pred, classes=[1, 2])
    assert set(result.keys()) == {1, 2}
    assert 0.0 <= result[1] <= 1.0
    assert result[2] == pytest.approx(0.8)  # precision=2/3, recall=1, F1 = 2*(2/3*1)/(2/3+1) = 0.8


def test_expected_calibration_error_perfect():
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    confidences = [1.0, 1.0, 1.0, 1.0]
    assert expected_calibration_error(y_true, y_pred, confidences, n_bins=10) == pytest.approx(0.0)


def test_expected_calibration_error_overconfident():
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 2, 2]
    confidences = [1.0, 1.0, 1.0, 1.0]
    ece = expected_calibration_error(y_true, y_pred, confidences, n_bins=10)
    assert ece == pytest.approx(0.5)


def test_brier_score_binary_style():
    y_true = [1, 1, 0, 0]
    probs_for_predicted = [0.9, 0.8, 0.6, 0.4]
    y_pred = [1, 1, 1, 0]
    score = brier_score(y_true, y_pred, probs_for_predicted)
    expected = np.mean([(0.9 - 1) ** 2, (0.8 - 1) ** 2, (0.6 - 0) ** 2, (0.4 - 1) ** 2])
    assert score == pytest.approx(expected)


def test_bootstrap_ci_shape():
    y_true = [1, 2, 3] * 20
    y_pred = [1, 2, 3] * 20
    lower, upper = bootstrap_ci(macro_f1, y_true, y_pred, n_iterations=200, seed=42)
    assert 0.9 <= lower <= upper <= 1.0


def test_confusion_matrix_perfect():
    from formrecap_lora.eval.metrics import confusion_matrix

    y_true = [1, 2, 3, 1, 2, 3]
    y_pred = [1, 2, 3, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, classes=[1, 2, 3])
    assert cm == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]


def test_confusion_matrix_with_errors():
    from formrecap_lora.eval.metrics import confusion_matrix

    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 1]
    cm = confusion_matrix(y_true, y_pred, classes=[1, 2])
    assert cm == [[1, 1], [1, 1]]


def test_confusion_matrix_6_classes():
    from formrecap_lora.eval.metrics import confusion_matrix

    y_true = [1, 2, 3, 4, 5, 6]
    y_pred = [1, 2, 3, 4, 5, 6]
    cm = confusion_matrix(y_true, y_pred, classes=[1, 2, 3, 4, 5, 6])
    assert len(cm) == 6
    assert all(len(row) == 6 for row in cm)
    for i in range(6):
        for j in range(6):
            assert cm[i][j] == (1 if i == j else 0)


def test_calibration_buckets_perfect():
    from formrecap_lora.eval.metrics import calibration_buckets

    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    confidences = [0.95, 0.85, 0.75, 0.65]
    buckets = calibration_buckets(y_true, y_pred, confidences, n_bins=10)
    for b in buckets:
        if b["count"] > 0:
            assert b["accuracy"] == 1.0


def test_calibration_buckets_structure():
    from formrecap_lora.eval.metrics import calibration_buckets

    y_true = [1, 1, 2, 2]
    y_pred = [1, 2, 2, 1]
    confidences = [0.9, 0.8, 0.7, 0.6]
    buckets = calibration_buckets(y_true, y_pred, confidences, n_bins=10)
    assert len(buckets) == 10
    assert all("bin_midpoint" in b for b in buckets)
    assert all("count" in b for b in buckets)
    assert all("avg_confidence" in b for b in buckets)
    assert all("accuracy" in b for b in buckets)
