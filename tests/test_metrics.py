import numpy as np
import pytest

from derm_context.metrics import choose_threshold, classification_metrics, high_sensitivity_pauc


def test_perfect_ranking_and_constant_baseline_pauc():
    y = np.array([0, 0, 1, 1])
    assert high_sensitivity_pauc(y, [0.1, 0.2, 0.8, 0.9]) == pytest.approx(0.2)
    assert high_sensitivity_pauc(y, [0.5] * 4) == pytest.approx(0.02)


def test_threshold_reaches_requested_sensitivity_with_ties():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.2, 0.8, 0.9])
    threshold = choose_threshold(y, p, 0.95)
    metrics = classification_metrics(y, p, threshold)
    assert metrics["operating_point"]["sensitivity"] >= 0.95
    assert threshold == pytest.approx(0.2)


def test_invalid_predictions_and_single_class_are_rejected():
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0.2, np.nan])
    with pytest.raises(ValueError):
        classification_metrics([0, 0], [0.1, 0.2])
