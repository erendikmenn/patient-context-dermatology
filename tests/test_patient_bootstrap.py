import numpy as np

from derm_context.evaluate_full import paired_bootstrap, weighted_metrics
from derm_context.metrics import classification_metrics


def test_weighted_metrics_equal_explicit_cluster_replication():
    y = np.array([0, 1, 1, 0, 0, 1])
    p = np.array([0.1, 0.8, 0.5, 0.6, 0.3, 0.7])
    weights = np.array([2, 2, 0, 0, 1, 1])
    repeated = np.repeat(np.arange(len(y)), weights)
    expected = classification_metrics(y[repeated], p[repeated])
    np.testing.assert_allclose(
        weighted_metrics(y, p, weights),
        [expected[k] for k in ["average_precision", "roc_auc", "pauc_tpr_ge_0_8_unstandardized"]],
    )


def test_identical_models_have_zero_paired_difference_and_interval():
    y = np.array([0, 1, 1, 0, 0, 1])
    p = np.array([0.1, 0.8, 0.5, 0.6, 0.3, 0.7])
    patients = np.array(["a", "a", "b", "b", "c", "c"])
    result = paired_bootstrap(y, patients, [p, p.copy()], iterations=20)
    assert result["valid_iterations"] == 20
    for metric in result["differences"][0].values():
        assert metric == {"delta": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}
