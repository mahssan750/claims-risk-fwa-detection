import numpy as np
import pandas as pd
import pytest

from claims_risk.modeling import classification_metrics, ensemble_scores, f2_threshold, feature_reasons


@pytest.mark.parametrize(
    ("supervised", "anomaly", "rules", "expected"),
    [
        (np.array([1.0]), np.array([0.0]), np.array([0.0]), 0.65),
        (np.array([0.0]), np.array([1.0]), np.array([0.0]), 0.20),
        (np.array([0.0]), np.array([0.0]), np.array([1.0]), 0.15),
    ],
)
def test_ensemble_component_has_documented_weight(supervised, anomaly, rules, expected) -> None:
    assert ensemble_scores(supervised, anomaly, rules).item() == pytest.approx(expected)


def test_f2_threshold_prioritizes_recall_on_validation_scores() -> None:
    labels = pd.Series([0, 0, 0, 1, 1])
    scores = np.array([0.05, 0.10, 0.40, 0.35, 0.90])

    threshold = f2_threshold(labels, scores)
    metrics = classification_metrics(labels, scores, threshold)

    assert metrics["recall"] == 1.0
    assert metrics["false_negatives"] == 0


def test_feature_reasons_compare_scaled_deviations_instead_of_raw_units() -> None:
    provider_row = pd.Series({"total_reimbursement": 1_000.0, "claim_count": 30.0})
    importance = pd.DataFrame({"feature": ["total_reimbursement", "claim_count"], "importance": [0.1, 0.1]})
    medians = pd.Series({"total_reimbursement": 0.0, "claim_count": 0.0})
    robust_scales = pd.Series({"total_reimbursement": 1_000.0, "claim_count": 5.0})

    reasons = feature_reasons(provider_row, importance, medians, robust_scales)

    assert reasons[0] == "claim count"
