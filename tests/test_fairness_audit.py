import numpy as np

from ml.evaluation.fairness_audit import _group_metrics


class TestGroupMetrics:
    def test_perfect_predictions_have_zero_fpr_and_fnr(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()
        metrics = _group_metrics(y_true, y_pred)
        assert metrics["fpr"] == 0.0
        assert metrics["fnr"] == 0.0
        assert metrics["n"] == 6

    def test_fpr_counts_wrongly_flagged_negatives(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0])  # 2 of 4 true negatives wrongly flagged
        metrics = _group_metrics(y_true, y_pred)
        assert metrics["fpr"] == 0.5
        assert metrics["approval_rate"] == 0.5

    def test_fnr_counts_wrongly_cleared_positives(self):
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1])  # 3 of 4 true positives wrongly cleared
        metrics = _group_metrics(y_true, y_pred)
        assert metrics["fnr"] == 0.75

    def test_approval_rate_is_fraction_predicted_negative(self):
        # prediction=1 means "flagged as high-risk", the adverse outcome here - approval
        # rate is 1 - selection_rate, not the raw mean of y_pred
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0])
        metrics = _group_metrics(y_true, y_pred)
        assert metrics["approval_rate"] == 0.25
