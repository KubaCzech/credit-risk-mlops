import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from ..config import CV_FOLDS, RANDOM_STATE

# average_precision = area under the precision-recall curve (PR-AUC): the metric
# that actually matters here, since ROC-AUC is optimistic on 6.7%-positive data.
# accuracy is tracked too, but only as a record of *why it's misleading* on this dataset
# (a do-nothing "always predict 0" model already scores ~93%) - never used to rank models.
SCORING = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "pr_auc": "average_precision",
    "accuracy": "accuracy",
}


def cross_validate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Stratified K-Fold CV on train only - test set stays untouched until final model selection."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return cross_validate(pipeline, X, y, cv=cv, scoring=SCORING, n_jobs=1)


def summarize_cv_results(cv_results: dict) -> dict:
    summary = {}
    for metric in SCORING:
        scores = cv_results[f"test_{metric}"]
        summary[f"{metric}_mean"] = float(np.mean(scores))
        summary[f"{metric}_std"] = float(np.std(scores))
    return summary
