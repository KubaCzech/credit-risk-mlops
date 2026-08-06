import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]


class DelinquencyAggregator(BaseEstimator, TransformerMixin):
    """Sums the 3 delinquency columns into one severity score, plus a simple ever-late flag.

    The 3 columns only correlate with each other at 0.22-0.31 (see EDA notebook) - they
    carry overlapping but distinct signal, so the sum is added alongside them rather than
    replacing them."""

    def __init__(self, cols: list[str] = None):
        self.cols = cols if cols is not None else DELINQUENCY_COLS

    def fit(self, X: pd.DataFrame, y=None) -> "DelinquencyAggregator":
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["total_delinquency"] = X[self.cols].sum(axis=1)
        X["has_any_delinquency"] = (X["total_delinquency"] > 0).astype(int)
        return X


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Hand-picked interaction terms - the explicit way to give a *linear* model (LogReg,
    linear SVM) access to a non-linear combination of two features, which it cannot form
    on its own. Built after preprocessing, so RevolvingUtilization/MonthlyIncome are
    already winsorized here, same as everywhere else in the pipeline."""

    def fit(self, X: pd.DataFrame, y=None) -> "InteractionFeatures":
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["utilization_x_delinquency"] = (
            X["RevolvingUtilizationOfUnsecuredLines"] * X["total_delinquency"]
        )
        X["income_per_dependent"] = X["MonthlyIncome"] / (X["NumberOfDependents"] + 1)
        return X


def build_feature_pipeline() -> Pipeline:
    """Runs after preprocessing: aggregate delinquency signal, then build interactions on top."""
    return Pipeline(
        steps=[
            ("delinquency_aggregator", DelinquencyAggregator()),
            ("interaction_features", InteractionFeatures()),
        ]
    )
