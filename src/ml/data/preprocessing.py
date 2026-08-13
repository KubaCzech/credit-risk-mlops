import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

DELINQUENCY_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
SENTINEL_VALUES = (96, 98)
OUTLIER_COLS = ["DebtRatio", "RevolvingUtilizationOfUnsecuredLines", "MonthlyIncome"]
MIN_PLAUSIBLE_AGE = 18


class DelinquencySentinelHandler(BaseEstimator, TransformerMixin):
    """96/98 in the delinquency columns are error codes, not counts: flag as missing then impute."""

    def __init__(self, cols: list[str] | None = None, sentinel_values: tuple = SENTINEL_VALUES):
        self.cols = cols if cols is not None else DELINQUENCY_COLS
        self.sentinel_values = sentinel_values

    def fit(self, X: pd.DataFrame, y=None) -> "DelinquencySentinelHandler":
        is_sentinel = X[self.cols].isin(self.sentinel_values)
        valid_values = X[self.cols].where(~is_sentinel)
        self.medians_ = valid_values.median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        is_sentinel = X[self.cols].isin(self.sentinel_values)
        X["has_delinquency_sentinel"] = is_sentinel.any(axis=1).astype(int)
        for col in self.cols:
            X.loc[is_sentinel[col], col] = self.medians_[col]
        return X


class MonthlyIncomeImputer(BaseEstimator, TransformerMixin):
    """Median-impute MonthlyIncome; missingness itself may signal risk, so it's kept as a flag."""

    def __init__(self, col: str = "MonthlyIncome"):
        self.col = col

    def fit(self, X: pd.DataFrame, y=None) -> "MonthlyIncomeImputer":
        self.median_ = X[self.col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[f"{self.col}_was_missing"] = X[self.col].isna().astype(int)
        X[self.col] = X[self.col].fillna(self.median_)
        return X


class DependentsImputer(BaseEstimator, TransformerMixin):
    """Median-impute NumberOfDependents (2.6% missing, no flag - too low a rate to be informative)."""

    def __init__(self, col: str = "NumberOfDependents"):
        self.col = col

    def fit(self, X: pd.DataFrame, y=None) -> "DependentsImputer":
        self.median_ = X[self.col].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.col] = X[self.col].fillna(self.median_)
        return X


class AgeCleaner(BaseEstimator, TransformerMixin):
    """Clip age to a plausible adult minimum (dataset has a single age=0 row)."""

    def __init__(self, col: str = "age", min_age: int = MIN_PLAUSIBLE_AGE):
        self.col = col
        self.min_age = min_age

    def fit(self, X: pd.DataFrame, y=None) -> "AgeCleaner":
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.col] = X[self.col].clip(lower=self.min_age)
        return X


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Winsorize the upper tail of skewed columns at a quantile fitted on train only.

    Empirically beat every alternative tried (log1p, quantile binning + one-hot, leaving
    columns untreated) on Logistic Regression PR-AUC - see the feature engineering
    ablation in the project history. Capping only touches the extreme top 1%, leaving the
    well-behaved bulk of each distribution on its original, apparently already close to
    linear-in-log-odds, scale - log/binning reshape that bulk too and lose that."""

    def __init__(self, cols: list[str] | None = None, upper_quantile: float = 0.99):
        self.cols = cols if cols is not None else OUTLIER_COLS
        self.upper_quantile = upper_quantile

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierCapper":
        self.upper_bounds_ = X[self.cols].quantile(self.upper_quantile).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].clip(upper=self.upper_bounds_[col])
        return X


def build_preprocessing_pipeline() -> Pipeline:
    """Assemble the cleaning steps into one Pipeline: fit on train, transform on train+test."""
    return Pipeline(
        steps=[
            ("delinquency_sentinel", DelinquencySentinelHandler()),
            ("income_imputer", MonthlyIncomeImputer()),
            ("dependents_imputer", DependentsImputer()),
            ("age_cleaner", AgeCleaner()),
            ("outlier_capper", OutlierCapper()),
        ]
    )
