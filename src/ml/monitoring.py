"""Population Stability Index (PSI) drift monitoring - the standard metric credit-risk
scorecards are monitored with in production: bin a reference distribution (train) into
deciles, then check what fraction of a current distribution (live predictions) falls into
each of those same bins. A shift shows up as bins that no longer hold ~10% each.

PSI = sum((current% - reference%) * ln(current% / reference%)) across bins. Widely-used
interpretation thresholds (industry convention, not derived here): < 0.1 -> stable,
0.1-0.25 -> moderate shift worth investigating, > 0.25 -> significant shift (retrain
candidate).

Two things get monitored, mirroring how scorecards are monitored in production:
- Score PSI: has a model's own OUTPUT distribution shifted? One reference per model - each
  of the 8 models has its own score distribution, scored on the same train data.
- Feature PSI: has the INPUT distribution shifted, independent of any model?

Both references are computed from train only, saved once by final_evaluation.py alongside
the model artifacts and the SHAP background sample. Comparing live predictions against them
here doesn't retrain or reselect anything - it's a read-only audit of an already-fixed
model, using the `predictions` table, not the test set - so it doesn't reopen the "test set
touched exactly once" question that governs evaluate_on_test().
"""

import json

import numpy as np
import pandas as pd

from .config import ARTIFACTS_DIR

REFERENCE_PATH = ARTIFACTS_DIR / "monitoring_reference.json"
N_BINS = 10
MIN_SAMPLE_SIZE = 30  # PSI on fewer points than this is noise, not signal

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

PSI_MODERATE_THRESHOLD = 0.1
PSI_SIGNIFICANT_THRESHOLD = 0.25


def _score_positive_proba(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Same predict_proba-or-sigmoid(decision_function) fallback as api/main.py's _score()
    - a separate, whole-DataFrame copy here since that one scores a single request row at a
    time and belongs to a different layer (api, not ml)."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    margin = pipeline.decision_function(X)
    return 1 / (1 + np.exp(-margin))


def _reference_bins(values) -> dict:
    """Decile edges of `values`, deduplicated - skewed/discrete columns (most delinquency
    counts are 0) collapse to fewer than N_BINS unique edges, and that's fine: PSI just runs
    with however many distinct bins the reference data actually supports."""
    values = pd.Series(values)
    quantiles = np.linspace(0, 1, N_BINS + 1)
    edges = np.unique(values.quantile(quantiles).to_numpy())
    edges[0], edges[-1] = -np.inf, np.inf
    binned = pd.cut(values, bins=edges, include_lowest=True)
    proportions = binned.value_counts(normalize=True, sort=False).to_numpy()
    return {"bin_edges": edges.tolist(), "reference_proportions": proportions.tolist()}


def save_reference_distributions(X_train: pd.DataFrame, pipelines: dict) -> None:
    """Called once from final_evaluation.py's __main__, alongside the model artifacts and
    the SHAP background sample - same train-only data, same "generated once, committed"
    pattern as everything else in artifacts/."""
    reference = {
        "features": {col: _reference_bins(X_train[col]) for col in RAW_FEATURES},
        "scores": {
            name: _reference_bins(_score_positive_proba(pipeline, X_train)) for name, pipeline in pipelines.items()
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REFERENCE_PATH, "w") as f:
        json.dump(reference, f, indent=2)


def load_reference_distributions() -> dict:
    with open(REFERENCE_PATH) as f:
        return json.load(f)


def compute_psi(bin_edges: list, reference_proportions: list, current_values) -> float:
    """Bins current_values into the reference distribution's own edges - current data has
    no say in defining the grid it's measured against, which is the whole point."""
    binned = pd.cut(pd.Series(current_values), bins=bin_edges, include_lowest=True)
    current_proportions = binned.value_counts(normalize=True, sort=False).to_numpy()

    reference = np.asarray(reference_proportions)
    # Epsilon smoothing: an empty bin under either distribution would otherwise make
    # ln(0) blow PSI up to infinity from one bin alone, hiding the signal from every other.
    epsilon = 1e-4
    actual = np.clip(current_proportions, epsilon, None)
    expected = np.clip(reference, epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def psi_status(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT_THRESHOLD:
        return "significant_shift"
    if psi >= PSI_MODERATE_THRESHOLD:
        return "moderate_shift"
    return "stable"
