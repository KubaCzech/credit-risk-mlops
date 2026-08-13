"""SHAP-based explainability for the 8 tuned model pipelines.

Three explainer families, chosen empirically (see README, Explainability section) rather
than one generic approach - a single algorithm would either be slow for models that have a
fast, exact alternative, or silently wrong for models whose native output isn't a
probability:

- Linear (LogReg, Linear SVM): shap.LinearExplainer, exact and near-instant. Output is in
  log-odds space (the model's own decision_function), not probability - SHAP's additivity
  guarantee (attributions + base value == model output) only holds in whatever space the
  model natively produces; converting per-feature values to probability afterward would
  break it, since sigmoid is nonlinear and additivity doesn't survive a nonlinear transform
  applied per-feature.
- Trees (Random Forest, Decision Tree, XGBoost): shap.TreeExplainer with
  feature_perturbation="tree_path_dependent" - exact, and needs no background dataset at
  all (it reads conditional expectations directly off the tree structure). Random Forest
  and Decision Tree output directly in probability space (each leaf's value already is a
  class-vote fraction); XGBoost outputs in log-odds space (it boosts an additive log-odds
  score, not per-tree probabilities) - same algorithm, different native output space per
  model, confirmed empirically rather than assumed (see scratch experiments run before
  writing this module).
- Everything else (Naive Bayes, RBF SVM, MLP): no closed-form SHAP algorithm exists for
  these model families, so they fall back to shap.Explainer's generic Permutation
  algorithm against a small background sample (SHAP_BACKGROUND_SIZE rows, sampled from
  train only - see save_shap_background). Naive Bayes and MLP explain predict_proba
  (probability space); RBF SVM has no predict_proba (see api/main.py's _score - same
  root cause), so it explains decision_function instead (uncalibrated margin space, exactly
  like /predict's fallback for the same model). RBF SVM is also the slow case here: cost
  scales roughly linearly with background size (~0.1s/row, empirically measured) because
  every permutation re-evaluates the RBF kernel against every support vector - a real,
  inherent cost of that model family, not an engineering gap, hence keeping the background
  small. Naive Bayes pays a one-time ~3s numba JIT compilation cost on the first call in a
  process, not per request - warm_up() pays that cost at API startup instead of on the
  first real user request.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from .config import ARTIFACTS_DIR, RANDOM_STATE

SHAP_BACKGROUND_PATH = ARTIFACTS_DIR / "shap_background.csv"
SHAP_BACKGROUND_SIZE = 20

TREE_MODELS = {"random_forest", "decision_tree", "xgboost"}
LOG_ODDS_TREE_MODELS = {"xgboost"}  # boosts an additive log-odds score, not vote fractions
LINEAR_MODELS = {"logistic_regression", "linear_svm"}


def save_shap_background(train_X: pd.DataFrame, n: int = SHAP_BACKGROUND_SIZE) -> None:
    """Samples from train only, never test - final_evaluation.py is the only place this
    project touches the test set, and this must not become a second one."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    train_X.sample(n, random_state=RANDOM_STATE).to_csv(SHAP_BACKGROUND_PATH, index=False)


def load_shap_background() -> pd.DataFrame:
    return pd.read_csv(SHAP_BACKGROUND_PATH)


def _feature_names(pipeline: Pipeline, X_sample: pd.DataFrame) -> list[str]:
    """Cleaning/feature-engineering steps are all DataFrame-in, DataFrame-out, so their
    output columns are the human-readable feature names SHAP should report against - the
    scaler step, if present, is the last thing before "clf" and turns that DataFrame into
    an unnamed ndarray, so it must be excluded from this slice too."""
    has_scaler = pipeline.steps[-2][0] == "scaler"
    pre = pipeline[:-2] if has_scaler else pipeline[:-1]
    return list(pre.transform(X_sample).columns)


def _score_fn(estimator):
    if hasattr(estimator, "predict_proba"):
        return lambda arr: estimator.predict_proba(arr)[:, 1]
    return lambda arr: estimator.decision_function(arr)


def build_explainer(model_name: str, pipeline: Pipeline, background: pd.DataFrame) -> tuple:
    """Returns (explainer, feature_names, output_space). output_space is one of
    "probability", "log_odds", "margin" - see module docstring for which model gets which."""
    estimator = pipeline[-1]
    feature_names = _feature_names(pipeline, background)

    if model_name in TREE_MODELS:
        explainer = shap.TreeExplainer(estimator, feature_perturbation="tree_path_dependent")
        output_space = "log_odds" if model_name in LOG_ODDS_TREE_MODELS else "probability"
        return explainer, feature_names, output_space

    background_t = pipeline[:-1].transform(background)

    if model_name in LINEAR_MODELS:
        explainer = shap.Explainer(estimator, background_t)
        return explainer, feature_names, "log_odds"

    fn = _score_fn(estimator)
    explainer = shap.Explainer(fn, background_t)
    output_space = "probability" if hasattr(estimator, "predict_proba") else "margin"
    return explainer, feature_names, output_space


def explain(pipeline: Pipeline, explainer, feature_names: list[str], X: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """Returns (base_value, {feature_name: shap_value}) for a single-row X, in whatever
    output_space build_explainer() reported for this model."""
    X_t = pipeline[:-1].transform(X)
    sv = explainer(X_t)
    values = np.asarray(sv.values[0])
    base = sv.base_values[0]
    if values.ndim == 2:  # per-class output (random_forest, decision_tree) - keep class 1
        values = values[:, 1]
        base = base[1]
    return float(base), dict(zip(feature_names, (float(v) for v in values), strict=True))


def warm_up(pipeline: Pipeline, explainer, feature_names: list[str], background: pd.DataFrame) -> None:
    """Pays any one-time cost (numba JIT compilation, mainly - see module docstring) at
    startup instead of on the first real request."""
    explain(pipeline, explainer, feature_names, background.iloc[[0]])
