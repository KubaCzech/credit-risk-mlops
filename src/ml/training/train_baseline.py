import pandas as pd

from ..data.loader import load_and_split, split_X_y, subsample_for_rbf
from ..evaluation.metrics import cross_validate_pipeline, summarize_cv_results
from ..models.baseline import (
    build_decision_tree_pipeline,
    build_linear_svm_pipeline,
    build_logistic_regression_pipeline,
    build_naive_bayes_pipeline,
    build_random_forest_pipeline,
    build_rbf_svm_pipeline,
    build_xgboost_pipeline,
)
from ..models.mlp import build_mlp_pipeline
from ..tracking.mlflow_utils import configure_mlflow, log_baseline_run

FEATURE_SET = "engineered_v1"

FULL_DATA_MODELS = {
    "logistic_regression": build_logistic_regression_pipeline,
    "linear_svm": build_linear_svm_pipeline,
    "random_forest": build_random_forest_pipeline,
    "decision_tree": build_decision_tree_pipeline,
    "naive_bayes": build_naive_bayes_pipeline,
    "mlp": build_mlp_pipeline,
}


def _run_one(model_name: str, pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    cv_results = cross_validate_pipeline(pipeline, X, y)
    cv_summary = summarize_cv_results(cv_results)
    pipeline.fit(X, y)
    log_baseline_run(model_name, pipeline, cv_summary, n_train_rows=len(X), feature_set=FEATURE_SET)
    print(f"{model_name}: done")
    return {"model": model_name, **cv_summary}


def run_baseline_experiments() -> pd.DataFrame:
    train_df, _test_df = load_and_split()
    X_train, y_train = split_X_y(train_df)

    configure_mlflow()

    rows = []
    for model_name, build_fn in FULL_DATA_MODELS.items():
        rows.append(_run_one(model_name, build_fn(), X_train, y_train))

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    rows.append(_run_one("xgboost", build_xgboost_pipeline(scale_pos_weight), X_train, y_train))

    X_sub, y_sub = subsample_for_rbf(X_train, y_train)
    rows.append(_run_one("rbf_svm", build_rbf_svm_pipeline(), X_sub, y_sub))

    results = pd.DataFrame(rows).set_index("model").sort_values("pr_auc_mean", ascending=False)
    return results


if __name__ == "__main__":
    results_df = run_baseline_experiments()
    pd.set_option("display.width", 120)
    print()
    print(results_df)
