from typing import cast

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score

from ..config import ARTIFACTS_DIR
from ..data.loader import load_and_split, split_X_y
from ..explainability import save_shap_background
from ..models.mlp import TorchMLPClassifier  # noqa: F401 - needed to unpickle MLP pipelines
from ..monitoring import save_reference_distributions
from ..tracking.mlflow_utils import configure_mlflow

MODEL_NAMES = [
    "logistic_regression",
    "linear_svm",
    "rbf_svm",
    "random_forest",
    "decision_tree",
    "xgboost",
    "naive_bayes",
    "mlp",
]


def _find_tuned_run(model_name: str) -> pd.Series:
    # output_format="pandas" is mlflow's own default, made explicit here because
    # search_runs()'s other output_format returns a plain list, not a DataFrame - the
    # .iloc[0] below only works because of this.
    runs = cast(
        pd.DataFrame,
        mlflow.search_runs(
            filter_string=f"tags.model_name = '{model_name}' and tags.stage = 'tuned'", output_format="pandas"
        ),
    )
    if len(runs) == 0:
        raise ValueError(f"No tuned MLflow run found for '{model_name}' - run tune.py first.")
    return runs.iloc[0]


def save_tuned_artifacts(model_names: list[str] = MODEL_NAMES) -> dict[str, str]:
    """Pulls each model's already-fitted pipeline (fit once during tuning, on train only -
    see tune.py) out of MLflow and saves it as a standalone joblib file. Not a retrain: the
    exact same fitted object that was CV-evaluated is what gets persisted and later
    evaluated on test, so there's no discrepancy from e.g. a different random init."""
    configure_mlflow()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}
    for model_name in model_names:
        run = _find_tuned_run(model_name)
        pipeline = mlflow.sklearn.load_model(f"runs:/{run.run_id}/model")
        path = ARTIFACTS_DIR / f"{model_name}.joblib"
        joblib.dump(pipeline, path)
        paths[model_name] = str(path)
        print(f"{model_name}: saved to {path}")
    return paths


def evaluate_on_test(model_names: list[str] = MODEL_NAMES) -> pd.DataFrame:
    """The only place in this project that touches the test set. Loads each artifact from
    disk, scores it once on the held-out 30k rows, and logs the result back onto the same
    MLflow run that already holds that model's CV metrics - so CV and test numbers for a
    given model live together, not in separate disconnected places."""
    configure_mlflow()
    _train_df, test_df = load_and_split()
    X_test, y_test = split_X_y(test_df)

    dumb_baseline_accuracy = (y_test == 0).mean()
    print(f"'always predict 0' baseline accuracy on test: {dumb_baseline_accuracy:.4f}")

    rows = []
    for model_name in model_names:
        path = ARTIFACTS_DIR / f"{model_name}.joblib"
        pipeline = joblib.load(path)

        y_pred = pipeline.predict(X_test)
        y_score = (
            pipeline.predict_proba(X_test)[:, 1]
            if hasattr(pipeline, "predict_proba")
            else pipeline.decision_function(X_test)
        )
        test_metrics = {
            "test_precision": precision_score(y_test, y_pred),
            "test_recall": recall_score(y_test, y_pred),
            "test_f1": f1_score(y_test, y_pred),
            "test_pr_auc": average_precision_score(y_test, y_score),
            # tracked for the record, never for ranking - see README "Why not accuracy"
            "test_accuracy": accuracy_score(y_test, y_pred),
        }

        run = _find_tuned_run(model_name)
        cv_pr_auc = run.get("metrics.pr_auc_mean", float("nan"))
        with mlflow.start_run(run_id=run.run_id):
            mlflow.log_metrics(test_metrics)

        rows.append({"model": model_name, "cv_pr_auc": cv_pr_auc, **test_metrics})
        print(f"{model_name}: test_pr_auc={test_metrics['test_pr_auc']:.4f} (cv was {cv_pr_auc:.4f})")

    results = pd.DataFrame(rows).set_index("model")
    results["cv_test_gap"] = results["cv_pr_auc"] - results["test_pr_auc"]
    return results.sort_values("test_pr_auc", ascending=False)


if __name__ == "__main__":
    save_tuned_artifacts()

    train_df, _test_df = load_and_split()
    X_train, _y_train = split_X_y(train_df)
    save_shap_background(X_train)
    print("SHAP background sample saved (from train only)")

    pipelines = {name: joblib.load(ARTIFACTS_DIR / f"{name}.joblib") for name in MODEL_NAMES}
    save_reference_distributions(X_train, pipelines)
    print("Monitoring reference distributions saved (from train only)")

    print()
    results_df = evaluate_on_test()
    pd.set_option("display.width", 140)
    print()
    print(results_df.round(4))
