import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline

from ..config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_baseline_run(
    model_name: str,
    pipeline: Pipeline,
    cv_summary: dict,
    n_train_rows: int,
    feature_set: str = "raw",
    stage: str = "baseline",
) -> None:
    clf_params = pipeline.named_steps["clf"].get_params()
    with mlflow.start_run(run_name=model_name):
        mlflow.set_tag("model_name", model_name)
        mlflow.set_tag("feature_set", feature_set)
        mlflow.set_tag("stage", stage)
        mlflow.log_param("n_train_rows", n_train_rows)
        mlflow.log_params({k: v for k, v in clf_params.items() if isinstance(v, (int, float, str, bool)) or v is None})
        mlflow.log_metrics(cv_summary)
        mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")
