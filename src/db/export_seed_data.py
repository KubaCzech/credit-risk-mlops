import json

import mlflow

from ml.evaluation.final_evaluation import MODEL_NAMES
from ml.tracking.mlflow_utils import configure_mlflow
from .seed_models import SEED_DATA_PATH


def export_seed_data() -> None:
    """Local dev only - refreshes seed_data.json from MLflow's tuned runs after retuning.
    Requires a live MLflow tracking store (mlflow.db), unlike seed_models.py itself."""
    configure_mlflow()

    data = []
    for name in MODEL_NAMES:
        runs = mlflow.search_runs(filter_string=f"tags.model_name = '{name}' and tags.stage = 'tuned'")
        if len(runs) == 0:
            raise ValueError(f"No tuned MLflow run found for '{name}' - run tune.py and final_evaluation.py first.")
        run = runs.iloc[0]
        data.append(
            {
                "name": name,
                "test_precision": run["metrics.test_precision"],
                "test_recall": run["metrics.test_recall"],
                "test_f1": run["metrics.test_f1"],
                "test_pr_auc": run["metrics.test_pr_auc"],
                "test_accuracy": run["metrics.test_accuracy"],
                "mlflow_run_id": run.run_id,
            }
        )

    with open(SEED_DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {len(data)} models to {SEED_DATA_PATH}")


if __name__ == "__main__":
    export_seed_data()
