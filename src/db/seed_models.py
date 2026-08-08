import mlflow

from ml.evaluation.final_evaluation import MODEL_NAMES
from ml.tracking.mlflow_utils import configure_mlflow
from .models import Base, Model
from .session import SessionLocal, engine


def seed_models() -> None:
    """Populates the models table from MLflow's tuned runs - the same source final_evaluation.py
    reads, so this table and the hardcoded TEST_METRICS in api/model_registry.py should never
    disagree. Upserts by name, so re-running after retuning updates rather than duplicates."""
    Base.metadata.create_all(engine)
    configure_mlflow()

    with SessionLocal() as session:
        for name in MODEL_NAMES:
            runs = mlflow.search_runs(filter_string=f"tags.model_name = '{name}' and tags.stage = 'tuned'")
            if len(runs) == 0:
                raise ValueError(f"No tuned MLflow run found for '{name}' - run tune.py and final_evaluation.py first.")
            run = runs.iloc[0]

            existing = session.query(Model).filter_by(name=name).one_or_none()
            row = existing or Model(name=name)
            row.test_precision = run["metrics.test_precision"]
            row.test_recall = run["metrics.test_recall"]
            row.test_f1 = run["metrics.test_f1"]
            row.test_pr_auc = run["metrics.test_pr_auc"]
            row.test_accuracy = run["metrics.test_accuracy"]
            row.mlflow_run_id = run.run_id
            if existing is None:
                session.add(row)
            print(f"{name}: {'updated' if existing else 'inserted'}")

        session.commit()


if __name__ == "__main__":
    seed_models()
