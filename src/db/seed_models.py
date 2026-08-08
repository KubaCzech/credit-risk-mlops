import json
from pathlib import Path

from .models import Base, Model
from .session import SessionLocal, engine

SEED_DATA_PATH = Path(__file__).resolve().parent / "seed_data.json"


def seed_models() -> None:
    """Populates the models table from a static JSON snapshot (seed_data.json), not a live
    MLflow query. That was the original design, and it broke inside Docker: mlflow.db lives
    only on the host machine, so a fresh container always saw an empty tracking store and
    failed to seed anything. This is the fix - seeding no longer depends on MLflow being
    reachable at all, in a container or otherwise. Run db/export_seed_data.py to refresh the
    snapshot after retuning. Upserts by name, so re-running is safe."""
    Base.metadata.create_all(engine)

    with open(SEED_DATA_PATH) as f:
        seed_data = json.load(f)

    with SessionLocal() as session:
        for entry in seed_data:
            name = entry["name"]
            existing = session.query(Model).filter_by(name=name).one_or_none()
            row = existing or Model(name=name)
            row.test_precision = entry["test_precision"]
            row.test_recall = entry["test_recall"]
            row.test_f1 = entry["test_f1"]
            row.test_pr_auc = entry["test_pr_auc"]
            row.test_accuracy = entry["test_accuracy"]
            row.mlflow_run_id = entry["mlflow_run_id"]
            if existing is None:
                session.add(row)
            print(f"{name}: {'updated' if existing else 'inserted'}")

        session.commit()


if __name__ == "__main__":
    seed_models()
