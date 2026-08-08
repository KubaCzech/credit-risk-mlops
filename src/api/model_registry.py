import joblib

from db.models import Model
from db.session import SessionLocal
from ml.config import ARTIFACTS_DIR
from ml.evaluation.final_evaluation import MODEL_NAMES
from ml.models.mlp import TorchMLPClassifier  # noqa: F401 - needed to unpickle the MLP pipeline

DEFAULT_MODEL = "xgboost"  # best by test PR-AUC - see README Final Evaluation

_models: dict[str, object] = {}
_model_rows: dict[str, Model] = {}


def load_models() -> None:
    """Loads the fitted pipelines (joblib) and their DB metadata (models table) once at
    startup. Two separate stores because they come from two separate places: the pipeline
    is a file, the metrics/id are a database row seeded by db/seed_models.py."""
    for name in MODEL_NAMES:
        _models[name] = joblib.load(ARTIFACTS_DIR / f"{name}.joblib")

    with SessionLocal() as session:
        for row in session.query(Model).all():
            _model_rows[row.name] = row
            session.expunge(row)  # detach - safe to read after the session that loaded it closes


def get_model(name: str):
    return _models[name]


def get_model_row(name: str) -> Model:
    return _model_rows[name]


def available_models() -> list[str]:
    return list(_models.keys())
