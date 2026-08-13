import joblib

from db.models import Model
from db.session import SessionLocal
from ml.config import ARTIFACTS_DIR
from ml.evaluation.final_evaluation import MODEL_NAMES
from ml.explainability import build_explainer, load_shap_background, warm_up
from ml.models.mlp import TorchMLPClassifier  # noqa: F401 - needed to unpickle the MLP pipeline
from ml.monitoring import load_reference_distributions

DEFAULT_MODEL = "xgboost"  # best by test PR-AUC - see README Final Evaluation

_models: dict[str, object] = {}
_model_rows: dict[str, Model] = {}
_explainers: dict[str, tuple] = {}  # name -> (explainer, feature_names, output_space)
_monitoring_reference: dict = {}


def load_models() -> None:
    """Loads the fitted pipelines (joblib) and their DB metadata (models table) once at
    startup. Two separate stores because they come from two separate places: the pipeline
    is a file, the metrics/id are a database row seeded by db/seed_models.py.

    Also builds and warms up a SHAP explainer per model here, not lazily on first request -
    one of them (naive_bayes) pays a one-time ~3s numba JIT cost on its first call in a
    process; better to pay that at startup than make the first real user eat it."""
    for name in MODEL_NAMES:
        _models[name] = joblib.load(ARTIFACTS_DIR / f"{name}.joblib")

    with SessionLocal() as session:
        for row in session.query(Model).all():
            _model_rows[row.name] = row
            session.expunge(row)  # detach - safe to read after the session that loaded it closes

    background = load_shap_background()
    for name in MODEL_NAMES:
        explainer, feature_names, output_space = build_explainer(name, _models[name], background)
        warm_up(_models[name], explainer, feature_names, background)
        _explainers[name] = (explainer, feature_names, output_space)

    _monitoring_reference.update(load_reference_distributions())


def get_model(name: str):
    return _models[name]


def get_model_row(name: str) -> Model:
    return _model_rows[name]


def get_explainer(name: str) -> tuple:
    return _explainers[name]


def get_monitoring_reference() -> dict:
    return _monitoring_reference


def available_models() -> list[str]:
    return list(_models.keys())
