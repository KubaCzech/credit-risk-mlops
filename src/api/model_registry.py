import joblib

from ml.config import ARTIFACTS_DIR
from ml.evaluation.final_evaluation import MODEL_NAMES
from ml.models.mlp import TorchMLPClassifier  # noqa: F401 - needed to unpickle the MLP pipeline

# From the Final Evaluation run (README, "Final Evaluation" section) - hardcoded rather than
# queried from MLflow at request time, so serving predictions doesn't depend on the tracking
# store being reachable. Re-copy these if final_evaluation.py is re-run with different data.
TEST_METRICS = {
    "xgboost": {"test_precision": 0.2170, "test_recall": 0.7870, "test_f1": 0.3402, "test_pr_auc": 0.4038, "test_accuracy": 0.7960},
    "random_forest": {"test_precision": 0.2325, "test_recall": 0.7536, "test_f1": 0.3554, "test_pr_auc": 0.4005, "test_accuracy": 0.8173},
    "mlp": {"test_precision": 0.2100, "test_recall": 0.7880, "test_f1": 0.3316, "test_pr_auc": 0.3975, "test_accuracy": 0.7877},
    "logistic_regression": {"test_precision": 0.2147, "test_recall": 0.7701, "test_f1": 0.3358, "test_pr_auc": 0.3933, "test_accuracy": 0.7964},
    "linear_svm": {"test_precision": 0.2016, "test_recall": 0.7880, "test_f1": 0.3211, "test_pr_auc": 0.3877, "test_accuracy": 0.7773},
    "decision_tree": {"test_precision": 0.2060, "test_recall": 0.7810, "test_f1": 0.3260, "test_pr_auc": 0.3705, "test_accuracy": 0.7842},
    "rbf_svm": {"test_precision": 0.2254, "test_recall": 0.6813, "test_f1": 0.3388, "test_pr_auc": 0.3693, "test_accuracy": 0.8223},
    "naive_bayes": {"test_precision": 0.3441, "test_recall": 0.5436, "test_f1": 0.4214, "test_pr_auc": 0.3554, "test_accuracy": 0.9002},
}

DEFAULT_MODEL = "xgboost"  # best by test PR-AUC - see README Final Evaluation

_models: dict[str, object] = {}


def load_models() -> None:
    for name in MODEL_NAMES:
        _models[name] = joblib.load(ARTIFACTS_DIR / f"{name}.joblib")


def get_model(name: str):
    return _models[name]


def available_models() -> list[str]:
    return list(_models.keys())
