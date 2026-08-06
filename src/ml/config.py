from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "cs-training.csv"
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"

INDEX_COL = 0
TARGET_COL = "SeriousDlqin2yrs"

RANDOM_STATE = 42
TEST_SIZE = 0.2

CV_FOLDS = 5
RBF_SVM_SUBSAMPLE_SIZE = 12_000

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT_NAME = "credit-risk-baseline"

OPTUNA_STORAGE = f"sqlite:///{PROJECT_ROOT / 'optuna.db'}"
OPTUNA_N_TRIALS_DEFAULT = 30
OPTUNA_N_TRIALS_OVERRIDES = {
    "naive_bayes": 15,  # single real hyperparameter (var_smoothing) - a 1D search doesn't need many trials
    "rbf_svm": 15,  # each trial is a full CV on top of an already-subsampled O(n^2-n^3) model
    "linear_svm": 15,  # penalty='l1' at high C converges slowly (~40s/trial vs ~1-15s for everything else)
    "mlp": 15,  # ~43s/trial (5-fold CV, each fold training epochs with early stopping)
}
