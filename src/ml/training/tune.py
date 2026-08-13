import time

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..config import (
    CV_FOLDS,
    OPTUNA_N_TRIALS_DEFAULT,
    OPTUNA_N_TRIALS_OVERRIDES,
    OPTUNA_STORAGE,
    RANDOM_STATE,
)
from ..data.loader import load_and_split, split_X_y, subsample_for_rbf
from ..evaluation.metrics import cross_validate_pipeline, summarize_cv_results
from ..models.baseline import cleaning_and_feature_steps
from ..models.mlp import TorchMLPClassifier
from ..tracking.mlflow_utils import configure_mlflow, log_baseline_run
from .train_baseline import FEATURE_SET

# --- per-fold CV with pruning -------------------------------------------------


def _decision_scores(fitted_pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    clf = fitted_pipeline.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return fitted_pipeline.predict_proba(X)[:, 1]
    return fitted_pipeline.decision_function(X)


def cv_pr_auc_with_pruning(
    build_pipeline_fn,
    X: pd.DataFrame,
    y: pd.Series,
    trial: optuna.Trial,
    cv_folds: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Same StratifiedKFold CV as the baseline, done manually (not via cross_validate) so
    each fold's running mean can be reported to Optuna - a clearly bad trial gets pruned
    after a couple of folds instead of wasting the remaining 3."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        pipeline = build_pipeline_fn()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_score = _decision_scores(pipeline, X.iloc[val_idx])
        scores.append(average_precision_score(y.iloc[val_idx], y_score))
        trial.report(float(np.mean(scores)), step=fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(np.mean(scores)), float(np.std(scores))


# --- search spaces + pipeline builders, one pair per model --------------------


def suggest_logistic_regression(trial: optuna.Trial) -> dict:
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
    params = {"C": trial.suggest_float("C", 1e-3, 1e2, log=True), "penalty": penalty}
    if penalty == "elasticnet":
        params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
    return params


def build_logistic_regression_trial_pipeline(params: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    clf_kwargs = dict(
        C=params["C"],
        penalty=params["penalty"],
        solver="saga",  # the only solver supporting l1 and elasticnet, not just l2
        class_weight="balanced",
        max_iter=3000,
        random_state=random_state,
    )
    if params["penalty"] == "elasticnet":
        clf_kwargs["l1_ratio"] = params["l1_ratio"]
    return Pipeline(
        [*cleaning_and_feature_steps(), ("scaler", StandardScaler()), ("clf", LogisticRegression(**clf_kwargs))]
    )


def suggest_linear_svm(trial: optuna.Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
    }


def build_linear_svm_trial_pipeline(params: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            *cleaning_and_feature_steps(),
            ("scaler", StandardScaler()),
            (
                "clf",
                LinearSVC(
                    C=params["C"],
                    penalty=params["penalty"],
                    loss="squared_hinge",
                    dual=False,
                    class_weight="balanced",
                    max_iter=2000,  # l1 penalty at high C converges slowly; capped lower so a
                    # hard-to-converge trial just scores mildly worse instead of dominating wall time
                    random_state=random_state,
                ),
            ),
        ]
    )


def suggest_rbf_svm(trial: optuna.Trial) -> dict:
    return {
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
        "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True),
    }


def build_rbf_svm_trial_pipeline(params: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            *cleaning_and_feature_steps(),
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=params["C"],
                    gamma=params["gamma"],
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def suggest_random_forest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
    }


def build_random_forest_trial_pipeline(params: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            *cleaning_and_feature_steps(),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=params["n_estimators"],
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    max_features=params["max_features"],
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def suggest_decision_tree(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
        "ccp_alpha": trial.suggest_float("ccp_alpha", 1e-5, 1e-1, log=True),
    }


def build_decision_tree_trial_pipeline(params: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            *cleaning_and_feature_steps(),
            (
                "clf",
                DecisionTreeClassifier(
                    max_depth=params["max_depth"],
                    min_samples_leaf=params["min_samples_leaf"],
                    min_samples_split=params["min_samples_split"],
                    ccp_alpha=params["ccp_alpha"],
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def suggest_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
    }


def build_xgboost_trial_pipeline(params: dict, scale_pos_weight: float, random_state: int = RANDOM_STATE) -> Pipeline:
    return Pipeline(
        [
            *cleaning_and_feature_steps(),
            (
                "clf",
                XGBClassifier(
                    **params,
                    scale_pos_weight=scale_pos_weight,  # fixed, not tuned - keep the imbalance correction constant
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def suggest_mlp(trial: optuna.Trial) -> dict:
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
    }


def build_mlp_trial_pipeline(params: dict, random_state: int = RANDOM_STATE) -> Pipeline:
    """Architecture (64->32 hidden units) stays fixed - only training hyperparameters are
    searched. A full neural architecture search is a different-scale undertaking than
    tuning the other 6 models and isn't the point of this step."""
    return Pipeline(
        [
            *cleaning_and_feature_steps(),
            ("scaler", StandardScaler()),
            (
                "clf",
                TorchMLPClassifier(
                    lr=params["lr"],
                    dropout=params["dropout"],
                    weight_decay=params["weight_decay"],
                    batch_size=params["batch_size"],
                    random_state=random_state,
                ),
            ),
        ]
    )


def suggest_naive_bayes(trial: optuna.Trial) -> dict:
    return {"var_smoothing": trial.suggest_float("var_smoothing", 1e-12, 1e-6, log=True)}


def build_naive_bayes_trial_pipeline(params: dict) -> Pipeline:
    return Pipeline(
        [*cleaning_and_feature_steps(), ("clf", GaussianNB(priors=[0.5, 0.5], var_smoothing=params["var_smoothing"]))]
    )


def build_model_registry(scale_pos_weight: float) -> dict:
    return {
        "logistic_regression": (suggest_logistic_regression, build_logistic_regression_trial_pipeline),
        "linear_svm": (suggest_linear_svm, build_linear_svm_trial_pipeline),
        "random_forest": (suggest_random_forest, build_random_forest_trial_pipeline),
        "decision_tree": (suggest_decision_tree, build_decision_tree_trial_pipeline),
        "naive_bayes": (suggest_naive_bayes, build_naive_bayes_trial_pipeline),
        "xgboost": (suggest_xgboost, lambda params: build_xgboost_trial_pipeline(params, scale_pos_weight)),
        "rbf_svm": (suggest_rbf_svm, build_rbf_svm_trial_pipeline),
        "mlp": (suggest_mlp, build_mlp_trial_pipeline),
    }


# --- orchestration --------------------------------------------------------------


def tune_model(model_name: str, suggest_fn, build_fn, X: pd.DataFrame, y: pd.Series, n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        params = suggest_fn(trial)
        pr_auc_mean, pr_auc_std = cv_pr_auc_with_pruning(lambda: build_fn(params), X, y, trial)
        trial.set_user_attr("pr_auc_std", pr_auc_std)
        return pr_auc_mean

    study = optuna.create_study(
        study_name=f"{model_name}_{FEATURE_SET}",
        storage=OPTUNA_STORAGE,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def run_tuning_experiments(model_names: list[str] | None = None) -> pd.DataFrame:
    """model_names restricts which studies run - useful to tune a newly added model without
    re-optimizing (and appending redundant trials to) studies that already completed and
    are cached in optuna.db (load_if_exists=True means re-running would add more trials on
    top, not restart cleanly)."""
    train_df, _test_df = load_and_split()
    X_train, y_train = split_X_y(train_df)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    configure_mlflow()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    registry = build_model_registry(scale_pos_weight)
    if model_names is not None:
        registry = {name: registry[name] for name in model_names}

    rows = []
    for model_name, (suggest_fn, build_fn) in registry.items():
        X_model, y_model = subsample_for_rbf(X_train, y_train) if model_name == "rbf_svm" else (X_train, y_train)
        n_trials = OPTUNA_N_TRIALS_OVERRIDES.get(model_name, OPTUNA_N_TRIALS_DEFAULT)

        t0 = time.time()
        study = tune_model(model_name, suggest_fn, build_fn, X_model, y_model, n_trials)
        elapsed = time.time() - t0

        best_pipeline = build_fn(study.best_params)
        cv_results = cross_validate_pipeline(best_pipeline, X_model, y_model)
        cv_summary = summarize_cv_results(cv_results)
        best_pipeline.fit(X_model, y_model)
        log_baseline_run(
            model_name, best_pipeline, cv_summary, n_train_rows=len(X_model), feature_set=FEATURE_SET, stage="tuned"
        )

        n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
        rows.append(
            {
                "model": model_name,
                "n_trials": n_trials,
                "n_pruned": n_pruned,
                "elapsed_s": round(elapsed, 1),
                **cv_summary,
            }
        )
        pr_auc = cv_summary["pr_auc_mean"]
        print(f"{model_name}: pr_auc={pr_auc:.4f} ({n_trials} trials, {n_pruned} pruned, {elapsed:.0f}s)")

    results = pd.DataFrame(rows).set_index("model").sort_values("pr_auc_mean", ascending=False)
    return results


if __name__ == "__main__":
    results_df = run_tuning_experiments()
    pd.set_option("display.width", 140)
    print()
    print(results_df)
