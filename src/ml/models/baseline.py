from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ..config import RANDOM_STATE
from ..data.preprocessing import build_preprocessing_pipeline
from ..features.engineering import build_feature_pipeline


def cleaning_and_feature_steps() -> list:
    return [*build_preprocessing_pipeline().steps, *build_feature_pipeline().steps]


def build_logistic_regression_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """LogReg is scale-sensitive: L2 regularization penalizes coefficients uniformly,
    so a feature on a 0-1 scale (Revolving) vs one on a 0-25000 scale (MonthlyIncome)
    would be penalized very unevenly without StandardScaler."""
    return Pipeline(
        steps=[
            *cleaning_and_feature_steps(),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)),
        ]
    )


def build_linear_svm_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """LinearSVC solves the primal problem (dual=False), which is faster than the dual
    formulation when n_samples >> n_features (120k rows vs ~12 features here) - the
    opposite regime from where kernel SVMs (which must use the dual) are needed."""
    return Pipeline(
        steps=[
            *cleaning_and_feature_steps(),
            ("scaler", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced", dual=False, max_iter=5000, random_state=random_state)),
        ]
    )


def build_rbf_svm_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """Kernel SVMs only have a dual formulation, whose training cost grows roughly
    O(n_samples^2 - n_samples^3) - intractable on the full 120k-row train set, hence
    this is always fit on a subsample (see RBF_SVM_SUBSAMPLE_SIZE in config.py)."""
    return Pipeline(
        steps=[
            *cleaning_and_feature_steps(),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=random_state)),
        ]
    )


def build_random_forest_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """No StandardScaler: tree splits threshold one feature at a time, so a monotonic
    rescaling can't change which split points get chosen - RF is scale-invariant."""
    return Pipeline(
        steps=[
            *cleaning_and_feature_steps(),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def build_decision_tree_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """A single tree, same scale-invariance reasoning as Random Forest (no scaler needed).
    Included specifically to contrast against Random Forest: same base learner, but without
    the ensemble's variance reduction - a concrete before/after story for why ensembling
    helps, not just another competing model."""
    return Pipeline(
        steps=[
            *cleaning_and_feature_steps(),
            ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=random_state)),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight: float, random_state: int = RANDOM_STATE) -> Pipeline:
    """Gradient-boosted trees - the closest thing to an industry default for tabular credit
    data. No StandardScaler for the same reason as RF/DecisionTree (tree splits). XGBoost has
    no class_weight param; scale_pos_weight (ratio of negative to positive samples) is its
    equivalent and must be computed from the actual training labels, not hardcoded, since it
    depends on exactly which rows land in train after the split."""
    return Pipeline(
        steps=[
            *cleaning_and_feature_steps(),
            (
                "clf",
                XGBClassifier(
                    n_estimators=300,
                    scale_pos_weight=scale_pos_weight,
                    random_state=random_state,
                    n_jobs=-1,
                    eval_metric="logloss",
                ),
            ),
        ]
    )


def build_naive_bayes_pipeline() -> Pipeline:
    """GaussianNB fits a per-class, per-feature mean/variance independently, so it's
    invariant to per-feature affine rescaling (shifting/scaling a feature shifts the fitted
    Gaussian the same way, cancelling out in the likelihood ratio) - no scaler needed, for a
    different reason than the tree models. priors=[0.5, 0.5] is the NB analogue of
    class_weight="balanced": without it, NB would use the empirical 93.3/6.7 class split as
    its prior, which is exactly the imbalance we're correcting for everywhere else."""
    return Pipeline(
        steps=[
            *cleaning_and_feature_steps(),
            ("clf", GaussianNB(priors=[0.5, 0.5])),
        ]
    )
