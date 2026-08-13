"""Post-hoc fairness audit of the already-finalized models, by age - the one
protected-characteristic-adjacent column in this dataset (GiveMeSomeCredit has no
race/gender/ethnicity fields; that's a real limitation of this analysis, not hidden here).

Age specifically because it's a real protected characteristic in US lending: the Equal
Credit Opportunity Act (ECOA) explicitly bars denying credit or offering worse terms
because an applicant is 62 or older. AGE_PROTECTED_THRESHOLD below follows that line.

This is a SECOND, deliberate read of the test set, after evaluate_on_test() - worth being
explicit about why that doesn't reopen the "touched exactly once" discipline the rest of
this project follows. That rule exists to stop the test set from leaking into modeling
decisions (which model, which hyperparameters, which features) through repeated peeking.
A fairness audit runs downstream of models that are already fully fixed: nothing here
feeds back into training, tuning, or model selection. It's an audit of a finished
artifact, not another round of model development.

prediction=1 here means "predicted to default", i.e. flagged as high-risk / denied credit
- an ADVERSE outcome, the opposite of the usual EEOC/ECOA framing where "selected" is a
good thing (hired, approved). approval_rate = 1 - selection_rate is used everywhere below
so "disparate impact" keeps its conventional meaning: are protected-group applicants
approved at a lower rate than everyone else?
"""

import joblib
import numpy as np
import pandas as pd

from ..config import ARTIFACTS_DIR
from ..data.loader import load_and_split, split_X_y
from ..models.mlp import TorchMLPClassifier  # noqa: F401 - needed to unpickle MLP pipelines
from .final_evaluation import MODEL_NAMES

AGE_PROTECTED_THRESHOLD = 62  # ECOA: creditors may not discriminate based on age 62+

AGE_BINS = [18, 25, 40, 62, 200]
AGE_BIN_LABELS = ["18-24", "25-39", "40-61", "62+"]

FOUR_FIFTHS_RULE = 0.8  # EEOC convention: a disparate impact ratio below this is a flag


def _group_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    actual_negative = y_true == 0
    actual_positive = y_true == 1
    return {
        "n": len(y_true),
        "actual_default_rate": float(y_true.mean()),
        "approval_rate": float((y_pred == 0).mean()),
        "fpr": float((y_pred[actual_negative] == 1).mean()),  # wrongly denied - didn't actually default
        "fnr": float((y_pred[actual_positive] == 0).mean()),  # wrongly approved - did actually default
    }


def fairness_by_age_bin(model_name: str, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Fine-grained breakdown for one model: 4 age bins, not just the ECOA-protected split."""
    pipeline = joblib.load(ARTIFACTS_DIR / f"{model_name}.joblib")
    y_pred = pipeline.predict(X_test)
    age_bin = pd.cut(X_test["age"], bins=AGE_BINS, labels=AGE_BIN_LABELS, right=False)

    rows = []
    for label in AGE_BIN_LABELS:
        mask = (age_bin == label).to_numpy()
        rows.append({"age_group": label, **_group_metrics(y_test[mask], y_pred[mask])})
    return pd.DataFrame(rows).set_index("age_group")


def fairness_summary(model_names: list[str] = MODEL_NAMES) -> pd.DataFrame:
    """One row per model: protected (62+) vs reference (<62) disparate impact ratio and
    FPR/FNR gaps - the headline ECOA-relevant numbers, across all 8 models at once."""
    _train_df, test_df = load_and_split()
    X_test, y_test = split_X_y(test_df)
    protected = (X_test["age"] >= AGE_PROTECTED_THRESHOLD).to_numpy()

    rows = []
    for model_name in model_names:
        pipeline = joblib.load(ARTIFACTS_DIR / f"{model_name}.joblib")
        y_pred = pipeline.predict(X_test)

        prot = _group_metrics(y_test[protected], y_pred[protected])
        ref = _group_metrics(y_test[~protected], y_pred[~protected])
        disparate_impact_ratio = prot["approval_rate"] / ref["approval_rate"]

        rows.append(
            {
                "model": model_name,
                "approval_rate_62plus": prot["approval_rate"],
                "approval_rate_under_62": ref["approval_rate"],
                "disparate_impact_ratio": disparate_impact_ratio,
                "fpr_62plus": prot["fpr"],
                "fpr_under_62": ref["fpr"],
                "fpr_gap": prot["fpr"] - ref["fpr"],
                "fnr_gap": prot["fnr"] - ref["fnr"],
                "flagged_four_fifths_rule": disparate_impact_ratio < FOUR_FIFTHS_RULE,
            }
        )

    return pd.DataFrame(rows).set_index("model")


if __name__ == "__main__":
    pd.set_option("display.width", 160)

    print("Disparate impact summary (protected: age >= 62, ECOA), all 8 models:")
    print(fairness_summary().round(4))

    print()
    print("XGBoost (default model), full age-bin breakdown:")
    _train_df, test_df = load_and_split()
    X_test, y_test = split_X_y(test_df)
    print(fairness_by_age_bin("xgboost", X_test, y_test).round(4))
