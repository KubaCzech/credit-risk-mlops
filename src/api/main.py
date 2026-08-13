from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import select
from sqlalchemy.orm import Session

from db.models import Prediction
from db.session import get_db
from ml.explainability import explain
from ml.monitoring import MIN_SAMPLE_SIZE, compute_psi, psi_status

from .model_registry import (
    DEFAULT_MODEL,
    available_models,
    get_explainer,
    get_model,
    get_model_row,
    get_monitoring_reference,
    load_models,
)
from .schemas import (
    Contribution,
    CreditApplication,
    DriftResponse,
    ExplanationResponse,
    ModelInfo,
    ModelMetrics,
    ModelsResponse,
    PredictionRequest,
    PredictionResponse,
    PSIResult,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="Credit Risk Model Comparison API",
    description="Serves predictions from the 8 tuned models in credit-risk-mlops. See /models for the full comparison.",
    lifespan=lifespan,
)
Instrumentator().instrument(app).expose(app)  # GET /metrics, Prometheus text format

# CreditApplication's own Pydantic aliases (Kaggle column names) are the one source of
# truth for "raw feature name" <-> "DB/Python attribute name" - re-deriving this mapping
# here instead of hand-duplicating it means it can't drift out of sync with schemas.py.
# `age` has no explicit alias (its Kaggle name is already lowercase "age", same as the
# field name) - `field.alias or name` covers that case instead of silently dropping it.
_KAGGLE_NAME_TO_DB_ATTR = {(field.alias or name): name for name, field in CreditApplication.model_fields.items()}


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "models_loaded": available_models()}


@app.get("/models", response_model=ModelsResponse)
def list_models() -> ModelsResponse:
    return ModelsResponse(
        default_model=DEFAULT_MODEL,
        models=[
            ModelInfo(
                name=name,
                metrics=ModelMetrics(
                    test_precision=row.test_precision,
                    test_recall=row.test_recall,
                    test_f1=row.test_f1,
                    test_pr_auc=row.test_pr_auc,
                    test_accuracy=row.test_accuracy,
                ),
            )
            for name in available_models()
            for row in [get_model_row(name)]
        ],
    )


def _score(pipeline, X: pd.DataFrame) -> float:
    """LinearSVC and the RBF SVC (2 of the 8 models - both margin-based, no `predict_proba`)
    only expose `decision_function`: an unbounded distance from the separating hyperplane,
    not a probability. sigmoid() squashes it to (0, 1) for a consistent API response shape -
    an uncalibrated approximation, not a true probability (that would need e.g.
    CalibratedClassifierCV, which means retraining). Doesn't change classification behavior:
    sigmoid is monotonic, so thresholding sigmoid(margin) >= 0.5 is exactly equivalent to the
    model's own margin >= 0 boundary - only the exposed number's scale changes."""
    if hasattr(pipeline, "predict_proba"):
        return float(pipeline.predict_proba(X)[0, 1])
    margin = float(pipeline.decision_function(X)[0])
    return float(1 / (1 + np.exp(-margin)))


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    try:
        pipeline = get_model(request.model_name)
        model_row = get_model_row(request.model_name)
    except KeyError:
        # Deliberately replacing KeyError with a 422 here, not handling it in place - the
        # original traceback would only be noise for the caller, hence `from None`.
        raise HTTPException(
            status_code=422, detail=f"Unknown model '{request.model_name}'. See GET /models for valid names."
        ) from None

    application = request.application
    X = pd.DataFrame([application.model_dump(by_alias=True)])
    probability_of_default = _score(pipeline, X)
    prediction = probability_of_default >= 0.5

    # Logged synchronously, in the request path: if the audit log can't be written, the
    # request fails loudly rather than silently serving predictions nobody can account for.
    db.add(
        Prediction(
            model_id=model_row.id,
            **application.model_dump(by_alias=False),
            probability_of_default=probability_of_default,
            prediction=prediction,
        )
    )
    db.commit()

    return PredictionResponse(
        model_used=request.model_name,
        probability_of_default=probability_of_default,
        prediction=int(prediction),
    )


@app.post("/predict/explain", response_model=ExplanationResponse)
def predict_explain(request: PredictionRequest) -> ExplanationResponse:
    try:
        pipeline = get_model(request.model_name)
        explainer, feature_names, output_space = get_explainer(request.model_name)
    except KeyError:
        # Deliberately replacing KeyError with a 422 here, not handling it in place - the
        # original traceback would only be noise for the caller, hence `from None`.
        raise HTTPException(
            status_code=422, detail=f"Unknown model '{request.model_name}'. See GET /models for valid names."
        ) from None

    X = pd.DataFrame([request.application.model_dump(by_alias=True)])
    probability_of_default = _score(pipeline, X)
    base_value, shap_values = explain(pipeline, explainer, feature_names, X)

    contributions = sorted(
        (Contribution(feature=feature, shap_value=value) for feature, value in shap_values.items()),
        key=lambda c: -abs(c.shap_value),
    )

    return ExplanationResponse(
        model_used=request.model_name,
        probability_of_default=probability_of_default,
        output_space=output_space,
        base_value=base_value,
        contributions=contributions,
    )


@app.get("/monitoring/drift", response_model=DriftResponse)
def monitoring_drift(model_name: str = DEFAULT_MODEL, limit: int = 500, db: Session = Depends(get_db)) -> DriftResponse:
    """Population Stability Index (PSI) of this model's live predictions against its train
    reference - see ml/monitoring.py and README Monitoring section for the metric itself
    and why it's the standard tool for this in credit scoring specifically."""
    try:
        model_row = get_model_row(model_name)
    except KeyError:
        raise HTTPException(
            status_code=422, detail=f"Unknown model '{model_name}'. See GET /models for valid names."
        ) from None

    rows = (
        db.execute(
            select(Prediction)
            .where(Prediction.model_id == model_row.id)
            .order_by(Prediction.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    n_predictions = len(rows)
    if n_predictions < MIN_SAMPLE_SIZE:
        return DriftResponse(model_name=model_name, n_predictions=n_predictions, sufficient_sample=False)

    reference = get_monitoring_reference()

    score_ref = reference["scores"][model_name]
    scores = [row.probability_of_default for row in rows]
    score_psi_value = compute_psi(score_ref["bin_edges"], score_ref["reference_proportions"], scores)

    feature_psi = {}
    for kaggle_name, db_attr in _KAGGLE_NAME_TO_DB_ATTR.items():
        if kaggle_name not in reference["features"]:
            continue
        # monthly_income/number_of_dependents are nullable - PSI only makes sense over the
        # rows that actually reported a value.
        values = [v for row in rows if (v := getattr(row, db_attr)) is not None]
        if len(values) < MIN_SAMPLE_SIZE:
            continue
        feat_ref = reference["features"][kaggle_name]
        psi_value = compute_psi(feat_ref["bin_edges"], feat_ref["reference_proportions"], values)
        feature_psi[kaggle_name] = PSIResult(value=psi_value, status=psi_status(psi_value))

    return DriftResponse(
        model_name=model_name,
        n_predictions=n_predictions,
        sufficient_sample=True,
        score_psi=PSIResult(value=score_psi_value, status=psi_status(score_psi_value)),
        feature_psi=feature_psi,
    )


if __name__ == "__main__":
    # Local dev convenience: `python -m api.main`, matching the `python -m ml.xxx.yyy`
    # pattern used by every other entry point in this project, instead of a
    # separately-remembered uvicorn CLI invocation. reload=True needs the "module:app"
    # string form (not the `app` object) - uvicorn re-imports it fresh in a subprocess on
    # every file change. Production (Docker/K8s) will invoke uvicorn directly with
    # different flags (host 0.0.0.0, no reload), bypassing this block.
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
