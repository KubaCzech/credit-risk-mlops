from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from db.models import Prediction
from db.session import get_db

from .model_registry import DEFAULT_MODEL, available_models, get_model, get_model_row, load_models
from .schemas import ModelInfo, ModelMetrics, ModelsResponse, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="Credit Risk Model Comparison API",
    description="Serves predictions from the 8 tuned models in credit-risk-mlops. See /models for the full comparison.",
    lifespan=lifespan,
)


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


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, db: Session = Depends(get_db)) -> PredictionResponse:
    try:
        pipeline = get_model(request.model_name)
        model_row = get_model_row(request.model_name)
    except KeyError:
        raise HTTPException(
            status_code=422, detail=f"Unknown model '{request.model_name}'. See GET /models for valid names."
        )

    application = request.application
    X = pd.DataFrame([application.model_dump(by_alias=True)])
    probability_of_default = float(pipeline.predict_proba(X)[0, 1])
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


if __name__ == "__main__":
    # Local dev convenience: `python -m api.main`, matching the `python -m ml.xxx.yyy`
    # pattern used by every other entry point in this project, instead of a
    # separately-remembered uvicorn CLI invocation. reload=True needs the "module:app"
    # string form (not the `app` object) - uvicorn re-imports it fresh in a subprocess on
    # every file change. Production (Docker/K8s) will invoke uvicorn directly with
    # different flags (host 0.0.0.0, no reload), bypassing this block.
    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
