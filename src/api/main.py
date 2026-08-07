from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .model_registry import DEFAULT_MODEL, TEST_METRICS, available_models, get_model, load_models
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
        models=[ModelInfo(name=name, metrics=ModelMetrics(**TEST_METRICS[name])) for name in available_models()],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        pipeline = get_model(request.model_name)
    except KeyError:
        raise HTTPException(
            status_code=422, detail=f"Unknown model '{request.model_name}'. See GET /models for valid names."
        )

    X = pd.DataFrame([request.application.model_dump(by_alias=True)])
    probability_of_default = float(pipeline.predict_proba(X)[0, 1])

    return PredictionResponse(
        model_used=request.model_name,
        probability_of_default=probability_of_default,
        prediction=int(probability_of_default >= 0.5),
    )
