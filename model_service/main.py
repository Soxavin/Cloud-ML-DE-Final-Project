"""Internal model service.

Loads the trained pipeline once at startup and answers prediction requests
from the backend on port 8001. It is never exposed to the internet: input
checking for users happens in the backend, so this service only checks that
the columns the model needs are present and usable.
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# reuse uvicorn's logger so our lines show up in `docker compose logs` next to the request lines
logger = logging.getLogger("uvicorn.error")


class PredictRequest(BaseModel):
    features: dict[str, Any]


class PredictResponse(BaseModel):
    probability: float
    prediction: int
    threshold: float
    model_version: str


def create_app(model_dir: Path) -> FastAPI:
    # filled in at startup; stays None if loading fails so /health can report it
    state = {"model": None, "metadata": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            state["model"] = joblib.load(model_dir / "model.joblib")
            state["metadata"] = json.loads((model_dir / "metadata.json").read_text())
            logger.info("Model %s loaded from %s", state["metadata"]["model_version"], model_dir)
        except Exception:
            # keep the service running so the problem is visible on /health instead of a crash loop
            logger.exception("Could not load the model from %s", model_dir)
        yield

    app = FastAPI(title="Bank Marketing model service", lifespan=lifespan)

    @app.get("/health")
    def health():
        if state["model"] is None:
            return JSONResponse(status_code=503, content={"status": "unavailable", "model_loaded": False})
        return {"status": "ok", "model_loaded": True}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        if state["model"] is None:
            return JSONResponse(status_code=503, content={"detail": "Model is not loaded"})

        expected = state["metadata"]["features"]
        missing = []
        for name in expected:
            if name not in request.features:
                missing.append(name)
        if missing:
            return JSONResponse(status_code=422, content={"detail": {"missing_features": missing}})

        # build the row in training column order; any extra keys are simply left out
        row = {}
        for name in expected:
            row[name] = request.features[name]

        try:
            probability = float(state["model"].predict_proba(pd.DataFrame([row]))[0, 1])
        except (ValueError, TypeError) as error:
            logger.warning("Rejected unusable feature values: %s", error)
            return JSONResponse(status_code=422, content={"detail": "Feature values could not be used by the model"})

        threshold = state["metadata"]["threshold"]
        prediction = int(probability >= threshold)
        logger.info("Prediction: probability=%.4f prediction=%d", probability, prediction)

        return PredictResponse(
            probability=round(probability, 4),
            prediction=prediction,
            threshold=threshold,
            model_version=state["metadata"]["model_version"],
        )

    return app


app = create_app(Path(os.getenv("MODEL_DIR", "/app/model")))
