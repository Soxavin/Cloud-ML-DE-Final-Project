import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.config import ECONOMIC_CONTEXT, MODEL_SERVICE_URL, MODEL_TIMEOUT_SECONDS
from backend.model_client import ModelClient, ModelServiceError, ModelServiceUnavailable
from backend.schemas import PredictRequest, PredictResponse

logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO)


def to_model_features(request: PredictRequest) -> dict:
    data = request.model_dump(exclude={"project", "days_since_last_contact"})
    days = request.days_since_last_contact
    data["pdays"] = 999 if days is None else days
    data.update(ECONOMIC_CONTEXT)
    return data


def create_app(model_client) -> FastAPI:
    app = FastAPI(title="Bank Marketing Subscription Predictor")

    @app.get("/health")
    async def health():
        if await model_client.is_healthy():
            return {"status": "ok", "backend": "ok", "model_service": "ok"}
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "backend": "ok", "model_service": "unreachable"},
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        try:
            features = to_model_features(request)
            result = await model_client.predict(features)
            response = PredictResponse(
                prediction="yes" if result["prediction"] == 1 else "no",
                subscribe_probability=result["probability"],
                threshold=result["threshold"],
                model_version=result["model_version"],
                economic_context={k.replace(".", "_"): v for k, v in ECONOMIC_CONTEXT.items()},
                input_received=request.model_dump(),
            )
            logger.info(
                "predict project=%s prediction=%s probability=%.4f",
                request.project, response.prediction, response.subscribe_probability,
            )
            return response
        except ModelServiceUnavailable as exc:
            logger.warning("model service unavailable: %s", exc)
            return JSONResponse(status_code=503, content={"detail": "Model service unavailable, try again shortly"})
        except ModelServiceError as exc:
            logger.warning("model service error: %s", exc)
            return JSONResponse(status_code=502, content={"detail": "Model service returned an error"})
        except Exception:
            logger.exception("unexpected error in /predict")
            return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app(ModelClient(MODEL_SERVICE_URL, MODEL_TIMEOUT_SECONDS))
