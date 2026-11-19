# Model service image. Build from the REPO ROOT so bankml/ and model/ are in the build context:
#   docker buildx build --platform linux/amd64 -f model_service/Dockerfile -t <user>/bank-model-service:<tag> --push .

# same Python minor version as Google Colab, where the model is trained
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_DIR=/app/model

WORKDIR /app

# install libraries first so Docker can reuse this layer when only our code changes
COPY model_service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bankml must be importable as "bankml": the saved model refers to it by that name
COPY bankml/ bankml/
COPY model_service/ model_service/
COPY model/ model/

# don't run as root inside the container
RUN useradd --create-home appuser
USER appuser

EXPOSE 8001

# curl isn't in the slim image, so the health check uses Python itself
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "model_service.main:app", "--host", "0.0.0.0", "--port", "8001"]
