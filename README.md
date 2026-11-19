# Bank Marketing Subscription Predictor

CLD 375 (Cloud Machine Learning Foundation and Data Engineering) final project, Fall 2026.

Predicts whether a bank client will subscribe to a term deposit **before** the marketing
call, using the UCI Bank Marketing dataset (`bank-additional-full.csv`). The model is
trained in a Colab notebook and served on AWS EC2 by two Docker containers: a public
backend API and an internal model service.

## Repo layout

| Path | What it is | Owner |
| --- | --- | --- |
| `bankml/` | Shared feature and preprocessing code. The saved model depends on it, so the notebook and the model service both import it. | ML |
| `notebooks/` | Colab notebook: profiling, cleaning, training, evaluation | ML |
| `model/` | `model.joblib` + `metadata.json` exported from the notebook | ML |
| `model_service/` | Internal FastAPI app that loads the model (port 8001) | ML |
| `backend/` | Public FastAPI app the client calls (port 8000) | Cloud |
| `scripts/` | `train_placeholder.py`: quick stand-in model so the backend can be built early | ML |
| `tests/` | pytest suite | both |

## Running the tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Model service (internal, port 8001)

Loads `model/model.joblib` + `model/metadata.json` once at startup.

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | `{"status": "ok", "model_loaded": true}`, or 503 if the model failed to load |
| POST | `/predict` | `{"features": {...17 columns...}}` -> `{"probability", "prediction", "threshold", "model_version"}`; 422 if a column is missing or unusable |

Run it locally (from the repo root, with the virtual environment active):

```bash
MODEL_DIR=model uvicorn model_service.main:app --port 8001
```

Build and push the image (from the repo root; `--platform linux/amd64` because EC2 is not ARM like Apple Silicon):

```bash
docker buildx build --platform linux/amd64 -f model_service/Dockerfile -t <dockerhub-user>/bank-model-service:v0 --push .
```

`v0` carried the placeholder model from `scripts/train_placeholder.py`. `model/` now holds **v1**, the final stacking model saved by notebook section 7.4 (threshold 0.74, test F1 0.531). Build the image as `bank-model-service:v1`.

`tests/model_service/test_parity.py` checks that the saved model gives the same probabilities as the notebook for 5 test clients (`tests/fixtures/parity_rows.json`). The 2 "yes" and 3 "no" clients in that file also make good Postman demo cases.
