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
| `tests/` | pytest suite | both |

## Running the tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```
