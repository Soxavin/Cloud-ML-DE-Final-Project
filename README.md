# Bank Marketing Subscription Predictor

CLD 375 (Cloud Machine Learning Foundation and Data Engineering) final project, Fall 2026.

Predicts whether a bank client will subscribe to a term deposit **before** the marketing call, using the
UCI Bank Marketing dataset (`bank-additional-full.csv`). The model is trained in a Colab notebook and served
on AWS EC2 by two Docker containers: a public backend API and an internal model service.

## Project status (24 Nov 2026)

| Part | Status |
| --- | --- |
| Notebook: cleaning, feature selection, 7 models, stacking, evaluation | Done |
| Final model `v1` (stacking, threshold 0.74, test F1 0.531, ROC-AUC 0.811) | Done, in `model/` |
| Model service (internal API, port 8001) | Done |
| Backend (public API, port 8000) | Done |
| Docker images + `docker-compose.yml` | Done |
| Postman collection | Done, in `postman/` |
| Tests | 37 passing (`python -m pytest`) |

## How a request flows

```
Postman --POST /predict--> backend (port 8000, public)
                             |  1. validates the input (422 if invalid)
                             |  2. maps days_since_last_contact -> pdays (null -> 999)
                             |  3. adds the 5 economic values from backend/config.py
                             v
                           model-service (port 8001, private Docker network only)
                             |  4. runs the saved pipeline: cleaning + stacking model
                             |  5. applies the threshold from model/metadata.json
                             v
Postman <--prediction, probability, threshold, model version, input received--
```

## Repo layout

| Path | What it is | Owner |
| --- | --- | --- |
| `notebooks/bank_marketing.ipynb` | Colab notebook: profiling, cleaning, feature selection, training, evaluation, saving the model | ML |
| `bankml/` | Shared code the saved model depends on (feature step, preprocessing, the frozen 17 inputs). The notebook and the model service both import it | ML |
| `model/` | `model.joblib` (whole trained pipeline) + `metadata.json` (inputs, threshold, categories, scores, library versions) | ML |
| `model_service/` | Internal FastAPI app that loads the model (port 8001) | ML |
| `backend/` | Public FastAPI app the client calls (port 8000) | Cloud |
| `docker-compose.yml` | Runs both containers together | Cloud |
| `postman/` | Postman collection for testing and the demo | Cloud |
| `scripts/train_placeholder.py` | Quick stand-in model used before `v1` existed (no longer needed) | ML |
| `tests/` | pytest suite for `bankml`, the model service and the backend | both |

## Running the tests

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest
```

Expected: `37 passed`. Always use `python -m pip` and `python -m pytest`, so the commands use the
environment's own Python (on a Mac with conda installed, plain `pytest` can pick up conda's Python instead).

The ML libraries are pinned to Google Colab's exact versions (Python 3.13, scikit-learn 1.6.1, numpy 2.4.6,
pandas 2.2.3, scipy 1.16.3, joblib 1.6.0). The model is saved with these versions, and a different
scikit-learn version may fail to load it.

## The public API (backend, port 8000)

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | `{"status": "ok", "backend": "ok", "model_service": "ok"}`, or 503 with `"model_service": "unreachable"` |
| POST | `/predict` | Prediction for one client (below) |

Request: all fields are required except `project`. `days_since_last_contact` is `null` for a client never
contacted before.

```json
{"project": "de-ml-final", "age": 30, "job": "student", "marital": "single",
 "education": "professional.course", "default": "no", "contact": "cellular", "month": "sep",
 "day_of_week": "tue", "campaign": 2, "days_since_last_contact": 6, "previous": 1, "poutcome": "success"}
```

Response:

```json
{"prediction": "yes", "subscribe_probability": 0.9273, "threshold": 0.74, "model_version": "v1",
 "economic_context": {"emp_var_rate": -1.1, "cons_price_idx": 94.767, "cons_conf_idx": -50.8,
                      "euribor3m": 1.028, "nr_employed": 4963.6},
 "input_received": {"...": "the validated request"}}
```

Allowed values for each text field are in `backend/schemas.py` (copied from `model/metadata.json`).

| Code | Meaning |
| --- | --- |
| 422 | Invalid input: missing or misspelled field, wrong type, value not allowed, out of range |
| 502 | The model service answered with an error |
| 503 | The model service is unreachable, still loading, or slower than 5 seconds |
| 500 | Anything unexpected (details only in the server log) |

**Economic values.** The backend always uses the last month in the dataset (Nov 2010) for the five economic
inputs. That was a low-interest-rate month in which about 45% of clients subscribed, so the API says "yes"
more often than the 11% overall average. This is expected: the economy is one of the model's strongest
signals (notebook sections 4.8 and 7.3).

## The internal model service (port 8001)

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | `{"status": "ok", "model_loaded": true}`, or 503 if the model failed to load |
| POST | `/predict` | `{"features": {...17 columns, dataset names...}}` -> `{"probability", "prediction", "threshold", "model_version"}` |

`tests/model_service/test_parity.py` checks that the saved model gives the notebook's exact probabilities for
5 test clients (`tests/fixtures/parity_rows.json`).

## Running both services locally (no Docker)

From the repo root, with the virtual environment active, in two terminals:

```bash
MODEL_DIR=model uvicorn model_service.main:app --port 8001
MODEL_SERVICE_URL=http://localhost:8001 uvicorn backend.main:app --port 8000
```

Then `curl localhost:8000/health`, or point Postman's `base_url` at `http://localhost:8000`.

## Docker images

Build from the **repo root** (the `.` at the end), so `bankml/` and `model/` are in the build context.
`--platform linux/amd64` is required when building on Apple Silicon, because EC2 is not ARM:

```bash
docker login
docker buildx build --platform linux/amd64 -f model_service/Dockerfile -t <dockerhub-user>/bank-model-service:v1 --push .
docker buildx build --platform linux/amd64 -f backend/Dockerfile -t <dockerhub-user>/bank-backend:v1 --push .
```

`docker-compose.yml` uses the images under `ponitaponlork/`. If the model files ever change, rebuild and push
the model-service image with a new tag and update the compose file.

## Deploying on EC2

Instance: Ubuntu 24.04 LTS, `t3.micro`, key pair `vockey`, an Elastic IP (so the address survives lab
restarts). Security group: SSH (22) from your own IP, TCP 8000 from anywhere. **Port 8001 stays closed.**

```bash
ssh -i labsuser.pem ubuntu@<elastic-ip>
curl -fsSL https://get.docker.com | sudo sh        # Docker + Compose plugin
sudo usermod -aG docker ubuntu                      # then log out and back in
git clone https://github.com/Soxavin/Cloud-ML-DE-Final-Project.git
cd Cloud-ML-DE-Final-Project
docker compose pull
docker compose up -d
docker compose ps                                   # both containers "healthy" after up to a minute
curl localhost:8000/health
```

Useful commands on the server:

| Command | What it does |
| --- | --- |
| `docker compose ps` | Shows both containers and whether they are healthy |
| `docker compose logs --tail 20` | Recent requests passing through both services |
| `docker compose stop model-service` / `start model-service` | Demonstrates the 503 error handling |
| `docker compose pull && docker compose up -d` | Updates to newly pushed images |

Both containers use `restart: unless-stopped`, so they start again by themselves after the lab stops and
restarts the instance.

## Demo checklist

1. Start the lab 15 minutes early and check the instance is running.
2. In a browser: `http://<elastic-ip>:8000/health` shows both services ok.
3. Postman: import `postman/bank-marketing.postman_collection.json`, check `base_url`, then run:
    - **Health**
    - **Likely yes** (expect 0.9273, "yes"), **Likely no** (0.3971, "no"), **Borderline** (0.7282, "no", just under
      the 0.74 threshold)
    - **Errors** (all 422)
    - **Model down**: run `docker compose stop model-service` on EC2 first (expect 503), then `start` it again
4. On EC2: `docker compose ps` and `docker compose logs --tail 20`.

## Troubleshooting

| Problem | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'sklearn'` when running tests | The wrong Python is running (often conda's). Check `which python` points into `.venv`, and use `python -m pytest` |
| `exec format error` on EC2 | Image built for ARM on a Mac; rebuild with `--platform linux/amd64` |
| `Permission denied (publickey)` on SSH | Old `labsuser.pem`; download it again from the current lab session's Details panel |
| `/health` returns 503 right after a restart | The model service is still loading the model; wait a few seconds |
| Postman can't connect after a lab restart | The instance has no Elastic IP and its address changed; use the new address or attach an Elastic IP |
