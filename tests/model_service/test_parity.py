"""The deployed model must predict exactly like the notebook.

parity_rows.json holds 5 test-set clients with the probability the notebook gave
each one (notebook section 7.4). This sends them through the real saved model
in model/ and compares.
"""
import json
from pathlib import Path

from fastapi.testclient import TestClient

from model_service.main import create_app

REPO_DIR = Path(__file__).resolve().parents[2]
PARITY_ROWS = json.loads((REPO_DIR / "tests" / "fixtures" / "parity_rows.json").read_text())


def test_real_model_matches_notebook():
    with TestClient(create_app(REPO_DIR / "model")) as client:
        for row in PARITY_ROWS:
            response = client.post("/predict", json={"features": row["features"]})
            assert response.status_code == 200
            body = response.json()
            assert abs(body["probability"] - row["expected_probability"]) < 1e-4
            assert body["prediction"] == row["expected_prediction"]
