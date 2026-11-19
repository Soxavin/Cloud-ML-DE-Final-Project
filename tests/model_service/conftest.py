import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from bankml.config import CATEGORICAL, LOG_NUMERIC, MODEL_INPUTS, NUMERIC
from bankml.pipeline import build_preprocessor
from model_service.main import create_app


@pytest.fixture
def model_dir(tmp_path, sample_rows):
    """A tiny but real model + metadata, trained on the 202 sample rows."""
    X = sample_rows[MODEL_INPUTS]
    y = (sample_rows["y"] == "yes").astype(int)
    model = Pipeline([
        ("prep", build_preprocessor(CATEGORICAL, NUMERIC, LOG_NUMERIC)),
        ("model", LogisticRegression(max_iter=1000)),
    ]).fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")

    metadata = {"model_version": "test", "features": MODEL_INPUTS, "threshold": 0.5}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


@pytest.fixture
def client(model_dir):
    # the with-block runs the app's startup, which is where the model gets loaded
    with TestClient(create_app(model_dir)) as test_client:
        yield test_client


@pytest.fixture
def valid_features(sample_rows):
    """One real row from the dataset, as the backend would send it."""
    row = sample_rows[MODEL_INPUTS].iloc[0].to_dict()
    return {"features": row}
