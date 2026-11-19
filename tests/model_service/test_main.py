from fastapi.testclient import TestClient

from model_service.main import create_app


def test_health_ok_when_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_health_503_when_model_files_missing(tmp_path):
    # empty folder: the service must still start, just report it isn't ready
    with TestClient(create_app(tmp_path)) as client:
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["model_loaded"] is False
        assert client.post("/predict", json={"features": {}}).status_code == 503


def test_predict_returns_probability_and_prediction(client, valid_features):
    response = client.post("/predict", json=valid_features)
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probability"] <= 1
    assert body["prediction"] == int(body["probability"] >= body["threshold"])
    assert body["threshold"] == 0.5
    assert body["model_version"] == "test"


def test_predict_422_lists_missing_features(client, valid_features):
    del valid_features["features"]["age"]
    response = client.post("/predict", json=valid_features)
    assert response.status_code == 422
    assert response.json()["detail"]["missing_features"] == ["age"]


def test_predict_ignores_extra_keys(client, valid_features):
    valid_features["features"]["project"] = "de-ml-final"
    assert client.post("/predict", json=valid_features).status_code == 200


def test_predict_handles_rare_category(client, valid_features):
    valid_features["features"]["education"] = "illiterate"
    assert client.post("/predict", json=valid_features).status_code == 200


def test_predict_accepts_int_for_float_column(client, valid_features):
    valid_features["features"]["nr.employed"] = 4963
    assert client.post("/predict", json=valid_features).status_code == 200


def test_predict_422_for_text_in_number_column(client, valid_features):
    valid_features["features"]["age"] = "thirty"
    response = client.post("/predict", json=valid_features)
    assert response.status_code == 422
