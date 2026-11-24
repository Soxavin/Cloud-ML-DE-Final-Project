import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.model_client import ModelServiceError, ModelServiceUnavailable
from backend.schemas import PredictRequest

VALID = {
    "project": "de-ml-final", "age": 30, "job": "student", "marital": "single",
    "education": "professional.course", "default": "no", "contact": "cellular",
    "month": "sep", "day_of_week": "tue", "campaign": 2,
    "days_since_last_contact": 6, "previous": 1, "poutcome": "success",
}


class FakeClient:
    def __init__(self, answer=None, error=None, healthy=True):
        self.answer = answer or {"probability": 0.93, "prediction": 1, "threshold": 0.74, "model_version": "v1"}
        self.error = error
        self.healthy = healthy
        self.sent = None

    async def predict(self, features):
        self.sent = features
        if self.error:
            raise self.error
        return self.answer

    async def is_healthy(self):
        return self.healthy


def make(fake):
    return TestClient(create_app(fake))


# ---- schema (Task A)
def test_schema_rejects_unknown_job():
    with pytest.raises(Exception):
        PredictRequest(**{**VALID, "job": "doctor"})

def test_schema_rejects_missing_age():
    body = {k: v for k, v in VALID.items() if k != "age"}
    with pytest.raises(Exception):
        PredictRequest(**body)

def test_schema_rejects_misspelled_field():
    body = {k: v for k, v in VALID.items() if k != "age"} | {"agee": 30}
    with pytest.raises(Exception):
        PredictRequest(**body)

def test_schema_accepts_null_days():
    assert PredictRequest(**{**VALID, "days_since_last_contact": None}).days_since_last_contact is None

def test_schema_requires_days_field():
    body = {k: v for k, v in VALID.items() if k != "days_since_last_contact"}
    with pytest.raises(Exception):
        PredictRequest(**body)


# ---- API (Task B)
def test_valid_request_returns_prediction():
    r = make(FakeClient()).post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == "yes"
    assert body["subscribe_probability"] == 0.93
    assert body["economic_context"]["emp_var_rate"] == -1.1
    assert body["input_received"]["age"] == 30

def test_prediction_zero_is_no():
    fake = FakeClient(answer={"probability": 0.4, "prediction": 0, "threshold": 0.74, "model_version": "v1"})
    assert make(fake).post("/predict", json=VALID).json()["prediction"] == "no"

def test_never_contacted_is_sent_as_pdays_999():
    fake = FakeClient()
    make(fake).post("/predict", json={**VALID, "days_since_last_contact": None})
    assert fake.sent["pdays"] == 999
    make(fake).post("/predict", json=VALID)
    assert fake.sent["pdays"] == 6
    assert "days_since_last_contact" not in fake.sent

def test_economic_values_are_added():
    fake = FakeClient()
    make(fake).post("/predict", json=VALID)
    assert fake.sent["euribor3m"] == 1.028
    assert fake.sent["emp.var.rate"] == -1.1
    assert fake.sent["cons.price.idx"] == 94.767
    assert fake.sent["cons.conf.idx"] == -50.8
    assert fake.sent["nr.employed"] == 4963.6
    assert len(fake.sent) == 17

def test_economic_values_cannot_be_overridden():
    r = make(FakeClient()).post("/predict", json={**VALID, "euribor3m": 5.0})
    assert r.status_code == 422

def test_project_field_is_not_sent_to_model():
    fake = FakeClient()
    make(fake).post("/predict", json=VALID)
    assert "project" not in fake.sent

def test_unknown_job_returns_422():
    assert make(FakeClient()).post("/predict", json={**VALID, "job": "doctor"}).status_code == 422

def test_misspelled_field_returns_422():
    body = {k: v for k, v in VALID.items() if k != "age"} | {"agee": 30}
    assert make(FakeClient()).post("/predict", json=body).status_code == 422

def test_bad_numbers_return_422():
    c = make(FakeClient())
    assert c.post("/predict", json={**VALID, "age": "thirty"}).status_code == 422
    assert c.post("/predict", json={**VALID, "campaign": 0}).status_code == 422

def test_model_unreachable_returns_503():
    r = make(FakeClient(error=ModelServiceUnavailable("down"))).post("/predict", json=VALID)
    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"]

def test_model_error_returns_502():
    assert make(FakeClient(error=ModelServiceError("bad"))).post("/predict", json=VALID).status_code == 502

def test_unexpected_error_returns_500_generic():
    c = TestClient(create_app(FakeClient(error=RuntimeError("secret detail"))), raise_server_exceptions=False)
    r = c.post("/predict", json=VALID)
    assert r.status_code == 500
    assert "secret" not in r.text

def test_health_ok():
    r = make(FakeClient()).get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "backend": "ok", "model_service": "ok"}

def test_health_503_when_model_unreachable():
    r = make(FakeClient(healthy=False)).get("/health")
    assert r.status_code == 503
    assert r.json()["model_service"] == "unreachable"
