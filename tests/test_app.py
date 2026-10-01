import pytest
from fastapi.testclient import TestClient

from app.main import app

FROSTY = {"air_temp_c": 1.5, "dew_point_c": -1.0, "wind_speed_ms": 0.8, "cloud_cover_pct": 10}
MILD = {"air_temp_c": 9.0, "dew_point_c": 4.0, "wind_speed_ms": 3.5, "cloud_cover_pct": 80}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_is_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_a_verdict(client):
    r = client.post("/predict", json=MILD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["frost_probability"] <= 1.0
    assert body["threshold"] == 0.35


def test_frosty_night_scores_higher_than_mild_one(client):
    frosty = client.post("/predict", json=FROSTY).json()
    mild = client.post("/predict", json=MILD).json()
    assert frosty["frost_probability"] > mild["frost_probability"]
    assert frosty["alert"] is True
    assert mild["alert"] is False


def test_impossible_reading_is_rejected(client):
    r = client.post("/predict", json={**MILD, "wind_speed_ms": -2})
    assert r.status_code == 422
