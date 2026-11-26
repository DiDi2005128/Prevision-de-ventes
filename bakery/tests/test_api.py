"""Exemple de départ pour P4 (à compléter)."""
from datetime import date, timedelta

from fastapi.testclient import TestClient

from app.api.main import app

client = TestClient(app)


def make_history(n, start=date(2022, 7, 4)):
    return [
        {"week_start": (start + timedelta(weeks=i)).isoformat(), "revenue": 5000 + i}
        for i in range(n)
    ]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_valid():
    r = client.post("/predict", json={"history": make_history(5)})
    assert r.status_code == 200
    assert r.json()["week_start"] == "2022-08-08"
    assert r.json()["prediction_eur"] > 0


def test_predict_less_than_4_weeks():
    r = client.post("/predict", json={"history": make_history(3)})
    assert r.status_code == 422


def test_predict_gap_in_weeks():
    h = make_history(5)
    h[3]["week_start"] = "2022-09-01"
    assert client.post("/predict", json={"history": h}).status_code == 422


def test_predict_negative_revenue():
    h = make_history(4)
    h[0]["revenue"] = -1
    assert client.post("/predict", json={"history": h}).status_code == 422
