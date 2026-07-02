"""
API tests with FastAPI's TestClient.

Inference is fully mocked (no network, no model file), so these tests only
exercise the HTTP layer: routing, status codes, and the response schema.
"""

from __future__ import annotations

from datetime import date, timedelta

from fastapi.testclient import TestClient

import src.api as api
from src.inference import ModelNotFoundError, PredictionResult

client = TestClient(api.app)


class _FakePath:
    """A minimal stand-in for MODEL_PATH with a controllable ``exists()``.

    ``pathlib.Path`` instances are immutable, so we swap the whole object in
    the ``api`` namespace rather than patching its ``exists`` attribute.
    """

    def __init__(self, exists: bool) -> None:
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def __str__(self) -> str:
        return "/fake/models/model.joblib"


def _fake_result() -> PredictionResult:
    feature_date = date(2024, 4, 1)
    return PredictionResult(
        city="Testville",
        latitude=1.23,
        longitude=4.56,
        feature_date=feature_date,
        prediction_date=feature_date + timedelta(days=1),
        predicted_pm25=42.0,
        features={"pm2_5_mean": 12.3, "pm10_mean": 20.1},
        model_version="StubModel@2024-04-01-26feat",
        history=[{"date": "2024-03-31", "pm2_5": 11.0}],
    )


# ── /health ──────────────────────────────────────────────────────────────────
def test_health_ok_when_model_loadable(monkeypatch) -> None:
    """Model present + loadable → status 'ok' with a model_version."""
    monkeypatch.setattr(api, "MODEL_PATH", _FakePath(exists=True))
    monkeypatch.setattr(api, "load_model", lambda: {"model": object()})
    monkeypatch.setattr(api, "model_version", lambda bundle: "StubModel@x-26feat")

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_present"] is True
    assert body["model_version"] == "StubModel@x-26feat"
    # Schema keys the dashboard/monitoring rely on.
    assert set(body) >= {"status", "city", "model_present", "model_path"}


def test_health_degraded_when_model_missing(monkeypatch) -> None:
    """No model file → status 'degraded' (still HTTP 200, with a detail)."""
    monkeypatch.setattr(api, "MODEL_PATH", _FakePath(exists=False))

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["model_present"] is False
    assert body["detail"]


# ── /predict ─────────────────────────────────────────────────────────────────
def test_predict_returns_expected_schema(monkeypatch) -> None:
    """Happy path: 200 with every documented field, correctly typed."""
    monkeypatch.setattr(api, "predict_next_day", lambda: _fake_result())

    resp = client.get("/predict")
    assert resp.status_code == 200
    body = resp.json()

    expected_keys = {
        "city",
        "latitude",
        "longitude",
        "feature_date",
        "prediction_date",
        "predicted_pm25",
        "units",
        "model_version",
        "features",
    }
    assert set(body) == expected_keys
    assert body["predicted_pm25"] == 42.0
    assert body["feature_date"] == "2024-04-01"
    assert body["prediction_date"] == "2024-04-02"
    assert isinstance(body["features"], dict)


def test_predict_503_when_model_missing(monkeypatch) -> None:
    """ModelNotFoundError surfaces as 503 (service not ready)."""
    def _raise() -> PredictionResult:
        raise ModelNotFoundError("no model on disk")

    monkeypatch.setattr(api, "predict_next_day", _raise)
    resp = client.get("/predict")
    assert resp.status_code == 503


def test_predict_422_when_data_insufficient(monkeypatch) -> None:
    """ValueError (insufficient live data) surfaces as 422."""
    def _raise() -> PredictionResult:
        raise ValueError("not enough history")

    monkeypatch.setattr(api, "predict_next_day", _raise)
    resp = client.get("/predict")
    assert resp.status_code == 422
