"""
Inference smoke test.

We monkeypatch the two fetch functions inside :mod:`src.inference` so no
network call happens, and inject a fake model bundle so the real
``models/model.joblib`` is never needed.  The goal is to confirm the serving
path wires together correctly end-to-end: one finite prediction, and
``prediction_date == feature_date + 1 day``.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd

import src.inference as inference
from src.features import feature_names


class _StubModel:
    """A stand-in model that returns a constant, checking the feature schema.

    It asserts it receives exactly the trained feature columns (in order),
    which doubles as a serving-side alignment check.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert list(X.columns) == feature_names()
        return np.full(len(X), 42.0)


def _fake_bundle() -> dict:
    return {
        "model": _StubModel(),
        "feature_names": feature_names(),
        "target": "pm25_next_day",
        "trained_through": "2024-01-01",
    }


def test_predict_next_day_smoke(monkeypatch, seasonal_raw: pd.DataFrame) -> None:
    """predict_next_day yields one finite number, dated one day ahead."""
    # Both fetch functions are joined on ``date`` inside fetch_recent_daily;
    # returning the same canned frame from each yields that frame after the
    # inner join (identical index, disjoint-enough columns for make_features).
    weather_cols = [
        "temperature_2m_mean",
        "wind_speed_10m_max",
        "wind_direction_10m_dominant",
        "relative_humidity_2m_mean",
        "precipitation_sum",
        "surface_pressure_mean",
    ]
    pollution_cols = ["pm2_5_mean", "pm10_mean"]

    def fake_weather(*args, **kwargs) -> pd.DataFrame:
        return seasonal_raw[weather_cols].copy()

    def fake_air_quality(*args, **kwargs) -> pd.DataFrame:
        return seasonal_raw[pollution_cols].copy()

    monkeypatch.setattr(inference, "fetch_daily_weather", fake_weather)
    monkeypatch.setattr(inference, "fetch_daily_air_quality", fake_air_quality)

    result = inference.predict_next_day(bundle=_fake_bundle(), today=date(2024, 4, 1))

    # One finite prediction.
    assert isinstance(result.predicted_pm25, float)
    assert math.isfinite(result.predicted_pm25)
    assert result.predicted_pm25 == 42.0  # the stub's constant, rounded

    # The core contract: predict tomorrow from the latest complete day.
    assert result.prediction_date == result.feature_date + timedelta(days=1)

    # The feature date must be the last day present in the canned frame.
    assert result.feature_date == seasonal_raw.index[-1].date()

    # Structured extras used by the API/dashboard.
    assert result.features and list(result.features) == feature_names()
    assert result.history, "history should carry recent actuals for the chart"


def test_predict_next_day_raises_on_insufficient_history(monkeypatch) -> None:
    """Too little history to fill the 30-day window → a clear ValueError."""
    short_index = pd.date_range("2024-01-01", periods=5, freq="D", name="date")
    short = pd.DataFrame(
        {
            "pm2_5_mean": np.arange(5, dtype=float),
            "pm10_mean": np.arange(5, dtype=float),
            "temperature_2m_mean": np.zeros(5),
            "wind_speed_10m_max": np.zeros(5),
            "wind_direction_10m_dominant": np.zeros(5),
            "relative_humidity_2m_mean": np.zeros(5),
            "precipitation_sum": np.zeros(5),
            "surface_pressure_mean": np.zeros(5),
        },
        index=short_index,
    )

    monkeypatch.setattr(inference, "fetch_daily_weather", lambda *a, **k: short.copy())
    monkeypatch.setattr(inference, "fetch_daily_air_quality", lambda *a, **k: short.copy())

    try:
        inference.predict_next_day(bundle=_fake_bundle(), today=date(2024, 1, 6))
    except ValueError:
        return
    raise AssertionError("expected ValueError on insufficient history")
