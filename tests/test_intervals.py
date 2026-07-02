"""
Tests for the prediction-interval (quantile) path added in Phase 6.5.

Covers three contracts:

* :func:`src.train.train_quantile_models` returns one model per level plus a
  conformal offset per level, and the offsets widen (never invert) the band.
* :func:`src.inference.predict_interval` applies offsets, orders the band, and
  degrades gracefully (``None``) for an older bundle with no quantile models.
* :func:`src.inference.predict_next_day` surfaces a well-ordered band when the
  bundle carries quantile models.

Everything is offline: a tiny synthetic supervised frame trains real (fast)
LightGBM quantile models; the serving test monkeypatches the fetch functions.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import src.inference as inference
from src.features import build_supervised, feature_names
from src.inference import predict_interval
from src.train import predict_quantiles, train_quantile_models


def _supervised_from(seasonal_raw: pd.DataFrame):
    # Repeat the fixture a few times to give the quantile models + conformal
    # calibration split enough rows to fit on.
    long = pd.concat(
        [seasonal_raw] * 6,
    )
    long.index = pd.date_range("2022-01-01", periods=len(long), freq="D", name="date")
    return build_supervised(long)


def test_train_quantile_models_returns_models_and_offsets(seasonal_raw) -> None:
    X, y = _supervised_from(seasonal_raw)
    models, offsets = train_quantile_models(X, y, levels=(0.1, 0.5, 0.9))

    assert set(models) == {0.1, 0.5, 0.9}
    assert set(offsets) == {0.1, 0.5, 0.9}
    assert offsets[0.5] == 0.0            # median needs no widening
    assert offsets[0.1] >= 0.0 and offsets[0.9] >= 0.0  # calibration only widens


def test_predict_quantiles_is_monotonic(seasonal_raw) -> None:
    X, y = _supervised_from(seasonal_raw)
    models, offsets = train_quantile_models(X, y, levels=(0.1, 0.5, 0.9))
    preds = predict_quantiles(models, X, offsets)
    # Per-row ordering: p10 <= p50 <= p90 everywhere (no quantile crossing).
    assert np.all(preds[0.1] <= preds[0.5] + 1e-9)
    assert np.all(preds[0.5] <= preds[0.9] + 1e-9)


def test_predict_interval_none_for_bundle_without_quantiles(seasonal_raw) -> None:
    """Older artifacts (no quantile models) degrade to a point forecast."""
    X, _ = build_supervised(seasonal_raw)
    row = X.iloc[[-1]]
    bundle = {"model": object(), "feature_names": feature_names()}
    assert predict_interval(bundle, row) == (None, None, None)


def test_predict_interval_orders_band(seasonal_raw) -> None:
    X, y = _supervised_from(seasonal_raw)
    models, offsets = train_quantile_models(X, y, levels=(0.1, 0.5, 0.9))
    bundle = {
        "model": models[0.5],
        "quantile_models": models,
        "quantile_offsets": offsets,
        "feature_names": feature_names(),
    }
    lower, upper, coverage = predict_interval(bundle, X.iloc[[-1]])
    assert lower is not None and upper is not None
    assert lower <= upper
    assert coverage == 0.8  # 0.9 - 0.1


def test_predict_next_day_surfaces_interval(monkeypatch, seasonal_raw) -> None:
    """End-to-end serving path returns a band bracketing the point forecast."""
    X, y = _supervised_from(seasonal_raw)
    models, offsets = train_quantile_models(X, y, levels=(0.1, 0.5, 0.9))
    bundle = {
        "model": models[0.5],
        "quantile_models": models,
        "quantile_offsets": offsets,
        "feature_names": feature_names(),
        "target": "pm25_next_day",
        "trained_through": "2024-01-01",
    }

    weather_cols = [
        "temperature_2m_mean", "wind_speed_10m_max", "wind_direction_10m_dominant",
        "relative_humidity_2m_mean", "precipitation_sum", "surface_pressure_mean",
    ]
    pollution_cols = ["pm2_5_mean", "pm10_mean"]
    monkeypatch.setattr(inference, "fetch_daily_weather",
                        lambda *a, **k: seasonal_raw[weather_cols].copy())
    monkeypatch.setattr(inference, "fetch_daily_air_quality",
                        lambda *a, **k: seasonal_raw[pollution_cols].copy())

    result = inference.predict_next_day(bundle=bundle, today=date(2024, 4, 1))
    assert result.pm25_lower is not None and result.pm25_upper is not None
    assert result.pm25_lower <= result.pm25_upper
    assert result.interval_coverage == 0.8
