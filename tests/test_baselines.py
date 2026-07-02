"""
Baseline sanity tests: persistence and seasonal-naive must return the exact
values their definitions imply on a tiny, hand-checkable fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.train import persistence_prediction, seasonal_naive_prediction


def _tiny_raw() -> pd.DataFrame:
    """10 consecutive days with distinct, easily-read PM2.5 values.

    date        pm2_5_mean
    2024-01-01  10
    2024-01-02  11
    ...
    2024-01-10  19
    """
    index = pd.date_range("2024-01-01", periods=10, freq="D", name="date")
    return pd.DataFrame({"pm2_5_mean": np.arange(10, 20, dtype=float)}, index=index)


def test_persistence_returns_todays_value() -> None:
    """Persistence: prediction for row t (target = PM2.5 of t+1) is PM2.5 of t."""
    raw = _tiny_raw()
    # Predict for the last 3 rows (2024-01-08, -09, -10 → values 17, 18, 19).
    index = raw.index[-3:]
    preds = persistence_prediction(raw, index)

    expected = pd.Series([17.0, 18.0, 19.0], index=index)
    pd.testing.assert_series_equal(preds, expected, check_names=False)


def test_seasonal_naive_returns_value_from_seven_days_before_target() -> None:
    """Seasonal-naive: for row t, use PM2.5 at (t+1) - 7 days = t - 6 days.

    Row 2024-01-08 (value 17): reference day = 2024-01-02 → value 11.
    Row 2024-01-09 (value 18): reference day = 2024-01-03 → value 12.
    Row 2024-01-10 (value 19): reference day = 2024-01-04 → value 13.
    """
    raw = _tiny_raw()
    index = raw.index[-3:]
    preds = seasonal_naive_prediction(raw, index)

    expected = pd.Series([11.0, 12.0, 13.0], index=index, name="seasonal_naive")
    pd.testing.assert_series_equal(preds, expected)


def test_seasonal_naive_is_nan_when_reference_day_missing() -> None:
    """When (t - 6) predates the data, the baseline yields NaN, not a crash."""
    raw = _tiny_raw()
    # Row 2024-01-03 needs 2023-12-28, which is outside the fixture.
    index = raw.index[2:3]  # 2024-01-03
    preds = seasonal_naive_prediction(raw, index)
    assert np.isnan(preds.iloc[0])
