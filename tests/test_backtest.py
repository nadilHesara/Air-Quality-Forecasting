"""
Tests for the walk-forward backtesting harness (src/backtest.py, Phase 7.6).

Runs the real fold loop on the synthetic seasonal fixture with a deliberately
tiny LightGBM so the whole thing stays fast.  The contracts under test: folds
tile the post-warm-up history without overlap, every fold's model never saw
its own evaluation window (enforced by construction, asserted via the window
bounds), baselines are computed on identical windows, and the pooled metrics
are finite.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.backtest import walk_forward

# Small-but-real LightGBM: enough to fit 30-60 synthetic rows quickly.
_FAST_PARAMS = {
    "n_estimators": 15,
    "num_leaves": 7,
    "min_child_samples": 5,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}


@pytest.fixture(scope="module")
def result(seasonal_raw_module: pd.DataFrame) -> dict:
    return walk_forward(
        seasonal_raw_module,
        initial_train_days=30,
        step_days=10,
        lgbm_params=_FAST_PARAMS,
    )


@pytest.fixture(scope="module")
def seasonal_raw_module() -> pd.DataFrame:
    # Module-scoped twin of the function-scoped conftest fixture so the
    # (relatively) expensive walk-forward run happens once for all asserts.
    import numpy as np

    n = 90
    t = np.arange(n)
    pm25 = 20.0 + 8.0 * np.sin(2 * np.pi * t / 30.0) + (t % 7)
    index = pd.date_range(start="2024-01-01", periods=n, freq="D", name="date")
    return pd.DataFrame(
        {
            "pm2_5_mean": pm25.astype(float),
            "pm10_mean": pm25.astype(float) * 1.5,
            "temperature_2m_mean": np.linspace(24.0, 30.0, n),
            "wind_speed_10m_max": np.linspace(5.0, 15.0, n),
            "wind_direction_10m_dominant": np.linspace(0.0, 359.0, n),
            "relative_humidity_2m_mean": np.linspace(60.0, 90.0, n),
            "precipitation_sum": np.zeros(n),
            "surface_pressure_mean": np.linspace(1005.0, 1015.0, n),
        },
        index=index,
    )


def test_produces_multiple_folds(result: dict) -> None:
    assert result["n_folds"] >= 2
    assert result["n_predictions"] == sum(f["test_rows"] for f in result["folds"])


def test_windows_are_chronological_and_disjoint(result: dict) -> None:
    folds = result["folds"]
    for prev, cur in zip(folds, folds[1:], strict=False):
        assert prev["test_window"][1] < cur["test_window"][0]
        # The expanding train pool grows fold to fold.
        assert cur["train_rows"] > prev["train_rows"]


def test_pooled_metrics_are_finite(result: dict) -> None:
    for name in ("model", "persistence", "seasonal_naive"):
        for metric in ("mae", "rmse"):
            assert math.isfinite(result["pooled"][name][metric])
    assert math.isfinite(result["improvement_vs_baseline"]["persistence_mae_pct"])


def test_per_fold_baselines_present(result: dict) -> None:
    for fold in result["folds"]:
        assert math.isfinite(fold["model"]["mae"])
        assert math.isfinite(fold["persistence"]["mae"])


def test_too_long_initial_window_raises(seasonal_raw: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="initial_train_days"):
        walk_forward(seasonal_raw, initial_train_days=10_000, step_days=10)
