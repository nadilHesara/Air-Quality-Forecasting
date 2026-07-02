"""
Shared pytest fixtures for the Air-Quality-Forecasting test suite.

Everything here is **synthetic and offline** — no test touches the network,
the trained model on disk, or MLflow.  The fixtures build small, deterministic
daily frames shaped exactly like what the Open-Meteo fetch functions return
(a ``DatetimeIndex`` named ``date`` plus the raw weather + pollution columns),
so the real feature/inference code can run against them unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ``import config`` and ``from src...`` work no matter where pytest is
# invoked from (mirrors the sys.path shim the source modules use).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features import POLLUTION_COLUMNS, WEATHER_COLUMNS  # noqa: E402


def _raw_frame(pm25: np.ndarray, *, start: str = "2024-01-01") -> pd.DataFrame:
    """Assemble a raw daily table around a given PM2.5 series.

    The other pollution/weather columns are filled with simple, distinct
    deterministic values.  What matters for the feature/leakage tests is the
    PM2.5 column; the rest just need to be present and finite.
    """
    n = len(pm25)
    index = pd.date_range(start=start, periods=n, freq="D", name="date")
    data = {
        "pm2_5_mean": pm25.astype(float),
        "pm10_mean": pm25.astype(float) * 1.5,
        "temperature_2m_mean": np.linspace(24.0, 30.0, n),
        "wind_speed_10m_max": np.linspace(5.0, 15.0, n),
        "wind_direction_10m_dominant": np.linspace(0.0, 359.0, n),
        "relative_humidity_2m_mean": np.linspace(60.0, 90.0, n),
        "precipitation_sum": np.zeros(n),
        "surface_pressure_mean": np.linspace(1005.0, 1015.0, n),
    }
    frame = pd.DataFrame(data, index=index)
    # Sanity: the fixture must supply every raw column the features need.
    expected = set(WEATHER_COLUMNS) | set(POLLUTION_COLUMNS)
    assert expected.issubset(frame.columns), "fixture is missing a raw column"
    return frame


@pytest.fixture
def ramp_raw() -> pd.DataFrame:
    """A strictly increasing PM2.5 ramp (1, 2, 3, …).

    A monotonic ramp makes leakage trivial to detect: if any feature on day
    ``t`` secretly used a future value, changing a *future* PM2.5 point would
    change that feature — which the leakage test checks directly.
    """
    n = 80  # comfortably longer than the 30-day rolling window + 14-day lag
    pm25 = np.arange(1, n + 1, dtype=float)
    return _raw_frame(pm25)


@pytest.fixture
def seasonal_raw() -> pd.DataFrame:
    """A longer frame with a smooth seasonal-ish PM2.5 signal.

    Used where we want realistic (non-degenerate) values, e.g. the inference
    smoke test, without any special structure a test relies on.
    """
    n = 90
    t = np.arange(n)
    pm25 = 20.0 + 8.0 * np.sin(2 * np.pi * t / 30.0) + (t % 7)
    return _raw_frame(pm25)
