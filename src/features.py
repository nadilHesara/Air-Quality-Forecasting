"""
Self-contained feature engineering for next-day PM2.5 forecasting.

This module is the **single source of truth** for turning a raw daily
weather + air-quality table into the model's feature matrix.  The exact
same :func:`make_features` function is used at training time and at
inference time, which guarantees that the columns, order, and semantics
the model sees in production are identical to the ones it was trained on.

Leakage guarantee
------------------
Every engineered feature for a given day ``t`` depends **only** on
information available up to and including day ``t`` (lags use ``.shift(k)``
with ``k >= 1``; rolling windows are computed on the current-and-past
series and are *not* centred).  The prediction target — tomorrow's PM2.5,
``pm25_next_day`` — is the only column that looks one day into the future,
and it is produced by a separate function (:func:`add_target`) that is
**never** called during inference.

Run a quick self-check::

    python -m src.features
"""

from __future__ import annotations

import pandas as pd

# ── Raw columns expected from the data pipeline ──────────────────────────────
# These come straight from ``src/build_dataset.py`` (weather + air-quality
# joined on ``date``).  ``make_features`` validates that they are present.
RAW_PM_COLUMN: str = "pm2_5_mean"

WEATHER_COLUMNS: list[str] = [
    "temperature_2m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
]

# Air-quality context available on day ``t`` (today's pollution levels).
POLLUTION_COLUMNS: list[str] = [
    "pm2_5_mean",
    "pm10_mean",
]

# Target column name (next-day PM2.5).  Defined here so training and any
# downstream code agree on the name.
TARGET_COLUMN: str = "pm25_next_day"

# Lag horizons (in days) for PM2.5.  1 = yesterday, 7 = same weekday last week.
PM_LAGS: list[int] = [1, 2, 3, 7, 14]

# Rolling-window sizes (in days) for PM2.5 statistics.
PM_ROLLING_WINDOWS: list[int] = [3, 7, 14, 30]


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if any required raw column is missing."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _ensure_sorted_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with a sorted ``DatetimeIndex`` (required for lags)."""
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "Input DataFrame must have a DatetimeIndex (the daily 'date' index). "
            f"Got index of type {type(df.index).__name__}."
        )
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model feature matrix from a raw daily table.

    This function is **self-contained** and is called identically at
    training and inference time.  Pass in the full available history (the
    more history, the more lag/rolling features can be populated) and read
    off the final row(s) to predict the next day.

    Every feature for day ``t`` uses only information available on day
    ``t`` or earlier — no future leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Daily table indexed by a ``DatetimeIndex`` named ``date``.  Must
        contain :data:`RAW_PM_COLUMN`, the :data:`WEATHER_COLUMNS`, and the
        :data:`POLLUTION_COLUMNS`.  The target column is *not* required and
        is ignored if present.

    Returns
    -------
    pd.DataFrame
        A new DataFrame, aligned to the input index, containing exactly the
        engineered feature columns (see :func:`feature_names`).  Early rows
        will contain ``NaN`` where there is insufficient history for a lag
        or rolling window; callers decide how to handle those (drop for
        training, take the last complete row for inference).
    """
    df = _ensure_sorted_datetime_index(df)
    _require_columns(df, [RAW_PM_COLUMN, *WEATHER_COLUMNS, *POLLUTION_COLUMNS])

    pm = df[RAW_PM_COLUMN]
    feats = pd.DataFrame(index=df.index)

    # ── Today's pollution context (available on day t) ───────────────────
    for col in POLLUTION_COLUMNS:
        feats[col] = df[col]

    # ── Weather (available on day t) ─────────────────────────────────────
    for col in WEATHER_COLUMNS:
        feats[col] = df[col]

    # ── Lagged PM2.5 ─────────────────────────────────────────────────────
    for lag in PM_LAGS:
        feats[f"pm25_lag{lag}"] = pm.shift(lag)

    # ── Rolling PM2.5 statistics (current + past window, never centred) ───
    # ``min_periods == window`` keeps early rows NaN rather than computing a
    # statistic from too little data; this is dropped for training and never
    # reached at inference once enough history is accumulated.
    for window in PM_ROLLING_WINDOWS:
        roll = pm.rolling(window=window, min_periods=window)
        feats[f"pm25_rolling{window}_mean"] = roll.mean()
        feats[f"pm25_rolling{window}_std"] = roll.std()

    # Day-over-day change in PM2.5 (yesterday vs the day before).
    feats["pm25_diff1"] = pm.shift(1) - pm.shift(2)

    # ── Calendar features ────────────────────────────────────────────────
    feats["day_of_week"] = df.index.dayofweek
    feats["month"] = df.index.month
    feats["day_of_year"] = df.index.dayofyear
    feats["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

    return feats


def feature_names() -> list[str]:
    """Return the ordered list of feature column names produced by
    :func:`make_features`.

    Useful for asserting train/inference column alignment without having to
    run the transform on data.
    """
    names: list[str] = []
    names += list(POLLUTION_COLUMNS)
    names += list(WEATHER_COLUMNS)
    names += [f"pm25_lag{lag}" for lag in PM_LAGS]
    for window in PM_ROLLING_WINDOWS:
        names += [f"pm25_rolling{window}_mean", f"pm25_rolling{window}_std"]
    names += ["pm25_diff1"]
    names += ["day_of_week", "month", "day_of_year", "is_weekend"]
    return names


def add_target(df: pd.DataFrame) -> pd.Series:
    """Create the supervised target: **next-day** mean PM2.5.

    This is the only future-looking transform and is used **only at
    training time** — never call it during inference.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain :data:`RAW_PM_COLUMN`.

    Returns
    -------
    pd.Series
        ``pm25_next_day`` aligned to ``df`` — each day's value is the PM2.5
        of the following day.  The final row is ``NaN`` (no known future).
    """
    _require_columns(df, [RAW_PM_COLUMN])
    target = df[RAW_PM_COLUMN].shift(-1)
    target.name = TARGET_COLUMN
    return target


def build_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Convenience wrapper: build features + target and drop incomplete rows.

    Rows with any ``NaN`` (early rows lacking lag/rolling history, and the
    final row lacking a known next-day target) are dropped so that ``X`` and
    ``y`` are fully aligned and model-ready.

    Parameters
    ----------
    df : pd.DataFrame
        Raw daily table (see :func:`make_features`).

    Returns
    -------
    (X, y) : tuple[pd.DataFrame, pd.Series]
        Feature matrix and aligned target, both indexed by date.
    """
    X = make_features(df)
    y = add_target(df)

    combined = X.copy()
    combined[TARGET_COLUMN] = y
    combined = combined.dropna()

    y_clean = combined[TARGET_COLUMN]
    X_clean = combined.drop(columns=[TARGET_COLUMN])
    # Guarantee deterministic column order matching feature_names().
    X_clean = X_clean[feature_names()]
    return X_clean, y_clean


if __name__ == "__main__":
    # Quick self-check against the saved training data.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import config  # noqa: E402

    raw = pd.read_parquet(config.DATA_DIR / "training_data.parquet")
    X, y = build_supervised(raw)
    print(f"Raw rows:        {len(raw)}")
    print(f"Supervised rows: {len(X)}  (features={X.shape[1]})")
    print(f"Columns match feature_names(): {list(X.columns) == feature_names()}")
    print(f"Any NaN in X: {bool(X.isnull().any().any())}   Any NaN in y: {bool(y.isnull().any())}")
    print("\nFeature columns:")
    for name in feature_names():
        print(f"  - {name}")
