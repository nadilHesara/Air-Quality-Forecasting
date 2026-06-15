"""
Build the daily training dataset for PM2.5 next-day forecasting.

Orchestrates the full pipeline:
1. Fetch daily weather + air-quality data via Open-Meteo.
2. Inner-join on ``date``.
3. Engineer lag / rolling / calendar features.
4. Create the prediction target (next-day PM2.5).
5. Drop rows with NaN values introduced by shifting/rolling.
6. Save to ``data/training_data.parquet``.

Run directly::

    python -m src.build_dataset
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow running from the project root or directly by adding project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

# ── Sibling modules ──────────────────────────────────────────────────────────
from src.fetch_air_quality import fetch_daily_air_quality
from src.fetch_weather import fetch_daily_weather
import config  # noqa: E402  (after sys.path manipulation)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Feature engineering ──────────────────────────────────────────────────────


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag, rolling, and calendar features in-place.

    New columns created:

    * ``pm25_lag1``      – previous day's mean PM2.5
    * ``pm25_rolling7``  – 7-day rolling mean of PM2.5
    * ``day_of_week``    – 0 (Mon) … 6 (Sun)
    * ``month``          – 1 … 12

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``pm2_5_mean`` and a ``DatetimeIndex`` named ``date``.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with added feature columns.
    """
    df = df.copy()

    # Lag features
    df["pm25_lag1"] = df["pm2_5_mean"].shift(1)

    # Rolling statistics
    df["pm25_rolling7"] = df["pm2_5_mean"].rolling(window=7, min_periods=7).mean()

    # Calendar features
    df["day_of_week"] = df.index.dayofweek  # type: ignore[union-attr]
    df["month"] = df.index.month  # type: ignore[union-attr]

    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the prediction target: next-day average PM2.5.

    The target ``pm25_next_day`` is the PM2.5 mean shifted backward by 1 row,
    meaning each row's target is *tomorrow's* PM2.5 value.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``pm2_5_mean``.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``pm25_next_day`` column.
    """
    df = df.copy()
    df["pm25_next_day"] = df["pm2_5_mean"].shift(-1)
    return df


# ── Main pipeline ────────────────────────────────────────────────────────────


def build_training_dataset() -> pd.DataFrame:
    """Run the full data-ingestion and feature-engineering pipeline.

    Returns
    -------
    pd.DataFrame
        Clean training table saved as Parquet and returned for inspection.
    """
    # 1. Fetch raw data -------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Starting data ingestion for %s", config.CITY_NAME)
    logger.info("=" * 60)

    weather_df = fetch_daily_weather(
        latitude=config.LATITUDE,
        longitude=config.LONGITUDE,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        api_url=config.WEATHER_API_URL,
        daily_variables=config.DAILY_WEATHER_VARIABLES,
    )

    aq_df = fetch_daily_air_quality(
        latitude=config.LATITUDE,
        longitude=config.LONGITUDE,
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        api_url=config.AIR_QUALITY_API_URL,
        hourly_variables=config.HOURLY_AQ_VARIABLES,
    )

    # 2. Join on date ---------------------------------------------------------
    df = weather_df.join(aq_df, how="inner")
    logger.info(
        "After inner join: %d rows  (weather=%d, aq=%d)",
        len(df),
        len(weather_df),
        len(aq_df),
    )

    # 3. Feature engineering --------------------------------------------------
    df = add_features(df)
    df = add_target(df)

    # 4. Handle missing values ------------------------------------------------
    rows_before = len(df)
    df = df.dropna()
    rows_after = len(df)
    rows_dropped = rows_before - rows_after
    logger.info(
        "Dropped %d rows with NaN (lag/rolling/target edges). "
        "Remaining: %d rows.",
        rows_dropped,
        rows_after,
    )

    # Quick sanity check
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning("[WARNING] %d null values still present after dropna!", remaining_nulls)
    else:
        logger.info("[OK] No null values remain.")

    # 5. Save -----------------------------------------------------------------
    output_path = config.DATA_DIR / "training_data.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    logger.info("Saved training data -> %s  (%d rows x %d cols)", output_path, *df.shape)

    # Summary
    logger.info("-" * 60)
    logger.info("Column summary:")
    for col in df.columns:
        logger.info("  %-35s  dtype=%-12s  nulls=%d", col, df[col].dtype, df[col].isnull().sum())
    logger.info("-" * 60)

    return df


if __name__ == "__main__":
    build_training_dataset()
