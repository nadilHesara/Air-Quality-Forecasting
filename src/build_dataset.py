"""
Build the daily training dataset for PM2.5 next-day forecasting.

Orchestrates the full pipeline:
1. Fetch daily weather + air-quality data via Open-Meteo.
2. Inner-join on ``date``.
3. Engineer features + create the target using :mod:`src.features` — the
   **single source of truth** shared with training and serving, so this
   module never grows its own (drifting) copy of the feature logic.
4. Drop rows with NaN values introduced by shifting/rolling.
5. Save to ``data/training_data.parquet``.

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

import config  # noqa: E402  (after sys.path manipulation)
from src.features import TARGET_COLUMN, add_target, make_features

# ── Sibling modules ──────────────────────────────────────────────────────────
from src.fetch_air_quality import fetch_daily_air_quality
from src.fetch_weather import fetch_daily_weather

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Feature engineering ──────────────────────────────────────────────────────
# NOTE: feature/target logic deliberately lives in :mod:`src.features` (the one
# place shared by training and serving).  We only assemble the persisted table
# here; there is no second copy to drift out of sync.


def assemble_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Combine the raw joined table with the engineered features + target.

    Uses :func:`src.features.make_features` and :func:`src.features.add_target`
    so the columns written to ``training_data.parquet`` match exactly what the
    model sees at training and inference time.  Raw pollution/weather columns
    are kept alongside the engineered features (the baselines in
    ``src/train.py`` read raw ``pm2_5_mean`` directly).

    Parameters
    ----------
    raw : pd.DataFrame
        Weather + air-quality table joined on ``date`` (DatetimeIndex).

    Returns
    -------
    pd.DataFrame
        ``raw`` columns + the engineered feature columns + ``pm25_next_day``.
    """
    features = make_features(raw)
    # Only add engineered columns that aren't already in ``raw`` (make_features
    # re-emits the raw pollution/weather columns; keep the originals once).
    new_cols = [c for c in features.columns if c not in raw.columns]
    combined = raw.join(features[new_cols], how="left")
    combined[TARGET_COLUMN] = add_target(raw)
    return combined


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

    # 3. Feature engineering (delegated to src.features — single source) ------
    df = assemble_dataset(df)

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
