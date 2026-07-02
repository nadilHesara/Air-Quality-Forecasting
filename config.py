"""
Central configuration for the PM2.5 forecasting data pipeline.

Edit LATITUDE, LONGITUDE, CITY_NAME, and date range to target a different
location.  Every fetch function reads from this module, so a single change
here propagates everywhere.
"""

from pathlib import Path

# ── Location ─────────────────────────────────────────────────────────────────
CITY_NAME: str = "Colombo"
LATITUDE: float = 6.9271
LONGITUDE: float = 79.8612

# ── Date range (inclusive) ───────────────────────────────────────────────────
START_DATE: str = "2023-06-15"
END_DATE: str = "2026-06-14"

# ── API endpoints (Open-Meteo, no key required) ─────────────────────────────
WEATHER_API_URL: str = "https://archive-api.open-meteo.com/v1/archive"
AIR_QUALITY_API_URL: str = "https://air-quality-api.open-meteo.com/v1/air-quality"

# ── Weather variables requested as daily aggregations ────────────────────────
DAILY_WEATHER_VARIABLES: list[str] = [
    "temperature_2m_mean",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "relative_humidity_2m_mean",
    "precipitation_sum",
    "surface_pressure_mean",
]

# ── Air-quality variables (returned hourly, aggregated to daily in code) ─────
HOURLY_AQ_VARIABLES: list[str] = [
    "pm2_5",
    "pm10",
]

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# ── Training / evaluation knobs ──────────────────────────────────────────────
# Kept here (not hardcoded in train.py or the CI workflow) so the whole
# pipeline is reconfigured from one place.
TEST_HORIZON_DAYS: int = 90   # most-recent window held out as the final test set
N_CV_SPLITS: int = 5          # expanding-window TimeSeriesSplit folds (train pool)
RANDOM_STATE: int = 42

# Prediction-interval quantiles.  The point forecast stays the dedicated
# MAE/L2 regressor; these three quantile models (LightGBM objective="quantile")
# add a lower/median/upper band so the API and dashboard can show a range, not
# just a point.  QUANTILE_LEVELS[0] and [-1] define the nominal interval
# coverage (0.1 → 0.9 = an 80% central interval).
QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.5, 0.9)

# ── MLOps: experiment tracking (MLflow) ──────────────────────────────────────
# Local, file-based tracking store committed to the repo (mlruns/).  To point
# at a remote tracking server instead, set the MLFLOW_TRACKING_URI environment
# variable (e.g. "http://my-mlflow-server:5000") — it overrides the local
# default below.  See the README ("Pointing MLflow at a remote server").
MLFLOW_TRACKING_URI: str = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME: str = "pm25-next-day-forecast"

# The Model Registry name used when a registry-capable tracking backend is
# configured (sqlite:// or a remote server — the plain file store has no
# registry).  Training registers each new model here under the "staging"
# alias; the promotion gate moves it to "production".  See src/registry.py.
MLFLOW_REGISTERED_MODEL_NAME: str = "pm25-next-day"

# ── MLOps: data validation (run before every training) ──────────────────────
# Hard gates checked by src/validate.py on the fetched training frame.
# A violation aborts the run loudly instead of training on garbage.
MIN_TRAINING_ROWS: int = 400                      # lags/CV/test need real history
PM25_VALID_RANGE: tuple[float, float] = (0.0, 500.0)    # µg/m³, sane daily means
PM10_VALID_RANGE: tuple[float, float] = (0.0, 1000.0)   # µg/m³
MAX_NAN_FRACTION: float = 0.05                    # per required raw column
MAX_DATE_GAP_DAYS: int = 14                       # largest tolerated hole in the index

# ── MLOps: champion/challenger promotion gate ────────────────────────────────
# After a retrain, the new model's held-out test MAE is compared to the
# previously committed reports/metrics.json.  The new model is promoted only
# if its MAE is not worse than the champion's by more than this percentage
# (the test window shifts week to week, so a small tolerance avoids rejecting
# every run on noise).  See src/promotion_gate.py.
PROMOTION_MAX_MAE_REGRESSION_PCT: float = 5.0

# ── MLOps: drift monitoring ──────────────────────────────────────────────────
# src/drift.py compares the most recent DRIFT_WINDOW_DAYS of feature values
# against the reference profile persisted at training time
# (reports/feature_reference.json) using the Population Stability Index.
# The reference is month-conditional (recent values are scored against the
# same calendar months across all training years) so the strong seasonality
# of PM2.5 doesn't trip the alarm every monsoon.  It also checks recent live
# prediction error against the committed test MAE.  60 days keeps the PSI
# sampling noise well below the warn threshold with quintile bins.
DRIFT_WINDOW_DAYS: int = 60
DRIFT_PSI_WARN: float = 0.10       # conventional "some shift" threshold
DRIFT_PSI_ALERT: float = 0.25      # conventional "significant shift" threshold
DRIFT_ERROR_RATIO_ALERT: float = 1.5  # recent MAE > 1.5 × test MAE → alert

# ── MLOps: walk-forward backtesting ──────────────────────────────────────────
# src/backtest.py refits the model repeatedly through history: first train on
# BACKTEST_INITIAL_TRAIN_DAYS, evaluate the next BACKTEST_STEP_DAYS, then roll
# the window forward and repeat — a far more robust performance estimate than
# the single 90-day held-out window.
BACKTEST_INITIAL_TRAIN_DAYS: int = 365
BACKTEST_STEP_DAYS: int = 30

# ── MLOps: automated retraining schedule ─────────────────────────────────────
# Cron expression (UTC) consumed by .github/workflows/retrain.yml so the
# schedule lives in config, not the workflow.  The workflow reads this value
# at run time; changing the cadence here keeps the workflow generic.
# Default: 03:00 UTC every Monday (weekly).
RETRAIN_CRON: str = "0 3 * * 1"

# Ensure output directories exist on import
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
