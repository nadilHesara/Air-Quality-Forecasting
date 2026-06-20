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

# ── MLOps: experiment tracking (MLflow) ──────────────────────────────────────
# Local, file-based tracking store committed to the repo (mlruns/).  To point
# at a remote tracking server instead, set the MLFLOW_TRACKING_URI environment
# variable (e.g. "http://my-mlflow-server:5000") — it overrides the local
# default below.  See the README ("Pointing MLflow at a remote server").
MLFLOW_TRACKING_URI: str = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME: str = "pm25-next-day-forecast"

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
