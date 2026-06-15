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

# Ensure output directories exist on import
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
