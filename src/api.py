"""
FastAPI serving layer for next-day PM2.5 forecasting.

Endpoints
---------
``GET /health``
    Liveness + readiness probe.  Reports whether the model artifact is
    present and loadable (so a missing model surfaces as ``"degraded"``
    rather than a 500 on first prediction).

``GET /predict``
    Pull live recent weather + PM2.5 for the configured city, build a single
    feature row with the *same* :func:`src.features.make_features` used in
    training, run the model, and return tomorrow's predicted PM2.5 alongside
    the exact input features and a model version string.

The actual fetch → feature → predict path lives in :mod:`src.inference` and
is shared with the Streamlit dashboard, so the API does not reimplement any
data or feature logic.

Run::

    uvicorn src.api:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make `import config` / `from src...` work regardless of launch directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from src.aqi import classify  # noqa: E402
from src.inference import (  # noqa: E402
    MODEL_PATH,
    ModelNotFoundError,
    load_model,
    model_version,
    predict_next_day,
)


# ── Structured JSON logging (8.5) ────────────────────────────────────────────
# One JSON object per line so a hosting platform's log collector can parse
# fields (level, message, timestamp) instead of scraping free text.
class _JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": self.formatTime(record),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            }
        )


_handler = logging.StreamHandler()
_handler.setFormatter(_JsonLogFormatter())
logging.basicConfig(level=logging.INFO, handlers=[_handler], force=True)
logger = logging.getLogger(__name__)

# ── Live-prediction cache (8.4) ──────────────────────────────────────────────
# Live data only refreshes daily, so we serve a cached prediction for a short
# window instead of hitting Open-Meteo on every request.  This doubles as a
# fallback: if a later fetch fails, we can still return the last good result.
_CACHE_TTL_SECONDS: int = 900  # 15 minutes
_prediction_cache: dict[str, object] = {"result": None, "at": 0.0}

app = FastAPI(
    title="PM2.5 Next-Day Forecast API",
    version="1.0.0",
    description=(
        "Predicts tomorrow's mean PM2.5 for the configured city using live "
        "Open-Meteo weather + air-quality data and a trained LightGBM model."
    ),
)


# ── Response models ──────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str = Field(..., description="'ok' if the model is loadable, else 'degraded'.")
    city: str
    model_present: bool
    model_path: str
    model_version: str | None = None
    detail: str | None = None


class PredictResponse(BaseModel):
    city: str
    latitude: float
    longitude: float
    feature_date: str = Field(..., description="The day whose features were used (today / latest complete day).")
    prediction_date: str = Field(..., description="The day the PM2.5 value is predicted for (tomorrow).")
    predicted_pm25: float = Field(..., description="Predicted next-day mean PM2.5 (point / p50 forecast).")
    pm25_lower: float | None = Field(
        None, description="Lower bound of the prediction interval (e.g. p10); null if the model has no quantile band."
    )
    pm25_upper: float | None = Field(
        None, description="Upper bound of the prediction interval (e.g. p90); null if the model has no quantile band."
    )
    interval_coverage: float | None = Field(
        None, description="Nominal coverage of [pm25_lower, pm25_upper], e.g. 0.8 for an 80% band."
    )
    aqi_category: str = Field(..., description="Plain-English air-quality band, e.g. 'Moderate'.")
    aqi_advice: str = Field(..., description="Short health guidance for that band.")
    unhealthy: bool = Field(..., description="True if the forecast is unhealthy for sensitive groups or worse.")
    units: str
    model_version: str
    features: dict[str, float] = Field(..., description="The exact feature row fed to the model.")


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + readiness: confirms the model artifact is present/loadable."""
    if not MODEL_PATH.exists():
        return HealthResponse(
            status="degraded",
            city=config.CITY_NAME,
            model_present=False,
            model_path=str(MODEL_PATH),
            detail=(
                f"Model file not found at '{MODEL_PATH}'. "
                "Run `python -m src.train` to create it."
            ),
        )
    try:
        bundle = load_model()
    except Exception as exc:  # malformed artifact, bad pickle, etc.
        return HealthResponse(
            status="degraded",
            city=config.CITY_NAME,
            model_present=True,
            model_path=str(MODEL_PATH),
            detail=f"Model present but failed to load: {exc}",
        )
    return HealthResponse(
        status="ok",
        city=config.CITY_NAME,
        model_present=True,
        model_path=str(MODEL_PATH),
        model_version=model_version(bundle),
    )


# Readiness alias so a host's readiness probe can point at a dedicated path.
@app.get("/ready", response_model=HealthResponse)
def ready() -> HealthResponse:
    """Readiness probe — same check as /health (model present & loadable)."""
    return health()


@app.get("/predict", response_model=PredictResponse)
def predict() -> PredictResponse:
    """Predict tomorrow's PM2.5 for the configured city from live data.

    Serves a cached result for :data:`_CACHE_TTL_SECONDS` to avoid re-fetching
    on every call.  If a fresh fetch fails (e.g. Open-Meteo is down) but a
    cached result exists, the stale result is returned instead of erroring.
    """
    now = time.time()
    cached = _prediction_cache["result"]
    if cached is not None and now - float(_prediction_cache["at"]) < _CACHE_TTL_SECONDS:
        return cached  # type: ignore[return-value]

    try:
        result = predict_next_day()
    except ModelNotFoundError as exc:
        # 503: the service can't serve until the model is trained/present.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        # 422: live data was insufficient to build a complete feature row.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # upstream API failure, network error, etc.
        logger.exception("Prediction failed")
        if cached is not None:
            logger.warning("Serving stale cached prediction after fetch failure.")
            return cached  # type: ignore[return-value]
        raise HTTPException(
            status_code=502,
            detail=f"Failed to produce a prediction from live data: {exc}",
        ) from exc

    category = classify(result.predicted_pm25)
    response = PredictResponse(
        city=result.city,
        latitude=result.latitude,
        longitude=result.longitude,
        feature_date=result.feature_date.isoformat(),
        prediction_date=result.prediction_date.isoformat(),
        predicted_pm25=result.predicted_pm25,
        pm25_lower=result.pm25_lower,
        pm25_upper=result.pm25_upper,
        interval_coverage=result.interval_coverage,
        aqi_category=category.name,
        aqi_advice=category.advice,
        unhealthy=category.unhealthy,
        units=result.units,
        model_version=result.model_version,
        features=result.features,
    )
    _prediction_cache["result"] = response
    _prediction_cache["at"] = now
    return response
