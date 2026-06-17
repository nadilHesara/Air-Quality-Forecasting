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

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Make `import config` / `from src...` work regardless of launch directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from src.inference import (  # noqa: E402
    ModelNotFoundError,
    MODEL_PATH,
    load_model,
    model_version,
    predict_next_day,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    predicted_pm25: float = Field(..., description="Predicted next-day mean PM2.5.")
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


@app.get("/predict", response_model=PredictResponse)
def predict() -> PredictResponse:
    """Predict tomorrow's PM2.5 for the configured city from live data."""
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
        raise HTTPException(
            status_code=502,
            detail=f"Failed to produce a prediction from live data: {exc}",
        ) from exc

    return PredictResponse(
        city=result.city,
        latitude=result.latitude,
        longitude=result.longitude,
        feature_date=result.feature_date.isoformat(),
        prediction_date=result.prediction_date.isoformat(),
        predicted_pm25=result.predicted_pm25,
        units=result.units,
        model_version=result.model_version,
        features=result.features,
    )
