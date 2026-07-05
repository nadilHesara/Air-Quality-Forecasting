"""
Shared inference path for next-day PM2.5 forecasting.

This module is the **single serving-side entry point** used by both the
FastAPI app (:mod:`src.api`) and the Streamlit dashboard
(:mod:`src.dashboard`).  Keeping the prediction logic here (rather than
duplicating it in each surface) guarantees the API and the dashboard agree
on exactly how a prediction is produced.

The path is deliberately schema-faithful to training:

1. Pull recent daily **weather** and **PM2.5/PM10** for the configured city
   from the Open-Meteo *forecast* endpoints (which, unlike the archive API,
   include the most recent days up to today and tomorrow's weather).  We
   reuse the project's existing fetch functions verbatim — see
   :func:`src.fetch_weather.fetch_daily_weather` and
   :func:`src.fetch_air_quality.fetch_daily_air_quality` — so there is no
   second, drifting copy of the ingestion logic.
2. Inner-join on ``date`` (identical to ``src/build_dataset.py``).
3. Build features with the *same* :func:`src.features.make_features` used at
   training time, so the column set and semantics cannot drift.
4. Take the **last fully-populated feature row** (the latest day ``t`` with
   enough history) and ask the model for ``pm25_next_day`` — i.e. the
   prediction for **tomorrow** (day ``t + 1``).

Note on "tomorrow's weather": the model was trained to predict tomorrow's
PM2.5 from information available *today* (today's weather + pollution and
their lags/rolling windows).  It therefore consumes the feature row for the
latest complete day; tomorrow's forecasted weather is fetched (so the daily
frame extends through tomorrow) but is not itself a model input, matching
how the model was trained.  This is intentional and avoids leakage / schema
drift.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Allow `import config` / `from src...` whether imported as a package or run
# from a different working directory (e.g. `streamlit run src/dashboard.py`).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from src.features import make_features  # noqa: E402
from src.fetch_air_quality import fetch_daily_air_quality  # noqa: E402
from src.fetch_weather import fetch_daily_weather  # noqa: E402

# ── Live (forecast) endpoints ────────────────────────────────────────────────
# The archive endpoints in config.py lag by several days, so they cannot see
# "today".  The forecast endpoints share the exact same response schema and
# accept the same start_date/end_date window, which lets us reuse the existing
# fetch functions unchanged while still getting recent + next-day data.
WEATHER_FORECAST_API_URL: str = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_FORECAST_API_URL: str = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
)

# Days of history to request.  The deepest feature lookback is the 30-day
# rolling window plus the 14-day lag; we pull a generous warm-up margin so the
# final row is always fully populated even with a few missing recent days.
HISTORY_DAYS: int = 60

MODEL_PATH: Path = config.PROJECT_ROOT / "models" / "model.joblib"


class ModelNotFoundError(FileNotFoundError):
    """Raised when the serialized model artifact is missing on disk."""


@dataclass
class PredictionResult:
    """Structured output of a single next-day PM2.5 prediction."""

    city: str
    latitude: float
    longitude: float
    feature_date: date          # the day `t` whose features were used
    prediction_date: date       # day `t + 1`, what we predicted PM2.5 for
    predicted_pm25: float
    features: dict[str, float]  # the exact feature row fed to the model
    model_version: str
    units: str = "µg/m³"
    history: list[dict[str, Any]] = field(default_factory=list)  # recent actuals
    # Prediction interval (from the bundled quantile models).  ``None`` when the
    # loaded model predates quantile support, so older artifacts still serve a
    # point forecast.
    pm25_lower: float | None = None      # lower quantile (e.g. p10)
    pm25_upper: float | None = None      # upper quantile (e.g. p90)
    interval_coverage: float | None = None  # nominal band width, e.g. 0.8
    # Top feature contributions to this prediction ("why this number"), each
    # {feature, value, impact}; empty when SHAP is unavailable.
    explanation: list[dict[str, Any]] = field(default_factory=list)


def _today_utc() -> date:
    return datetime.now(UTC).date()


def load_model(model_path: Path = MODEL_PATH) -> dict[str, Any]:
    """Load the serving model bundle, with a clear error if it's missing.

    Two sources, selected by the ``MODEL_SOURCE`` environment variable:

    - ``local`` (default) — the committed ``models/model.joblib``.  Works with
      zero infrastructure.
    - ``registry`` — the current **production** version in the MLflow Model
      Registry (resolved via the ``@production`` alias; requires
      ``MLFLOW_TRACKING_URI`` to point at a registry-capable backend — see
      ``src/registry.py`` and the README).

    Returns the dict persisted by ``src/train.py``:
    ``{"model", "feature_names", "target", "trained_through", ...}``.
    """
    import os

    if os.environ.get("MODEL_SOURCE", "local").strip().lower() == "registry":
        from src.registry import load_production_bundle

        bundle = load_production_bundle()
    else:
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Train the model first with `python -m src.train` "
                "(which requires `python -m src.build_dataset` to have run)."
            )
        bundle = joblib.load(model_path)
    required = {"model", "feature_names", "target"}
    missing = required - set(bundle)
    if missing:
        raise ValueError(
            f"Model bundle is malformed; missing key(s): {sorted(missing)}."
        )
    return bundle


def model_version(bundle: dict[str, Any]) -> str:
    """Derive a human-readable model version string from the artifact."""
    model_type = type(bundle["model"]).__name__
    trained_through = bundle.get("trained_through", "unknown")
    n_features = len(bundle["feature_names"])
    return f"{model_type}@{trained_through}-{n_features}feat"


def predict_interval(
    bundle: dict[str, Any], feature_row: pd.DataFrame
) -> tuple[float | None, float | None, float | None]:
    """Compute the (lower, upper, nominal_coverage) band for one feature row.

    Reads the quantile models + conformal offsets the training run bundled
    (see :func:`src.train.train_quantile_models`) and applies the same
    calibration: lower levels shift down, upper levels shift up, then the row's
    quantiles are sorted so the band can't invert.  Returns ``(None, None,
    None)`` for older artifacts that have no quantile models, so the serving
    path degrades to a point forecast instead of erroring.
    """
    qmodels: dict[float, Any] = bundle.get("quantile_models") or {}
    if not qmodels:
        return None, None, None
    offsets: dict[float, float] = bundle.get("quantile_offsets") or {}
    levels = sorted(qmodels)

    preds = []
    for a in levels:
        p = float(qmodels[a].predict(feature_row)[0])
        off = offsets.get(a, 0.0)
        p = p - off if a < 0.5 else p + off
        preds.append(p)
    preds.sort()  # guarantee lower <= ... <= upper

    lower, upper = preds[0], preds[-1]
    coverage = float(levels[-1] - levels[0])
    return round(lower, 2), round(upper, 2), coverage


def explain_prediction(
    bundle: dict[str, Any], feature_row: pd.DataFrame, top_n: int = 5
) -> list[dict[str, Any]]:
    """Explain one prediction: which features pushed it up or down, and by how much.

    Uses SHAP on the point model.  Each returned item is one feature with its
    actual value and its SHAP contribution in µg/m³ (positive = pushed the
    prediction *up*, negative = pushed it *down*), sorted by size so the
    dashboard can show the top few drivers of "why this number".

    Returns an empty list if SHAP isn't available or fails, so the serving
    path never breaks just because the explanation couldn't be built.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(bundle["model"])
        shap_values = explainer.shap_values(feature_row)[0]  # one row
    except Exception:  # shap missing, unsupported model, etc.
        return []

    row = feature_row.iloc[0]
    contributions = [
        {
            "feature": name,
            "value": round(float(row[name]), 3),
            "impact": round(float(shap_val), 3),  # µg/m³, signed
        }
        for name, shap_val in zip(feature_row.columns, shap_values, strict=False)
    ]
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return contributions[:top_n]


def fetch_recent_daily(
    *,
    latitude: float = config.LATITUDE,
    longitude: float = config.LONGITUDE,
    history_days: int = HISTORY_DAYS,
    today: date | None = None,
) -> pd.DataFrame:
    """Fetch and join recent daily weather + PM2.5/PM10 for the city.

    Reuses the project's existing fetch functions (pointed at the live
    forecast endpoints) and joins them exactly as ``src/build_dataset.py``
    does, returning the raw daily table that :func:`make_features` expects.
    """
    today = today or _today_utc()
    start = (today - timedelta(days=history_days)).isoformat()
    # Extend through tomorrow so the daily frame includes the forecast day.
    end = (today + timedelta(days=1)).isoformat()

    weather_df = fetch_daily_weather(
        latitude=latitude,
        longitude=longitude,
        start_date=start,
        end_date=end,
        api_url=WEATHER_FORECAST_API_URL,
        daily_variables=config.DAILY_WEATHER_VARIABLES,
    )
    aq_df = fetch_daily_air_quality(
        latitude=latitude,
        longitude=longitude,
        start_date=start,
        end_date=end,
        api_url=AIR_QUALITY_FORECAST_API_URL,
        hourly_variables=config.HOURLY_AQ_VARIABLES,
    )

    # Same inner-join contract as the training pipeline.
    df = weather_df.join(aq_df, how="inner").sort_index()
    return df


def predict_next_day(
    *,
    bundle: dict[str, Any] | None = None,
    latitude: float = config.LATITUDE,
    longitude: float = config.LONGITUDE,
    city: str = config.CITY_NAME,
    history_days: int = HISTORY_DAYS,
    history_tail: int = 30,
    today: date | None = None,
) -> PredictionResult:
    """Run the full serving path and return a structured prediction.

    Parameters
    ----------
    bundle :
        A pre-loaded model bundle.  If ``None``, the model is loaded from
        :data:`MODEL_PATH` (raising :class:`ModelNotFoundError` if absent).
    history_tail :
        How many recent days of *actual* PM2.5 to include in the result's
        ``history`` (used by the dashboard chart).

    Raises
    ------
    ModelNotFoundError
        If the model artifact is missing.
    ValueError
        If live data is insufficient to build a complete feature row.
    """
    bundle = bundle if bundle is not None else load_model()
    model = bundle["model"]
    feature_names: list[str] = list(bundle["feature_names"])

    raw = fetch_recent_daily(
        latitude=latitude,
        longitude=longitude,
        history_days=history_days,
        today=today,
    )
    if raw.empty:
        raise ValueError(
            "No daily data returned from the live APIs for the requested "
            "window; cannot build a feature row."
        )

    feats = make_features(raw)

    # The model needs a fully-populated row; drop rows with any NaN (early
    # warm-up days) and take the most recent complete day as `t`.
    complete = feats.dropna(subset=feature_names)
    if complete.empty:
        raise ValueError(
            f"Insufficient recent history to populate all {len(feature_names)} "
            f"features (got {len(raw)} raw days). Try increasing history_days."
        )

    feature_row = complete.iloc[[-1]][feature_names]
    feature_date: date = feature_row.index[-1].date()
    prediction_date = feature_date + timedelta(days=1)

    predicted = float(model.predict(feature_row)[0])
    lower, upper, coverage = predict_interval(bundle, feature_row)
    explanation = explain_prediction(bundle, feature_row)

    # Recent actual PM2.5 for the dashboard chart.
    pm_actual = raw["pm2_5_mean"].dropna().tail(history_tail)
    history = [
        {"date": idx.date().isoformat(), "pm2_5": float(val)}
        for idx, val in pm_actual.items()
    ]

    return PredictionResult(
        city=city,
        latitude=latitude,
        longitude=longitude,
        feature_date=feature_date,
        prediction_date=prediction_date,
        predicted_pm25=round(predicted, 2),
        features={k: float(v) for k, v in feature_row.iloc[0].items()},
        model_version=model_version(bundle),
        history=history,
        pm25_lower=lower,
        pm25_upper=upper,
        interval_coverage=coverage,
        explanation=explanation,
    )
