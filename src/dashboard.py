"""
Minimal Streamlit dashboard for next-day PM2.5 forecasting.

Shows, for the configured city:

* tomorrow's predicted mean PM2.5 (a single headline metric), and
* a chart of the last ~30 days of *actual* PM2.5 with the latest prediction
  overlaid as a forward point.

The dashboard calls the **same** prediction path as the API
(:func:`src.inference.predict_next_day`), so what you see here is exactly
what ``GET /predict`` would return — no duplicated data or model logic.

Run::

    streamlit run src/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make `import config` / `from src...` work under `streamlit run`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402
from src.inference import ModelNotFoundError, predict_next_day  # noqa: E402

st.set_page_config(page_title="PM2.5 Next-Day Forecast", page_icon="🌫️", layout="centered")

st.title("🌫️ Next-Day PM2.5 Forecast")
st.caption(f"City: **{config.CITY_NAME}**  ·  Source: Open-Meteo (live)  ·  Model: LightGBM")


@st.cache_data(ttl=900, show_spinner="Fetching live data and predicting…")
def _run_prediction() -> dict:
    """Run the shared serving path; cached for 15 min to avoid re-fetching."""
    result = predict_next_day()
    return {
        "city": result.city,
        "feature_date": result.feature_date.isoformat(),
        "prediction_date": result.prediction_date.isoformat(),
        "predicted_pm25": result.predicted_pm25,
        "pm25_lower": result.pm25_lower,
        "pm25_upper": result.pm25_upper,
        "interval_coverage": result.interval_coverage,
        "units": result.units,
        "model_version": result.model_version,
        "features": result.features,
        "history": result.history,
    }


if st.button("🔄 Refresh"):
    _run_prediction.clear()

try:
    res = _run_prediction()
except ModelNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:  # network / upstream API / insufficient data
    st.error(f"Could not produce a prediction: {exc}")
    st.stop()

# ── Headline metric ──────────────────────────────────────────────────────────
history_df = pd.DataFrame(res["history"])
latest_actual = float(history_df["pm2_5"].iloc[-1]) if not history_df.empty else None
delta = (
    round(res["predicted_pm25"] - latest_actual, 2)
    if latest_actual is not None
    else None
)

lower, upper = res.get("pm25_lower"), res.get("pm25_upper")
coverage = res.get("interval_coverage")
has_interval = lower is not None and upper is not None

col1, col2 = st.columns(2)
col1.metric(
    label=f"Predicted PM2.5 for {res['prediction_date']}",
    value=f"{res['predicted_pm25']} {res['units']}",
    delta=f"{delta:+.2f} vs latest actual" if delta is not None else None,
    delta_color="inverse",  # higher PM2.5 is worse → red
)
if has_interval:
    band = f"{coverage * 100:.0f}%" if coverage else "prediction"
    col1.caption(f"{band} interval: **{lower} – {upper}** {res['units']}")
if latest_actual is not None:
    col2.metric(
        label=f"Latest actual ({res['feature_date']})",
        value=f"{round(latest_actual, 2)} {res['units']}",
    )

# ── Chart: last ~30 days actual + tomorrow's prediction overlaid ─────────────
st.subheader("Last 30 days of actual PM2.5 (prediction overlaid)")

if history_df.empty:
    st.info("No recent actual PM2.5 data was returned to plot.")
else:
    history_df["date"] = pd.to_datetime(history_df["date"])
    chart_df = history_df.rename(columns={"pm2_5": "Actual PM2.5"}).set_index("date")

    # Overlay the prediction as a separate series at the prediction date so it
    # renders as a distinct forward point on the same axes.  When the model
    # carries a quantile band, plot the lower/upper bounds as their own forward
    # points so the range is visible alongside the point forecast.
    pred_date = pd.to_datetime(res["prediction_date"])
    chart_df.loc[pred_date, "Predicted PM2.5"] = res["predicted_pm25"]
    y_series = ["Actual PM2.5", "Predicted PM2.5"]
    if has_interval:
        chart_df.loc[pred_date, "Lower bound"] = lower
        chart_df.loc[pred_date, "Upper bound"] = upper
        y_series += ["Lower bound", "Upper bound"]
    chart_df = chart_df.sort_index()

    st.line_chart(chart_df, y=y_series)
    if has_interval:
        st.caption(
            f"Shaded range = {coverage * 100:.0f}% prediction interval "
            f"(p{int((1 - coverage) / 2 * 100)}–p{int((1 - (1 - coverage) / 2) * 100)})."
            if coverage else "Lower/upper bounds shown as forward points."
        )

# ── Details ──────────────────────────────────────────────────────────────────
st.caption(f"Model version: `{res['model_version']}`")
with st.expander("Input features fed to the model"):
    feat_df = (
        pd.Series(res["features"], name="value")
        .rename_axis("feature")
        .reset_index()
    )
    st.dataframe(feat_df, use_container_width=True, hide_index=True)
