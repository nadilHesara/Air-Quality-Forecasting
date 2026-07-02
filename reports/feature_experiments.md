# Phase 6.2 — Feature experiments (leakage-guarded)

Baseline (26 features) CV MAE: **5.002 ± 2.224** (expanding-window TimeSeriesSplit, train pool only).

Each candidate is additive on top of the baseline features, passed the 5.3 perturbation leakage test, and was scored with the identical CV. Δ is candidate CV MAE minus baseline (negative = improvement).

| Candidate | Δ CV MAE vs baseline |
|---|---:|
| wind_vector (sin/cos) | -0.060 |
| cyclical_calendar | +0.015 |
| weather_interactions | +0.004 |
| more_lags (4/5/21/28) | -0.065 |
| pm10_pm25_ratio | +0.019 |
| lagged_weather | +0.024 |

**Keepers promoted to `src/features.py`:** wind_vector (sin/cos), more_lags (4/5/21/28).

_Note: on a series with lag-1 autocorrelation ≈ 0.83 the deltas are small by nature; we keep only features with a consistent (non-noise) CV improvement and re-confirm on the held-out test once, in `src/train.py`._
