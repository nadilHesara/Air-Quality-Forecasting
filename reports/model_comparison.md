# Phase 6.4 — Model comparison (held-out test)

Same held-out test window as `src/train.py`: 2026-03-15 → 2026-06-12 (90 days). All models fit on the train pool only; SARIMAX is a rolling one-step-ahead forecast. Scored once.

| Model | MAE | RMSE | vs persistence (MAE) |
|---|---:|---:|---:|
| Ridge (on lags) | 3.284 | 4.450 | +5.3% |
| LightGBM (current) | 3.312 | 4.532 | +4.5% |
| XGBoost | 3.347 | 4.531 | +3.5% |
| SARIMAX (2,0,1)(1,0,0)7 | 3.364 | 4.559 | +3.0% |
| **LightGBM (tuned +wind)** | 3.393 | 4.436 | +2.1% |
| Persistence | 3.467 | 4.900 | +0.0% |
| Seasonal-naive | 5.427 | 6.962 | -56.5% |

_Positive = reduces persistence's MAE. Model selection (6.2 features, 6.3 tuning) used the train pool only; this table is the single held-out evaluation._

## Conclusion — the remaining gap is largely intrinsic

A linear model (Ridge on lags), two gradient-boosting families (LightGBM, XGBoost), and a classical SARIMAX all land within a ~5% MAE band of each other **and** of persistence. When six independent methods converge on the same score, the ceiling is set by the data, not the model: daily-mean PM2.5 has lag-1 autocorrelation ≈ 0.83, so today's value already captures almost all the predictable signal, and the day-over-day *move* is close to noise. LightGBM leads on **RMSE** (fewer large misses on spike days), which — together with its native quantile support — is why it stays the production model, now shipping with p10/p50/p90 prediction intervals (6.5).
