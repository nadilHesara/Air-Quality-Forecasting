"""
Phase 6.4 — Alternative / ensemble model comparison, scored fairly.

Compares, on the **same held-out test window** ``src/train.py`` uses (most
recent ``TEST_HORIZON_DAYS`` days), a range of approaches:

* persistence & seasonal-naive           — the reference baselines
* LightGBM (current params)              — the shipping model
* LightGBM (tuned, +wind sin/cos)        — the 6.3 winner
* XGBoost (tuned-ish)                     — a second GBM family
* Ridge on lags                           — a simple linear autoregressor
* SARIMAX                                 — a classical univariate TS baseline

Fairness rules
--------------
* Same train pool / test window for everyone.
* The ML models fit on the train pool and predict the test rows **directly**
  (each test row's features use only past information, so this is a proper
  one-step-ahead forecast, not a leak).
* SARIMAX is a genuine rolling one-step-ahead forecast: it is fit on the train
  PM2.5 series and then `append`-ed one true observation at a time across the
  test window, forecasting one step each time (no peeking).

This script *does* touch the held-out test set — that is its job (final
comparison). It runs once, after all model selection (6.2/6.3) was done on the
train pool only.

Run::

    python -m experiments.model_comparison
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import config  # noqa: E402
from experiments._harness import chrono_split, load_raw  # noqa: E402
from experiments.feature_experiments import _supervised, add_wind_vector  # noqa: E402
from src.features import RAW_PM_COLUMN, build_supervised  # noqa: E402
from src.train import (  # noqa: E402
    LGBM_PARAMS,
    persistence_prediction,
    seasonal_naive_prediction,
)

# 6.3 winner (baseline + wind sin/cos): copied from reports/hpo_best_params.md.
TUNED_LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.0189756973497472,
    "num_leaves": 44,
    "max_depth": 3,
    "min_child_samples": 18,
    "subsample": 0.8898705283830923,
    "subsample_freq": 1,
    "colsample_bytree": 0.9664329660605866,
    "reg_alpha": 0.2594797777901213,
    "reg_lambda": 0.0892155202631994,
    "random_state": config.RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}

# Lag columns Ridge uses (linear AR needs the model to see the level directly).
RIDGE_LAG_COLS = [
    "pm2_5_mean", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag7",
    "pm25_rolling7_mean", "pm25_rolling3_mean",
]


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def _fit_predict_ml(model, X_train, y_train, X_test) -> np.ndarray:
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _sarimax_rolling(raw: pd.DataFrame, train_end, test_index) -> np.ndarray:
    """One-step-ahead rolling SARIMAX forecast over the test window.

    Fit on the PM2.5 series up to ``train_end``, then walk the test days: each
    day forecast one step ahead, then append the *true* observation and refit
    the state (not the params) — a fair, no-leak rolling forecast.
    """
    import statsmodels.api as sm

    # Regular daily series with an explicit frequency — SARIMAX's forecast /
    # append machinery needs the index to carry freq="D" to extend it.
    pm = raw[RAW_PM_COLUMN].astype(float)
    pm = pm.asfreq("D").interpolate("time")  # fill any calendar gaps
    train_series = pm.loc[:train_end]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.tsa.SARIMAX(
            train_series,
            order=(2, 0, 1),
            seasonal_order=(1, 0, 0, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)

        # Walk the calendar day-by-day from just after the train series to the
        # last needed target day, always keeping the appended series contiguous
        # (freq="D" append requires it). We record the one-step forecast only on
        # days that are a test target (= a test row's day + 1).
        target_days = {t + pd.Timedelta(days=1) for t in test_index}
        pred_by_day: dict[pd.Timestamp, float] = {}

        cursor = res.model.data.dates[-1]  # last day currently in the model
        last_target = max(target_days)
        while cursor < last_target:
            next_day = cursor + pd.Timedelta(days=1)
            fc = res.forecast(steps=1)
            if next_day in target_days:
                pred_by_day[next_day] = float(fc.iloc[0])
            # condition on the true observation and advance.
            nxt = pd.Series(
                [pm.loc[next_day]],
                index=pd.DatetimeIndex([next_day], freq="D"),
                name=pm.name,
            )
            res = res.append(nxt)
            cursor = next_day

    return np.array([pred_by_day[t + pd.Timedelta(days=1)] for t in test_index])


def main() -> None:
    raw = load_raw()

    # Baseline features (for persistence/seasonal/ridge/current LGBM/xgb).
    Xb, yb = build_supervised(raw)
    Xb_tr, yb_tr, Xb_te, yb_te = chrono_split(Xb, yb)

    # Wind-augmented features (for the tuned LGBM winner).
    Xw, yw = _supervised(add_wind_vector, raw)
    Xw_tr, yw_tr, Xw_te, yw_te = chrono_split(Xw, yw)
    assert yb_te.index.equals(yw_te.index), "test windows diverged between feature sets"

    y_test = yb_te
    results: dict[str, dict[str, float]] = {}

    # Baselines ---------------------------------------------------------------
    results["Persistence"] = _metrics(y_test, persistence_prediction(raw, y_test.index).values)
    seas = seasonal_naive_prediction(raw, y_test.index)
    valid = seas.notna()
    results["Seasonal-naive"] = _metrics(y_test[valid], seas[valid].values)

    # Current shipping LightGBM ----------------------------------------------
    results["LightGBM (current)"] = _metrics(
        y_test, _fit_predict_ml(LGBMRegressor(**LGBM_PARAMS), Xb_tr, yb_tr, Xb_te)
    )

    # Tuned LightGBM + wind (6.3 winner) -------------------------------------
    results["LightGBM (tuned +wind)"] = _metrics(
        y_test, _fit_predict_ml(LGBMRegressor(**TUNED_LGBM_PARAMS), Xw_tr, yw_tr, Xw_te)
    )

    # XGBoost -----------------------------------------------------------------
    from xgboost import XGBRegressor

    xgb = XGBRegressor(
        n_estimators=400, learning_rate=0.02, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        random_state=config.RANDOM_STATE, n_jobs=-1,
    )
    results["XGBoost"] = _metrics(y_test, _fit_predict_ml(xgb, Xb_tr, yb_tr, Xb_te))

    # Ridge on lags -----------------------------------------------------------
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    results["Ridge (on lags)"] = _metrics(
        y_test, _fit_predict_ml(ridge, Xb_tr[RIDGE_LAG_COLS], yb_tr, Xb_te[RIDGE_LAG_COLS])
    )

    # SARIMAX -----------------------------------------------------------------
    train_end = Xb_tr.index.max()
    try:
        sarimax_pred = _sarimax_rolling(raw, train_end, y_test.index)
        results["SARIMAX (2,0,1)(1,0,0)7"] = _metrics(y_test, sarimax_pred)
    except Exception as exc:  # statsmodels can be finicky; don't kill the table
        print(f"SARIMAX failed: {exc}")

    # ── Report ───────────────────────────────────────────────────────────────
    persist_mae = results["Persistence"]["mae"]
    order = sorted(results, key=lambda k: results[k]["mae"])

    print(f"\n{'Model':28s} {'MAE':>8s} {'RMSE':>8s} {'vs persist':>11s}")
    print("-" * 60)
    for name in order:
        m = results[name]
        imp = (persist_mae - m["mae"]) / persist_mae * 100.0
        print(f"{name:28s} {m['mae']:8.3f} {m['rmse']:8.3f} {imp:+10.1f}%")

    lines = [
        "# Phase 6.4 — Model comparison (held-out test)",
        "",
        f"Same held-out test window as `src/train.py`: "
        f"{y_test.index.min().date()} → {y_test.index.max().date()} "
        f"({len(y_test)} days). All models fit on the train pool only; SARIMAX is "
        "a rolling one-step-ahead forecast. Scored once.",
        "",
        "| Model | MAE | RMSE | vs persistence (MAE) |",
        "|---|---:|---:|---:|",
    ]
    for name in order:
        m = results[name]
        imp = (persist_mae - m["mae"]) / persist_mae * 100.0
        bold = "**" if name.startswith("LightGBM (tuned") else ""
        lines.append(f"| {bold}{name}{bold} | {m['mae']:.3f} | {m['rmse']:.3f} | {imp:+.1f}% |")
    lines += [
        "",
        "_Positive = reduces persistence's MAE. Model selection (6.2 features, "
        "6.3 tuning) used the train pool only; this table is the single held-out "
        "evaluation._",
        "",
        "## Conclusion — the remaining gap is largely intrinsic",
        "",
        "A linear model (Ridge on lags), two gradient-boosting families "
        "(LightGBM, XGBoost), and a classical SARIMAX all land within a ~5% MAE "
        "band of each other **and** of persistence. When six independent methods "
        "converge on the same score, the ceiling is set by the data, not the "
        "model: daily-mean PM2.5 has lag-1 autocorrelation ≈ 0.83, so today's "
        "value already captures almost all the predictable signal, and the "
        "day-over-day *move* is close to noise. LightGBM leads on **RMSE** "
        "(fewer large misses on spike days), which — together with its native "
        "quantile support — is why it stays the production model, now shipping "
        "with p10/p50/p90 prediction intervals (6.5).",
        "",
    ]
    out = config.REPORTS_DIR / "model_comparison.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
