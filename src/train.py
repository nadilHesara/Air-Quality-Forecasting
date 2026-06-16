"""
Train and honestly evaluate a next-day PM2.5 forecasting model.

Pipeline
--------
1. Load ``data/training_data.parquet``.
2. Build features + target via :mod:`src.features` (the *same* function
   used at inference time).
3. Split **chronologically**: the most recent ~3 months are held out as an
   untouched final test set; everything before is the training pool.
4. Establish two naïve baselines on the test set:
     * persistence  — tomorrow's PM2.5 = today's PM2.5
     * seasonal-naive — tomorrow's PM2.5 = same weekday last week
5. Tune/validate a LightGBM regressor with an expanding-window
   :class:`~sklearn.model_selection.TimeSeriesSplit` on the training pool
   (never shuffling, never peeking at the test set).
6. Refit on the full training pool and evaluate once on the held-out test.
7. Report MAE / RMSE for the model vs both baselines, with % improvement.
8. Persist: ``models/model.joblib``, ``reports/metrics.json``,
   ``reports/results.md``, ``reports/shap_summary.png``.

Run::

    python -m src.train
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Allow running as a module or directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.features import (  # noqa: E402
    RAW_PM_COLUMN,
    TARGET_COLUMN,
    build_supervised,
    feature_names,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────
TEST_HORIZON_DAYS: int = 90          # ~3 months held out as final test set
N_CV_SPLITS: int = 5                 # expanding-window folds on the train pool
RANDOM_STATE: int = 42

MODELS_DIR: Path = config.PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LGBM_PARAMS: dict = {
    "n_estimators": 600,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 5,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbose": -1,
}


# ── Metrics helpers ──────────────────────────────────────────────────────────
def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": _rmse(np.asarray(y_true), np.asarray(y_pred)),
    }


def _pct_improvement(baseline: float, model: float) -> float:
    """Percentage by which ``model`` improves (reduces) ``baseline``."""
    if baseline == 0:
        return float("nan")
    return float((baseline - model) / baseline * 100.0)


# ── Baselines ────────────────────────────────────────────────────────────────
def persistence_prediction(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Persistence baseline: predict tomorrow's PM2.5 = today's PM2.5.

    For a row dated ``t`` (whose target is the PM2.5 of ``t + 1``), the
    prediction is simply today's observed PM2.5.
    """
    return raw.loc[index, RAW_PM_COLUMN].copy()


def seasonal_naive_prediction(raw: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Seasonal-naive baseline: tomorrow's PM2.5 = same weekday last week.

    The target for row ``t`` is the PM2.5 on day ``t + 1``.  "Same weekday
    last week" relative to that target day is the PM2.5 observed 7 days
    before ``t + 1`` — i.e. on day ``t - 6``.  We look this value up *by
    date* (robust to any gaps) and return NaN where unavailable.
    """
    pm = raw[RAW_PM_COLUMN]
    preds = []
    for t in index:
        ref_day = t - timedelta(days=6)  # (t + 1) - 7 days
        preds.append(pm.get(ref_day, np.nan))
    return pd.Series(preds, index=index, name="seasonal_naive")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    # 1. Load raw data --------------------------------------------------------
    parquet_path = config.DATA_DIR / "training_data.parquet"
    if not parquet_path.exists():
        logger.error("%s not found. Run `python -m src.build_dataset` first.", parquet_path)
        sys.exit(1)

    raw = pd.read_parquet(parquet_path)
    raw = raw.sort_index()
    logger.info("Loaded %d daily rows (%s → %s)", len(raw), raw.index.min().date(), raw.index.max().date())

    # 2. Build supervised matrix (same function used at inference) ------------
    X, y = build_supervised(raw)
    assert list(X.columns) == feature_names(), "Feature column order drifted from feature_names()."
    logger.info("Supervised dataset: %d rows × %d features", X.shape[0], X.shape[1])

    # 3. Chronological train / test split ------------------------------------
    cutoff = X.index.max() - timedelta(days=TEST_HORIZON_DAYS)
    train_mask = X.index <= cutoff
    test_mask = X.index > cutoff

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    logger.info(
        "Train: %d rows (%s → %s)   |   Test (held out): %d rows (%s → %s)",
        len(X_train), X_train.index.min().date(), X_train.index.max().date(),
        len(X_test), X_test.index.min().date(), X_test.index.max().date(),
    )

    # 4. Baselines on the held-out test set ----------------------------------
    persist_pred = persistence_prediction(raw, X_test.index)
    seasonal_pred = seasonal_naive_prediction(raw, X_test.index)

    # Seasonal-naive may have NaNs at the very start if history is short; the
    # 30-row feature warm-up means it's fully populated here, but guard anyway.
    valid = seasonal_pred.notna()
    if not valid.all():
        logger.warning("Seasonal-naive has %d NaN prediction(s); excluding from its metrics.", (~valid).sum())

    baseline_persist = _metrics(y_test, persist_pred.values)
    baseline_seasonal = _metrics(y_test[valid], seasonal_pred[valid].values)

    logger.info("Baseline [persistence]     MAE=%.3f  RMSE=%.3f", baseline_persist["mae"], baseline_persist["rmse"])
    logger.info("Baseline [seasonal-naive]  MAE=%.3f  RMSE=%.3f", baseline_seasonal["mae"], baseline_seasonal["rmse"])

    # 5. Time-series cross-validation on the TRAIN pool only ------------------
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
    cv_maes, cv_rmses = [], []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

        fold_model = LGBMRegressor(**LGBM_PARAMS)
        fold_model.fit(X_tr, y_tr)
        va_pred = fold_model.predict(X_va)

        m = _metrics(y_va, va_pred)
        cv_maes.append(m["mae"])
        cv_rmses.append(m["rmse"])
        logger.info(
            "  CV fold %d/%d  train=%d val=%d  MAE=%.3f  RMSE=%.3f",
            fold, N_CV_SPLITS, len(tr_idx), len(va_idx), m["mae"], m["rmse"],
        )

    cv_summary = {
        "mae_mean": float(np.mean(cv_maes)),
        "mae_std": float(np.std(cv_maes)),
        "rmse_mean": float(np.mean(cv_rmses)),
        "rmse_std": float(np.std(cv_rmses)),
    }
    logger.info(
        "CV (expanding window, %d folds)  MAE=%.3f±%.3f  RMSE=%.3f±%.3f",
        N_CV_SPLITS, cv_summary["mae_mean"], cv_summary["mae_std"],
        cv_summary["rmse_mean"], cv_summary["rmse_std"],
    )

    # 6. Refit on the full train pool, evaluate ONCE on the held-out test -----
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test)
    model_test = _metrics(y_test, test_pred)
    logger.info("MODEL (held-out test)      MAE=%.3f  RMSE=%.3f", model_test["mae"], model_test["rmse"])

    # 7. Assemble metrics with % improvement vs baselines --------------------
    metrics = {
        "dataset": {
            "raw_rows": int(len(raw)),
            "supervised_rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "date_range": [str(X.index.min().date()), str(X.index.max().date())],
            "test_horizon_days": TEST_HORIZON_DAYS,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "test_period": [str(X_test.index.min().date()), str(X_test.index.max().date())],
        },
        "validation": {
            "scheme": f"TimeSeriesSplit expanding window, {N_CV_SPLITS} folds (train pool only)",
            **cv_summary,
        },
        "baselines": {
            "persistence": baseline_persist,
            "seasonal_naive": baseline_seasonal,
        },
        "model": {
            "type": "LGBMRegressor",
            "params": LGBM_PARAMS,
            "test": model_test,
        },
        "improvement_vs_baseline": {
            "persistence": {
                "mae_pct": _pct_improvement(baseline_persist["mae"], model_test["mae"]),
                "rmse_pct": _pct_improvement(baseline_persist["rmse"], model_test["rmse"]),
            },
            "seasonal_naive": {
                "mae_pct": _pct_improvement(baseline_seasonal["mae"], model_test["mae"]),
                "rmse_pct": _pct_improvement(baseline_seasonal["rmse"], model_test["rmse"]),
            },
        },
    }

    metrics_path = config.REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Wrote metrics → %s", metrics_path)

    # 8. SHAP summary plot ----------------------------------------------------
    _save_shap_summary(model, X_test)

    # 9. Persist model + feature contract ------------------------------------
    model_path = MODELS_DIR / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names(),
            "target": TARGET_COLUMN,
            "trained_through": str(X_train.index.max().date()),
        },
        model_path,
    )
    logger.info("Saved model → %s", model_path)

    # 10. Human-readable results report --------------------------------------
    _write_results_md(metrics)


def _save_shap_summary(model: LGBMRegressor, X_test: pd.DataFrame) -> None:
    """Compute SHAP values on the test set and save a summary plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False, plot_size=(10, 8))
    plt.title("SHAP feature importance — next-day PM2.5", fontsize=12)
    plt.tight_layout()
    out = config.REPORTS_DIR / "shap_summary.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved SHAP summary → %s", out)


def _write_results_md(metrics: dict) -> None:
    """Render a short Markdown results summary."""
    d = metrics["dataset"]
    b = metrics["baselines"]
    m = metrics["model"]["test"]
    imp = metrics["improvement_vs_baseline"]
    cv = metrics["validation"]

    lines = [
        "# PM2.5 Next-Day Forecast — Results",
        "",
        f"**Target:** `{TARGET_COLUMN}` (next-day mean PM2.5, μg/m³)  ",
        f"**Data:** {d['supervised_rows']} supervised rows "
        f"({d['date_range'][0]} → {d['date_range'][1]}), {d['n_features']} features  ",
        f"**Validation:** {cv['scheme']}  ",
        f"**Final test (held out, untouched):** {d['test_rows']} days "
        f"({d['test_period'][0]} → {d['test_period'][1]})",
        "",
        "## Headline",
        "",
        "| Model | MAE | RMSE |",
        "|---|---:|---:|",
        f"| Persistence (today → tomorrow) | {b['persistence']['mae']:.3f} | {b['persistence']['rmse']:.3f} |",
        f"| Seasonal-naive (same weekday last week) | {b['seasonal_naive']['mae']:.3f} | {b['seasonal_naive']['rmse']:.3f} |",
        f"| **LightGBM** | **{m['mae']:.3f}** | **{m['rmse']:.3f}** |",
        "",
        "## Improvement of LightGBM over baselines",
        "",
        "| Baseline | MAE improvement | RMSE improvement |",
        "|---|---:|---:|",
        f"| vs persistence | {imp['persistence']['mae_pct']:+.1f}% | {imp['persistence']['rmse_pct']:+.1f}% |",
        f"| vs seasonal-naive | {imp['seasonal_naive']['mae_pct']:+.1f}% | {imp['seasonal_naive']['rmse_pct']:+.1f}% |",
        "",
        "_(Positive = the model reduces the baseline's error.)_",
        "",
        "## Cross-validation (train pool only)",
        "",
        f"{cv['scheme']}:",
        "",
        f"- MAE  = {cv['mae_mean']:.3f} ± {cv['mae_std']:.3f}",
        f"- RMSE = {cv['rmse_mean']:.3f} ± {cv['rmse_std']:.3f}",
        "",
        "## Method notes (honesty / leakage)",
        "",
        "- **No shuffling, no random split.** Train/test are split strictly by "
        "date; the most recent ~3 months were untouched during model development.",
        "- **Leakage-safe features.** Every feature for day *t* uses only "
        "information available up to day *t* (lags `shift(k≥1)`, non-centred "
        "rolling windows). The only future-looking column is the target.",
        "- **Same feature function at train and inference.** Features come from "
        "`src.features.make_features`, so production inputs match training inputs.",
        "- **Single test evaluation.** The held-out test set was scored once, "
        "after model selection on the train pool — no test-set tuning.",
        "",
        "![SHAP summary](shap_summary.png)",
        "",
    ]
    out = config.REPORTS_DIR / "results.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote results report → %s", out)


if __name__ == "__main__":
    main()
