"""
Shared harness for the Phase 6 experiments.

Provides three things every experiment needs, defined *once* so a feature
experiment, a tuning run, and a model comparison all evaluate identically and
comparably:

* :func:`load_raw` — the cached training parquet.
* :func:`chrono_split` — the *exact* chronological train/test split
  ``src/train.py`` uses (most-recent ``TEST_HORIZON_DAYS`` held out).
* :func:`cv_mae` — expanding-window :class:`TimeSeriesSplit` MAE/RMSE on a given
  ``(X, y)``, **train-pool only**, with the same fold count as production.
* :func:`assert_no_leakage` — a reusable version of the 5.3 perturbation test:
  bump one *future* PM2.5 value and confirm no feature row *before* it changes.

The golden rule of the whole phase lives here: **model selection (feature
choice, CV, tuning) touches the train pool only; the held-out test set is scored
exactly once, at the very end.**
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sklearn.metrics import mean_absolute_error, mean_squared_error  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

import config  # noqa: E402

TEST_HORIZON_DAYS: int = config.TEST_HORIZON_DAYS
N_CV_SPLITS: int = config.N_CV_SPLITS


def load_raw() -> pd.DataFrame:
    parquet = config.DATA_DIR / "training_data.parquet"
    if not parquet.exists():
        raise SystemExit(f"Missing {parquet}; run `python -m src.train` first.")
    return pd.read_parquet(parquet).sort_index()


def chrono_split(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return (X_train, y_train, X_test, y_test) with train.py's cutoff."""
    cutoff = X.index.max() - timedelta(days=TEST_HORIZON_DAYS)
    train = X.index <= cutoff
    test = X.index > cutoff
    return X[train], y[train], X[test], y[test]


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def cv_mae(
    make_model: Callable[[], object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    n_splits: int = N_CV_SPLITS,
) -> dict[str, float]:
    """Expanding-window TimeSeriesSplit CV MAE/RMSE on the train pool only.

    ``make_model`` is a zero-arg factory returning a fresh, unfitted estimator
    each fold (so folds never share fitted state).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses = [], []
    for tr_idx, va_idx in tscv.split(X_train):
        X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
        y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
        model = make_model()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_va)
        maes.append(mean_absolute_error(y_va, pred))
        rmses.append(rmse(y_va, pred))
    return {
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
    }


def assert_no_leakage(
    build_features: Callable[[pd.DataFrame], pd.DataFrame],
    raw: pd.DataFrame,
    *,
    perturb_pos: int = 60,
) -> None:
    """Reusable 5.3 leakage guard for an arbitrary feature builder.

    Perturb a single *future* PM2.5 value and require every feature row for a
    day strictly *before* it to be byte-for-byte identical. Raises
    ``AssertionError`` on any leak.
    """
    baseline = build_features(raw)
    perturbed_raw = raw.copy()
    col = perturbed_raw.columns.get_loc("pm2_5_mean")
    perturbed_raw.iloc[perturb_pos, col] += 1000.0
    perturbed = build_features(perturbed_raw)

    perturb_date = raw.index[perturb_pos]
    before = baseline.index < perturb_date
    pd.testing.assert_frame_equal(
        baseline.loc[before],
        perturbed.loc[before],
        obj="feature rows before the perturbed future day",
    )
