"""
Tests for the feature engineering — the leakage guarantee and the
train/inference column alignment that everything else depends on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import (
    TARGET_COLUMN,
    add_target,
    build_supervised,
    feature_names,
    make_features,
)


# ── Leakage ──────────────────────────────────────────────────────────────────
def test_no_future_leakage_by_perturbation(ramp_raw: pd.DataFrame) -> None:
    """No feature on day ``t`` may depend on any value from day > ``t``.

    We build features once, then bump a *single future* PM2.5 value and rebuild.
    A leakage-free transform means every feature row strictly *before* that
    perturbed day is byte-for-byte identical; only rows on/after it may move.
    """
    baseline = make_features(ramp_raw)

    # Pick a day well inside the frame and perturb only that (future) point.
    perturb_pos = 60
    perturbed_raw = ramp_raw.copy()
    perturbed_raw.iloc[perturb_pos, perturbed_raw.columns.get_loc("pm2_5_mean")] += 1000.0
    perturbed = make_features(perturbed_raw)

    perturb_date = ramp_raw.index[perturb_pos]

    # Every feature row for a day strictly before the perturbed day must be
    # unchanged — otherwise a feature "saw" a future value.
    before = baseline.index < perturb_date
    pd.testing.assert_frame_equal(
        baseline.loc[before],
        perturbed.loc[before],
        obj="features for days before the perturbed (future) day",
    )


def test_lags_are_strictly_backward_looking(ramp_raw: pd.DataFrame) -> None:
    """On a 1,2,3,… ramp, ``pm25_lagK`` on day t equals today's value minus K."""
    feats = make_features(ramp_raw)
    pm = ramp_raw["pm2_5_mean"]
    for lag in (1, 2, 3, 7, 14):
        col = f"pm25_lag{lag}"
        expected = pm.shift(lag)
        pd.testing.assert_series_equal(
            feats[col], expected, check_names=False, obj=col
        )


def test_target_is_next_day_only(ramp_raw: pd.DataFrame) -> None:
    """The target for day t is exactly day t+1's PM2.5, and the last is NaN."""
    target = add_target(ramp_raw)
    pm = ramp_raw["pm2_5_mean"]
    assert target.name == TARGET_COLUMN
    pd.testing.assert_series_equal(
        target, pm.shift(-1), check_names=False, obj="target"
    )
    assert np.isnan(target.iloc[-1]), "final target must be NaN (no known future)"


# ── Alignment ────────────────────────────────────────────────────────────────
def test_make_features_columns_match_feature_names(seasonal_raw: pd.DataFrame) -> None:
    """list(make_features(df).columns) == feature_names() — exact order & set."""
    assert list(make_features(seasonal_raw).columns) == feature_names()


def test_feature_count_is_26(seasonal_raw: pd.DataFrame) -> None:
    """Guardrail against silent feature drift (README/plan document 26)."""
    assert len(feature_names()) == 26
    assert make_features(seasonal_raw).shape[1] == 26


# ── build_supervised ─────────────────────────────────────────────────────────
def test_build_supervised_has_no_nans_and_is_aligned(seasonal_raw: pd.DataFrame) -> None:
    """X and y have no NaNs, share an index, and X's columns are canonical."""
    X, y = build_supervised(seasonal_raw)

    assert not X.isnull().any().any(), "X contains NaNs"
    assert not y.isnull().any(), "y contains NaNs"
    assert X.index.equals(y.index), "X and y indices are not aligned"
    assert list(X.columns) == feature_names(), "X columns drifted from feature_names()"
    assert len(X) == len(y) > 0


def test_build_supervised_target_matches_next_day(seasonal_raw: pd.DataFrame) -> None:
    """Each retained row's target equals the raw PM2.5 of the following day."""
    X, y = build_supervised(seasonal_raw)
    pm = seasonal_raw["pm2_5_mean"]
    for day, target_value in y.items():
        next_day = day + pd.Timedelta(days=1)
        assert next_day in pm.index
        assert target_value == pm.loc[next_day]
