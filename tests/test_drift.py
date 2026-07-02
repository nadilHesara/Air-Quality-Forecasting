"""
Tests for drift monitoring (src/drift.py, Phase 7.4).

Offline throughout: reference profiles and recent windows are built from the
synthetic fixtures, and the "model" in the error-drift tests is a stub.  The
contracts under test:

- PSI is ~0 for identical distributions and large for disjoint ones;
- a shifted feature is flagged with status "alert" and trips the report;
- calendar features are never scored (a 30-day window always "drifts"
  against a multi-year calendar distribution);
- recent prediction error is compared against the committed test MAE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.drift import (
    CALENDAR_FEATURES,
    build_reference_profile,
    monitored_features,
    population_stability_index,
    run_drift_check,
    score_feature_drift,
    write_report,
)
from src.features import feature_names, make_features


class _ConstModel:
    """Stub model predicting a constant — enough for the error-drift math."""

    def __init__(self, value: float = 20.0):
        self.value = value

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.value)


def _bundle() -> dict:
    return {
        "model": _ConstModel(),
        "feature_names": feature_names(),
        "target": "pm25_next_day",
    }


def _features(raw: pd.DataFrame) -> pd.DataFrame:
    return make_features(raw).dropna()


# ── PSI ──────────────────────────────────────────────────────────────────────
def test_psi_zero_for_identical_distributions() -> None:
    fractions = np.array([0.1, 0.2, 0.3, 0.4])
    assert population_stability_index(fractions, fractions) == 0.0


def test_psi_large_for_disjoint_distributions() -> None:
    expected = np.array([0.5, 0.5, 0.0, 0.0])
    actual = np.array([0.0, 0.0, 0.5, 0.5])
    assert population_stability_index(expected, actual) > 1.0


# ── Reference profile ────────────────────────────────────────────────────────
def test_reference_profile_structure(seasonal_raw: pd.DataFrame) -> None:
    X = _features(seasonal_raw)
    profile = build_reference_profile(X)

    assert profile["n_rows"] == len(X)
    assert profile["features"], "profile should describe at least one feature"
    # Calendar features are excluded by design.
    assert not CALENDAR_FEATURES & set(profile["features"])
    assert set(profile["features"]).issubset(set(monitored_features()))
    # Bin fractions of a non-constant feature form a distribution.
    pm_ref = profile["features"]["pm2_5_mean"]
    assert not pm_ref["constant"]
    assert math.isclose(sum(pm_ref["global"]["bin_fractions"]), 1.0, rel_tol=1e-9)
    # Months with enough training rows get their own seasonal distribution
    # (the 90-day fixture has full Februaries/Marches after feature warm-up).
    assert pm_ref["by_month"], "expected per-month reference distributions"
    for dist in pm_ref["by_month"].values():
        assert math.isclose(sum(dist["bin_fractions"]), 1.0, rel_tol=1e-9)


def test_constant_feature_marked_constant(seasonal_raw: pd.DataFrame) -> None:
    # The fixture's precipitation is all zeros — PSI would be meaningless.
    X = _features(seasonal_raw)
    profile = build_reference_profile(X)
    assert profile["features"]["precipitation_sum"]["constant"]


# ── Feature-drift scoring ────────────────────────────────────────────────────
def test_same_distribution_is_stable(seasonal_raw: pd.DataFrame) -> None:
    X = _features(seasonal_raw)
    profile = build_reference_profile(X)
    scores = score_feature_drift(profile, X)
    assert all(s["status"] in {"stable", "skipped"} for s in scores)


def test_shifted_feature_alerts(seasonal_raw: pd.DataFrame) -> None:
    X = _features(seasonal_raw)
    profile = build_reference_profile(X)

    shifted = X.copy()
    shifted["pm2_5_mean"] = shifted["pm2_5_mean"] * 3.0  # a 3× pollution jump
    scores = {s["feature"]: s for s in score_feature_drift(profile, shifted)}

    assert scores["pm2_5_mean"]["status"] == "alert"
    assert scores["pm2_5_mean"]["psi"] >= 0.25


# ── End-to-end drift check + report ──────────────────────────────────────────
def test_error_drift_alerts_when_recent_mae_blows_up(seasonal_raw: pd.DataFrame) -> None:
    # Constant-20 stub vs a signal swinging ±8 → recent MAE far above a
    # committed test MAE of 1.0 → the 1.5× threshold must trip.
    report = run_drift_check(
        raw=seasonal_raw,
        profile=None,
        bundle=_bundle(),
        committed_metrics={"model": {"test": {"mae": 1.0}}},
    )
    assert report["error_drift"] is not None
    assert report["error_drift"]["alert"]
    assert report["drift_detected"]


def test_error_drift_quiet_when_within_budget(seasonal_raw: pd.DataFrame) -> None:
    report = run_drift_check(
        raw=seasonal_raw,
        profile=None,
        bundle=_bundle(),
        committed_metrics={"model": {"test": {"mae": 100.0}}},
    )
    assert report["error_drift"] is not None
    assert not report["error_drift"]["alert"]
    assert not report["drift_detected"]


def test_full_check_and_report_files(tmp_path: Path, seasonal_raw: pd.DataFrame) -> None:
    X = _features(seasonal_raw)
    report = run_drift_check(
        raw=seasonal_raw,
        profile=build_reference_profile(X),
        bundle=_bundle(),
        committed_metrics={"model": {"test": {"mae": 100.0}}},
    )
    json_path, md_path = write_report(report, tmp_path / "drift_report")

    assert json.loads(json_path.read_text(encoding="utf-8"))["feature_drift"] is not None
    md = md_path.read_text(encoding="utf-8")
    assert "Feature drift" in md and "Prediction-error drift" in md
