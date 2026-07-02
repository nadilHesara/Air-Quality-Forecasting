"""
Tests for the pre-training data validation gate (src/validate.py, Phase 7.1).

The contract under test: a clean frame passes silently; a broken frame raises
:class:`DataValidationError` naming EVERY violated check, so an automated
retrain fails loudly (and completely) instead of training on garbage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.validate import DataValidationError, validate_training_frame


def _validate(df: pd.DataFrame, **kwargs):
    """Run validation with test-friendly defaults (fixtures are ~90 rows)."""
    kwargs.setdefault("min_rows", 50)
    return validate_training_frame(df, **kwargs)


def _failed_names(df: pd.DataFrame, **kwargs) -> list[str]:
    report = _validate(df, raise_on_failure=False, **kwargs)
    return [name for name, _ in report.failures]


def test_clean_frame_passes(seasonal_raw: pd.DataFrame) -> None:
    report = _validate(seasonal_raw)
    assert report.ok
    assert report.failures == []


def test_too_few_rows_fails(seasonal_raw: pd.DataFrame) -> None:
    failed = _failed_names(seasonal_raw.head(10))
    assert any(name == "row_count" for name in failed)


def test_pm25_out_of_range_fails(seasonal_raw: pd.DataFrame) -> None:
    df = seasonal_raw.copy()
    df.iloc[5, df.columns.get_loc("pm2_5_mean")] = -3.0        # impossible
    df.iloc[7, df.columns.get_loc("pm2_5_mean")] = 9_000.0     # implausible
    failed = _failed_names(df)
    assert "range[pm2_5_mean]" in failed


def test_all_nan_column_fails(seasonal_raw: pd.DataFrame) -> None:
    df = seasonal_raw.copy()
    df["relative_humidity_2m_mean"] = np.nan
    failed = _failed_names(df)
    assert "not_all_nan[relative_humidity_2m_mean]" in failed


def test_excessive_nan_fraction_fails(seasonal_raw: pd.DataFrame) -> None:
    df = seasonal_raw.copy()
    df.iloc[:20, df.columns.get_loc("pm10_mean")] = np.nan  # ~22% NaN
    failed = _failed_names(df)
    assert "nan_fraction[pm10_mean]" in failed


def test_unsorted_dates_fail(seasonal_raw: pd.DataFrame) -> None:
    shuffled = seasonal_raw.sample(frac=1.0, random_state=0)
    failed = _failed_names(shuffled)
    assert "dates_monotonic" in failed


def test_duplicate_dates_fail(seasonal_raw: pd.DataFrame) -> None:
    df = pd.concat([seasonal_raw, seasonal_raw.tail(3)]).sort_index()
    failed = _failed_names(df)
    assert "dates_unique" in failed


def test_large_date_gap_fails(seasonal_raw: pd.DataFrame) -> None:
    # Remove 20 consecutive days from the middle → a 21-day hole.
    df = pd.concat([seasonal_raw.iloc[:40], seasonal_raw.iloc[60:]])
    failed = _failed_names(df, min_rows=40)
    assert "date_gaps" in failed


def test_missing_required_column_fails(seasonal_raw: pd.DataFrame) -> None:
    failed = _failed_names(seasonal_raw.drop(columns=["wind_speed_10m_max"]))
    assert "required_columns" in failed


def test_constant_pm25_fails(seasonal_raw: pd.DataFrame) -> None:
    df = seasonal_raw.copy()
    df["pm2_5_mean"] = 17.0
    failed = _failed_names(df)
    assert "pm25_varies" in failed


def test_raises_listing_every_violation(seasonal_raw: pd.DataFrame) -> None:
    """One broken frame, several problems — all named in a single error."""
    df = seasonal_raw.copy()
    df["pm2_5_mean"] = -1.0                       # out of range AND constant
    df["surface_pressure_mean"] = np.nan          # all-NaN column
    with pytest.raises(DataValidationError) as exc_info:
        _validate(df)
    message = str(exc_info.value)
    assert "range[pm2_5_mean]" in message
    assert "not_all_nan[surface_pressure_mean]" in message
    assert "pm25_varies" in message
