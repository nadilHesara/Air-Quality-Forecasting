"""
Data validation for the training pipeline (Phase 7.1).

Retraining is automated, so nothing human looks at the fetched data before a
model is fit to it.  This module is the gate: :func:`validate_training_frame`
runs schema + range + sanity checks on the raw daily training table and raises
:class:`DataValidationError` listing **every** violation (not just the first),
so one failed CI run tells you the whole story.

The checks are deliberately hand-rolled (no pandera dependency) and cover:

- index integrity  — a sorted, unique ``DatetimeIndex`` with no gap larger
  than :data:`config.MAX_DATE_GAP_DAYS`;
- schema           — every raw column :mod:`src.features` needs is present;
- volume           — at least :data:`config.MIN_TRAINING_ROWS` rows (lags,
  CV folds, and the 90-day test split all need real history);
- completeness     — no required column is all-NaN or more than
  :data:`config.MAX_NAN_FRACTION` NaN;
- physical ranges  — PM2.5/PM10 within sane bounds, humidity in [0, 100],
  wind direction in [0, 360], precipitation non-negative;
- degeneracy       — PM2.5 actually varies (a constant feed means the
  upstream API broke, not that the air got perfectly stable).

Used by ``python -m src.train`` right after the data is loaded — a violation
fails the run (and the CI retrain job) loudly before any model is trained.

Run standalone against the cached parquet::

    python -m src.validate
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.features import POLLUTION_COLUMNS, RAW_PM_COLUMN, WEATHER_COLUMNS

logger = logging.getLogger(__name__)

# Every raw column the feature pipeline consumes (deduplicated, order kept).
REQUIRED_RAW_COLUMNS: list[str] = list(
    dict.fromkeys([RAW_PM_COLUMN, *POLLUTION_COLUMNS, *WEATHER_COLUMNS])
)

# Physical bounds per column: (min, max) inclusive; None = unbounded that side.
COLUMN_RANGES: dict[str, tuple[float | None, float | None]] = {
    "pm2_5_mean": config.PM25_VALID_RANGE,
    "pm10_mean": config.PM10_VALID_RANGE,
    "relative_humidity_2m_mean": (0.0, 100.0),
    "wind_direction_10m_dominant": (0.0, 360.0),
    "wind_speed_10m_max": (0.0, None),
    "precipitation_sum": (0.0, None),
    "surface_pressure_mean": (800.0, 1100.0),  # hPa, generous sea-level range
    "temperature_2m_mean": (-60.0, 60.0),      # °C, generous global range
}


class DataValidationError(ValueError):
    """Raised when the training frame fails one or more validation checks."""


@dataclass
class ValidationReport:
    """Outcome of a validation run: every check with its pass/fail detail."""

    n_rows: int
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def add(self, name: str, passed: bool, detail: str) -> None:
        self.checks.append((name, passed, detail))

    @property
    def failures(self) -> list[tuple[str, str]]:
        return [(name, detail) for name, ok, detail in self.checks if not ok]

    @property
    def ok(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        lines = [f"Data validation: {len(self.checks)} checks on {self.n_rows} rows"]
        for name, ok, detail in self.checks:
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        return "\n".join(lines)


def validate_training_frame(
    df: pd.DataFrame,
    *,
    min_rows: int = config.MIN_TRAINING_ROWS,
    max_nan_fraction: float = config.MAX_NAN_FRACTION,
    max_gap_days: int = config.MAX_DATE_GAP_DAYS,
    raise_on_failure: bool = True,
) -> ValidationReport:
    """Run every check against the raw daily training table.

    Parameters
    ----------
    df :
        The frame about to be fed to :func:`src.features.build_supervised`
        (raw weather + air-quality columns, ``DatetimeIndex``).  Extra columns
        (engineered features, target) are ignored.
    raise_on_failure :
        If ``True`` (the default), raise :class:`DataValidationError` listing
        all failed checks.  Set ``False`` to inspect the report instead.

    Returns
    -------
    ValidationReport
        Per-check outcomes; ``report.ok`` is the overall verdict.
    """
    report = ValidationReport(n_rows=len(df))

    # ── Index integrity ──────────────────────────────────────────────────
    is_dt = isinstance(df.index, pd.DatetimeIndex)
    report.add("index_is_datetime", is_dt, f"index type = {type(df.index).__name__}")

    if is_dt and len(df) > 1:
        report.add(
            "dates_monotonic",
            bool(df.index.is_monotonic_increasing),
            "index sorted ascending" if df.index.is_monotonic_increasing else "index is NOT sorted by date",
        )
        n_dupes = int(df.index.duplicated().sum())
        report.add("dates_unique", n_dupes == 0, f"{n_dupes} duplicated date(s)")

        gaps = df.index.sort_values().to_series().diff().dt.days.dropna()
        max_gap = int(gaps.max()) if len(gaps) else 0
        report.add(
            "date_gaps",
            max_gap <= max_gap_days,
            f"largest gap = {max_gap} day(s) (limit {max_gap_days})",
        )

    # ── Schema ───────────────────────────────────────────────────────────
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    report.add(
        "required_columns",
        not missing,
        "all present" if not missing else f"missing column(s): {missing}",
    )
    present = [c for c in REQUIRED_RAW_COLUMNS if c in df.columns]

    # ── Volume ───────────────────────────────────────────────────────────
    report.add("row_count", len(df) >= min_rows, f"{len(df)} rows (minimum {min_rows})")

    # ── Completeness ─────────────────────────────────────────────────────
    for col in present:
        frac = float(df[col].isna().mean())
        if frac == 1.0:
            report.add(f"not_all_nan[{col}]", False, "column is entirely NaN")
        else:
            report.add(
                f"nan_fraction[{col}]",
                frac <= max_nan_fraction,
                f"{frac:.1%} NaN (limit {max_nan_fraction:.0%})",
            )

    # ── Physical ranges ──────────────────────────────────────────────────
    for col, (lo, hi) in COLUMN_RANGES.items():
        if col not in df.columns:
            continue  # missing columns already reported by the schema check
        values = df[col].dropna()
        if values.empty:
            continue  # all-NaN already reported by the completeness check
        bad = pd.Series(False, index=values.index)
        if lo is not None:
            bad |= values < lo
        if hi is not None:
            bad |= values > hi
        n_bad = int(bad.sum())
        bounds = f"[{lo if lo is not None else '-inf'}, {hi if hi is not None else 'inf'}]"
        report.add(
            f"range[{col}]",
            n_bad == 0,
            f"all values within {bounds}"
            if n_bad == 0
            else f"{n_bad} value(s) outside {bounds} "
            f"(observed min={values.min():.2f}, max={values.max():.2f})",
        )

    # ── Degeneracy: a constant PM2.5 feed means the upstream source broke ─
    if RAW_PM_COLUMN in df.columns:
        pm = df[RAW_PM_COLUMN].dropna()
        varies = len(pm) > 1 and float(np.std(pm)) > 0.0
        report.add(
            "pm25_varies",
            varies,
            "PM2.5 shows variation" if varies else "PM2.5 is constant — upstream data source likely broken",
        )

    if report.ok:
        logger.info("Data validation passed (%d checks, %d rows).", len(report.checks), report.n_rows)
    else:
        logger.error("Data validation FAILED:\n%s", report.summary())
        if raise_on_failure:
            failed = "; ".join(f"{name}: {detail}" for name, detail in report.failures)
            raise DataValidationError(
                f"Training data failed {len(report.failures)} validation check(s) — {failed}"
            )
    return report


def main() -> int:
    """Validate the cached training parquet and print the full report."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parquet_path = config.DATA_DIR / "training_data.parquet"
    if not parquet_path.exists():
        logger.error("No cached data at %s — run `python -m src.build_dataset` first.", parquet_path)
        return 2
    df = pd.read_parquet(parquet_path)
    report = validate_training_frame(df, raise_on_failure=False)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
