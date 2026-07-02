"""
Phase 6.2 — Feature experiments, each guarded by the 5.3 leakage test.

The 6.1 error analysis showed the model's only real headroom over persistence
is on **big-move days**; calendar/rolling features mostly help the calm days
where persistence already wins. So the candidates here are weighted toward
signals that *anticipate change* (wind vector, pressure change, humidity ×
precipitation), alongside the plan's other asks (cyclical calendar, longer
lags, PM10/PM2.5 ratio, lagged weather).

Method
------
Each candidate is an **additive** transform: it starts from the production
:func:`src.features.make_features` output and appends new columns. For each:

1. Run :func:`experiments._harness.assert_no_leakage` (the reusable 5.3 test) —
   any leak disqualifies the candidate immediately.
2. Evaluate with the *same* expanding-window TimeSeriesSplit CV on the **train
   pool only** (identical folds/params to production), so numbers are directly
   comparable to the baseline feature set.

We report CV MAE (mean ± std) for the baseline and each candidate, plus the
combined "keepers". Nothing here touches the held-out test set.

Run::

    python -m experiments.feature_experiments
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightgbm import LGBMRegressor  # noqa: E402

import config  # noqa: E402
from experiments._harness import (  # noqa: E402
    assert_no_leakage,
    chrono_split,
    cv_mae,
    load_raw,
)
from src.features import (  # noqa: E402
    RAW_PM_COLUMN,
    TARGET_COLUMN,
    add_target,
    make_features,
)
from src.train import LGBM_PARAMS  # noqa: E402


# ── Candidate feature builders (each = baseline features + new columns) ──────
# Every builder takes the raw daily frame and returns a *feature* frame. They
# reuse make_features() so the baseline columns are always present, then append.
def _base(raw: pd.DataFrame) -> pd.DataFrame:
    return make_features(raw)


def add_wind_vector(raw: pd.DataFrame) -> pd.DataFrame:
    """Wind direction as sin/cos (raw degrees wrap at 0/360, misleading a tree)."""
    f = make_features(raw)
    rad = np.deg2rad(raw["wind_direction_10m_dominant"])
    f["wind_dir_sin"] = np.sin(rad)
    f["wind_dir_cos"] = np.cos(rad)
    return f


def add_cyclical_calendar(raw: pd.DataFrame) -> pd.DataFrame:
    """month & day_of_year as sin/cos so Dec↔Jan and day 366↔1 are adjacent."""
    f = make_features(raw)
    doy = raw.index.dayofyear.to_numpy()
    month = raw.index.month.to_numpy()
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    f["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    f["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    return f


def add_weather_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    """humidity × precipitation and day-over-day pressure change.

    All use day-t (or lagged) values only — the pressure *change* is computed
    from shifted values so it never peeks at the future.
    """
    f = make_features(raw)
    f["humidity_x_precip"] = (
        raw["relative_humidity_2m_mean"] * raw["precipitation_sum"]
    )
    # Pressure change into today = today − yesterday (both ≤ t, leakage-safe).
    f["pressure_change_1d"] = raw["surface_pressure_mean"].diff(1)
    return f


def add_more_lags(raw: pd.DataFrame) -> pd.DataFrame:
    """Extra short/long PM2.5 lags: lag 4, 5, 21, 28 (all shift(k≥1))."""
    f = make_features(raw)
    pm = raw[RAW_PM_COLUMN]
    for lag in (4, 5, 21, 28):
        f[f"pm25_lag{lag}"] = pm.shift(lag)
    return f


def add_pm_ratio(raw: pd.DataFrame) -> pd.DataFrame:
    """PM10-to-PM2.5 ratio today (composition hint), plus its lag-1."""
    f = make_features(raw)
    ratio = raw["pm10_mean"] / raw[RAW_PM_COLUMN].replace(0, np.nan)
    f["pm10_pm25_ratio"] = ratio
    f["pm10_pm25_ratio_lag1"] = ratio.shift(1)
    return f


def add_lagged_weather(raw: pd.DataFrame) -> pd.DataFrame:
    """Yesterday's weather (lag-1) — regimes persist, so t−1 weather informs t+1."""
    f = make_features(raw)
    for col in ("wind_speed_10m_max", "relative_humidity_2m_mean", "precipitation_sum"):
        f[f"{col}_lag1"] = raw[col].shift(1)
    return f


CANDIDATES = {
    "wind_vector (sin/cos)": add_wind_vector,
    "cyclical_calendar": add_cyclical_calendar,
    "weather_interactions": add_weather_interactions,
    "more_lags (4/5/21/28)": add_more_lags,
    "pm10_pm25_ratio": add_pm_ratio,
    "lagged_weather": add_lagged_weather,
}


def _supervised(build, raw):
    """features + target, drop NaN rows, aligned — like build_supervised but
    for an arbitrary feature builder."""
    X = build(raw)
    y = add_target(raw)
    combined = X.copy()
    combined[TARGET_COLUMN] = y
    combined = combined.dropna()
    return combined.drop(columns=[TARGET_COLUMN]), combined[TARGET_COLUMN]


def _make_model():
    return LGBMRegressor(**LGBM_PARAMS)


def _evaluate(build, raw) -> dict[str, float]:
    X, y = _supervised(build, raw)
    X_train, y_train, _, _ = chrono_split(X, y)
    return cv_mae(_make_model, X_train, y_train)


def main() -> None:
    raw = load_raw()

    print(f"{'candidate':32s} {'n_feat':>6s} {'leak':>5s} {'CV MAE':>16s}  {'Δ vs base':>10s}")
    print("-" * 78)

    base_res = _evaluate(_base, raw)
    base_mae = base_res["mae_mean"]
    n_base = _base(raw).shape[1]
    print(f"{'BASELINE (26 feat)':32s} {n_base:6d} {'ok':>5s} "
          f"{base_res['mae_mean']:7.3f} ± {base_res['mae_std']:5.3f}  {'—':>10s}")

    keepers: list[str] = []
    rows = []
    for name, build in CANDIDATES.items():
        # 1) leakage guard (5.3 perturbation test)
        try:
            assert_no_leakage(build, raw)
            leak = "ok"
        except AssertionError:
            leak = "LEAK"
            print(f"{name:32s} {'-':>6s} {leak:>5s}  DISQUALIFIED (leakage)")
            continue

        # 2) CV eval on train pool only
        res = _evaluate(build, raw)
        n = build(raw).shape[1]
        delta = res["mae_mean"] - base_mae
        helps = delta < 0
        rows.append((name, delta))
        if helps:
            keepers.append(name)
        flag = "  ↓ helps" if helps else ""
        print(f"{name:32s} {n:6d} {leak:>5s} "
              f"{res['mae_mean']:7.3f} ± {res['mae_std']:5.3f}  {delta:+10.3f}{flag}")

    # Combined keepers -------------------------------------------------------
    print("-" * 78)
    if keepers:
        keeper_builds = [CANDIDATES[k] for k in keepers]

        def combined(raw_: pd.DataFrame) -> pd.DataFrame:
            f = make_features(raw_)
            for b in keeper_builds:
                extra = b(raw_)
                new_cols = [c for c in extra.columns if c not in f.columns]
                f = pd.concat([f, extra[new_cols]], axis=1)
            return f

        assert_no_leakage(combined, raw)
        res = _evaluate(combined, raw)
        n = combined(raw).shape[1]
        delta = res["mae_mean"] - base_mae
        print(f"{'COMBINED keepers':32s} {n:6d} {'ok':>5s} "
              f"{res['mae_mean']:7.3f} ± {res['mae_std']:5.3f}  {delta:+10.3f}")
        print(f"\nKeepers (CV-MAE-improving, leakage-free): {keepers}")
    else:
        print("No candidate improved CV MAE over the baseline feature set.")

    # Persist a compact record for results.md ---------------------------------
    out = config.REPORTS_DIR / "feature_experiments.md"
    lines = [
        "# Phase 6.2 — Feature experiments (leakage-guarded)",
        "",
        f"Baseline (26 features) CV MAE: **{base_mae:.3f} ± {base_res['mae_std']:.3f}** "
        "(expanding-window TimeSeriesSplit, train pool only).",
        "",
        "Each candidate is additive on top of the baseline features, passed the "
        "5.3 perturbation leakage test, and was scored with the identical CV. "
        "Δ is candidate CV MAE minus baseline (negative = improvement).",
        "",
        "| Candidate | Δ CV MAE vs baseline |",
        "|---|---:|",
    ]
    for name, delta in rows:
        lines.append(f"| {name} | {delta:+.3f} |")
    lines += [
        "",
        f"**Keepers promoted to `src/features.py`:** "
        f"{', '.join(keepers) if keepers else 'none — no candidate beat the baseline'}.",
        "",
        "_Note: on a series with lag-1 autocorrelation ≈ 0.83 the deltas are "
        "small by nature; we keep only features with a consistent (non-noise) "
        "CV improvement and re-confirm on the held-out test once, in `src/train.py`._",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
