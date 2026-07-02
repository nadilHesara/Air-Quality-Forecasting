"""
Phase 6.6 (stretch) — Multi-horizon forecasts: +1 / +2 / +3 days.

Uses the **direct** multi-horizon strategy: one LightGBM per horizon h, each
predicting PM2.5 h days ahead from the *same* leakage-safe features available on
day t (``src.features.make_features``). Direct (vs recursive) avoids compounding
one-step errors and needs no change to the feature builder — only the target
shifts further into the future (``pm2_5_mean.shift(-h)``).

For each horizon we compare against the natural persistence baseline for that
horizon (tomorrow-and-beyond = today's value) on the same held-out test window,
and report the model's MAE, persistence MAE, and % improvement.

This is intentionally an *experiment*, not wired into serving: the API/dashboard
ship the calibrated next-day forecast (+ interval). It documents that the
approach extends cleanly, and shows how skill decays with horizon.

Run::

    python -m experiments.multi_horizon
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error  # noqa: E402

import config  # noqa: E402
from experiments._harness import load_raw  # noqa: E402
from src.features import RAW_PM_COLUMN, make_features  # noqa: E402
from src.train import LGBM_PARAMS, TEST_HORIZON_DAYS  # noqa: E402

HORIZONS = (1, 2, 3)


def _supervised_h(raw: pd.DataFrame, h: int):
    """Features (day t) + target = PM2.5 on day t+h, NaN rows dropped."""
    X = make_features(raw)
    y = raw[RAW_PM_COLUMN].shift(-h).rename(f"pm25_h{h}")
    combined = X.copy()
    combined[y.name] = y
    combined = combined.dropna()
    return combined.drop(columns=[y.name]), combined[y.name]


def _persistence_h(raw: pd.DataFrame, index: pd.DatetimeIndex) -> np.ndarray:
    """Persistence for any horizon: predict day t+h = today's (day t) PM2.5."""
    return raw.loc[index, RAW_PM_COLUMN].to_numpy()


def main() -> None:
    raw = load_raw()
    rows = []
    for h in HORIZONS:
        X, y = _supervised_h(raw, h)
        cutoff = X.index.max() - timedelta(days=TEST_HORIZON_DAYS)
        tr, te = X.index <= cutoff, X.index > cutoff
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        model_mae = mean_absolute_error(y[te], pred)
        persist_mae = mean_absolute_error(y[te], _persistence_h(raw, X[te].index))
        imp = (persist_mae - model_mae) / persist_mae * 100.0
        rows.append((h, model_mae, persist_mae, imp))
        print(f"h=+{h}d  model MAE={model_mae:.3f}  persistence MAE={persist_mae:.3f}  "
              f"improvement={imp:+.1f}%")

    # Plot skill decay.
    hs = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(hs, [r[1] for r in rows], "o-", color="#1f77b4", label="model")
    ax.plot(hs, [r[2] for r in rows], "s--", color="#ff7f0e", label="persistence")
    ax.set_xticks(hs)
    ax.set_xlabel("forecast horizon (days ahead)")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("Multi-horizon skill: MAE by horizon (held-out test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "multi_horizon.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Phase 6.6 — Multi-horizon forecasts (+1 / +2 / +3 days)",
        "",
        "Direct strategy: one LightGBM per horizon, each predicting PM2.5 *h* "
        "days ahead from the day-*t* features (`make_features`, unchanged). Same "
        "held-out test window; persistence baseline = today's value carried *h* "
        "days forward.",
        "",
        "| Horizon | Model MAE | Persistence MAE | Improvement |",
        "|---|---:|---:|---:|",
    ]
    for h, mm, pm, imp in rows:
        lines.append(f"| +{h} day | {mm:.3f} | {pm:.3f} | {imp:+.1f}% |")
    lines += [
        "",
        "Both the model and persistence lose accuracy as the horizon grows "
        "(today's value ages), and the model's edge over persistence is small and "
        "**noisy** across horizons rather than a clean trend — consistent with "
        "the 6.4 finding that the predictable signal beyond persistence is thin. "
        "The direct strategy extends cleanly with no feature-builder changes, so "
        "this is a ready foundation if longer-horizon serving is ever needed; for "
        "now the API/dashboard ship the calibrated next-day forecast + interval.",
        "",
        "![multi-horizon MAE](multi_horizon.png)",
        "",
    ]
    out = config.REPORTS_DIR / "multi_horizon.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
