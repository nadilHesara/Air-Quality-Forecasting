"""
Phase 6.1 — Error analysis for the next-day PM2.5 model.

Reproduces the *exact* chronological train/test split and LightGBM model from
``src/train.py`` (importing its helpers so there is no second, drifting copy),
then dissects **where** the model wins and loses against the persistence
baseline.  The motivating question: the model only beats persistence by ~4.5%
MAE — is that gap improvable, or is it intrinsic to a strongly auto-correlated
series?

Outputs (all under ``reports/``):

* ``error_residuals_over_time.png`` — model vs persistence absolute error per
  test day, plus the signed residual, to see *when* the model loses.
* ``error_by_month.png`` — MAE by calendar month (model vs persistence).
* ``error_vs_level.png`` — signed residual and |error| vs the actual PM2.5
  level, to test the "under-predicts sharp spikes" hypothesis.
* ``error_analysis.md`` — a written, honest summary of the findings.

Run::

    python -m experiments.error_analysis            # fetch-free, uses cached parquet
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

import config  # noqa: E402
from src.features import RAW_PM_COLUMN, build_supervised  # noqa: E402
from src.train import (  # noqa: E402
    LGBM_PARAMS,
    TEST_HORIZON_DAYS,
    persistence_prediction,
)


def _load_raw() -> pd.DataFrame:
    parquet = config.DATA_DIR / "training_data.parquet"
    if not parquet.exists():
        raise SystemExit(f"Missing {parquet}; run `python -m src.train` first.")
    return pd.read_parquet(parquet).sort_index()


def _split(X: pd.DataFrame, y: pd.Series):
    """Same chronological split train.py uses."""
    cutoff = X.index.max() - timedelta(days=TEST_HORIZON_DAYS)
    train = X.index <= cutoff
    test = X.index > cutoff
    return X[train], y[train], X[test], y[test]


def main() -> None:
    raw = _load_raw()
    X, y = build_supervised(raw)
    X_train, y_train, X_test, y_test = _split(X, y)

    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    model_pred = pd.Series(model.predict(X_test), index=X_test.index, name="model")
    persist_pred = persistence_prediction(raw, X_test.index)
    persist_pred.name = "persistence"

    actual = y_test.rename("actual")
    today_pm = raw.loc[X_test.index, RAW_PM_COLUMN].rename("today_pm")

    df = pd.concat([actual, model_pred, persist_pred, today_pm], axis=1)
    df["model_err"] = df["model"] - df["actual"]           # signed
    df["persist_err"] = df["persistence"] - df["actual"]
    df["model_abserr"] = df["model_err"].abs()
    df["persist_abserr"] = df["persist_err"].abs()
    df["model_wins"] = df["model_abserr"] < df["persist_abserr"]
    df["day_change"] = df["actual"] - df["today_pm"]        # true next-day move
    df["month"] = df.index.month

    reports = config.REPORTS_DIR

    # ── Headline numbers ─────────────────────────────────────────────────────
    model_mae = df["model_abserr"].mean()
    persist_mae = df["persist_abserr"].mean()
    win_rate = df["model_wins"].mean()
    # On "big move" days (|true change| in top quartile), who wins?
    big_move = df["day_change"].abs() >= df["day_change"].abs().quantile(0.75)
    calm = ~big_move

    # ── Plot 1: errors over time ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax = axes[0]
    ax.plot(df.index, df["actual"], color="#444", lw=1.2, label="actual next-day PM2.5")
    ax.plot(df.index, df["model"], color="#1f77b4", lw=1.0, alpha=0.9, label="model")
    ax.plot(df.index, df["persistence"], color="#ff7f0e", lw=1.0, alpha=0.7, ls="--", label="persistence")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Held-out test: predictions vs actual")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    ax.plot(df.index, df["model_abserr"], color="#1f77b4", lw=1.0, label="|model error|")
    ax.plot(df.index, df["persist_abserr"], color="#ff7f0e", lw=1.0, alpha=0.7, ls="--", label="|persistence error|")
    ax.fill_between(
        df.index, 0, df["model_abserr"], where=df["model_wins"],
        color="#2ca02c", alpha=0.15, label="model wins",
    )
    ax.set_ylabel("absolute error")
    ax.set_title(f"Absolute error over time (model wins {win_rate:.0%} of days)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(reports / "error_residuals_over_time.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: MAE by month ─────────────────────────────────────────────────
    by_month = df.groupby("month")[["model_abserr", "persist_abserr"]].mean()
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.arange(len(by_month))
    w = 0.4
    ax.bar(idx - w / 2, by_month["model_abserr"], w, label="model", color="#1f77b4")
    ax.bar(idx + w / 2, by_month["persist_abserr"], w, label="persistence", color="#ff7f0e")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(m) for m in by_month.index])
    ax.set_xlabel("calendar month (test window)")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("MAE by month — model vs persistence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "error_by_month.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: error vs actual level & vs true day-over-day move ────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.axhline(0, color="#999", lw=0.8)
    ax.scatter(df["actual"], df["model_err"], s=18, alpha=0.6, color="#1f77b4", label="model residual")
    # trend line
    z = np.polyfit(df["actual"], df["model_err"], 1)
    xs = np.linspace(df["actual"].min(), df["actual"].max(), 50)
    ax.plot(xs, np.polyval(z, xs), color="#d62728", lw=1.5, label=f"trend (slope={z[0]:+.2f})")
    ax.set_xlabel("actual next-day PM2.5")
    ax.set_ylabel("model residual (pred − actual)")
    ax.set_title("Residual vs actual level\n(negative at high levels ⇒ under-predicts spikes)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.axhline(0, color="#999", lw=0.8)
    ax.scatter(df["day_change"], df["model_err"], s=18, alpha=0.6, color="#1f77b4", label="model")
    ax.scatter(df["day_change"], df["persist_err"], s=18, alpha=0.4, color="#ff7f0e", marker="x", label="persistence")
    ax.set_xlabel("true next-day move (actual − today)")
    ax.set_ylabel("residual (pred − actual)")
    ax.set_title("Residual vs true day-over-day move\n(persistence residual = −move by construction)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports / "error_vs_level.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ── Written summary ──────────────────────────────────────────────────────
    mae_big_model = df.loc[big_move, "model_abserr"].mean()
    mae_big_persist = df.loc[big_move, "persist_abserr"].mean()
    mae_calm_model = df.loc[calm, "model_abserr"].mean()
    mae_calm_persist = df.loc[calm, "persist_abserr"].mean()
    high_level = df["actual"] >= df["actual"].quantile(0.75)
    bias_high = df.loc[high_level, "model_err"].mean()

    lines = [
        "# Phase 6.1 — Error Analysis: where the model loses to persistence",
        "",
        f"**Test window:** {df.index.min().date()} → {df.index.max().date()} "
        f"({len(df)} days)  ",
        f"**Model MAE:** {model_mae:.3f}  ·  **Persistence MAE:** {persist_mae:.3f}  "
        f"·  **Model wins on {win_rate:.0%} of days**",
        "",
        "## Why persistence is so hard to beat",
        "",
        f"The daily PM2.5 series has lag-1 autocorrelation ≈ "
        f"{raw[RAW_PM_COLUMN].autocorr(1):.2f}: today's value is by far the best "
        "single predictor of tomorrow's. Persistence encodes exactly that, so the "
        "*only* headroom above persistence is in correctly anticipating the "
        "**day-over-day move** — which is close to noise for a daily-mean series.",
        "",
        "## Split by regime",
        "",
        "| Regime | Model MAE | Persistence MAE |",
        "|---|---:|---:|",
        f"| Calm days (small true move, bottom 75%) | {mae_calm_model:.3f} | {mae_calm_persist:.3f} |",
        f"| Big-move days (top 25% of \\|true move\\|) | {mae_big_model:.3f} | {mae_big_persist:.3f} |",
        "",
        "On calm days the two are near-identical (both essentially copy today). "
        "The model's edge, such as it is, comes on big-move days — but that is "
        "also where absolute errors are largest and hardest to nail.",
        "",
        "## Bias at high pollution levels",
        "",
        f"Mean model residual on high-PM2.5 days (top quartile of actual): "
        f"**{bias_high:+.2f} µg/m³**. A negative value confirms the model "
        "**under-predicts spikes** — it regresses sharp peaks toward the recent "
        "mean, the classic smoothing behaviour of a tree ensemble trained on MAE.",
        "",
        "## Takeaways for 6.2–6.5",
        "",
        "1. Headroom is concentrated in *big-move* days; features that hint at an "
        "imminent change (wind shift, pressure change, precipitation) are the "
        "only plausible lever. Calendar/rolling features mostly help persistence-"
        "like calm days where there's little to gain.",
        "2. The under-prediction of spikes argues for **quantile / interval** "
        "outputs (6.5): even if the point forecast can't nail a spike, an upper "
        "band can flag the risk.",
        "3. A large point-MAE win over persistence is unlikely to be real; the "
        "honest bar is a *small but consistent* improvement plus useful "
        "uncertainty — which is what the rest of Phase 6 targets.",
        "",
        "![errors over time](error_residuals_over_time.png)",
        "",
        "![MAE by month](error_by_month.png)",
        "",
        "![error vs level](error_vs_level.png)",
        "",
    ]
    out = reports / "error_analysis.md"
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Model MAE={model_mae:.3f}  Persistence MAE={persist_mae:.3f}  win_rate={win_rate:.1%}")
    print(f"Calm: model {mae_calm_model:.3f} vs persist {mae_calm_persist:.3f}")
    print(f"Big-move: model {mae_big_model:.3f} vs persist {mae_big_persist:.3f}")
    print(f"High-level bias: {bias_high:+.2f}")
    print(f"Wrote {out} and 3 PNGs to {reports}")


if __name__ == "__main__":
    main()
