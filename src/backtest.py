"""
Walk-forward backtesting harness (Phase 7.6).

The headline metrics in ``reports/metrics.json`` come from a *single* 90-day
held-out window — one draw from a noisy distribution.  This harness answers
the robustness question properly: it walks forward through the whole history,
**retraining at each step** exactly as production would have,

    fold 1: train [start … T₀]              → evaluate (T₀ … T₀+step]
    fold 2: train [start … T₀+step]         → evaluate (T₀+step … T₀+2·step]
    …

with ``T₀ = start + BACKTEST_INITIAL_TRAIN_DAYS`` and
``step = BACKTEST_STEP_DAYS`` (both in ``config.py``).  Every prediction is
made by a model that never saw its evaluation window — the same expanding
retrain-then-forecast loop the weekly CI job lives, replayed over history.

The persistence and seasonal-naive baselines are evaluated on the identical
windows, so the "does the model beat persistence?" question gets an answer
backed by every era of the data, not one quarter.

Outputs: ``reports/backtest.json``, ``reports/backtest.md``, and
``reports/backtest_mae.png`` (per-fold MAE, model vs persistence).

Run (offline, uses the cached parquet)::

    python -m src.backtest
    python -m src.backtest --initial-days 365 --step-days 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.features import build_supervised
from src.train import (
    LGBM_PARAMS,
    _metrics,
    _pct_improvement,
    persistence_prediction,
    seasonal_naive_prediction,
)

logger = logging.getLogger(__name__)


def walk_forward(
    raw: pd.DataFrame,
    *,
    initial_train_days: int = config.BACKTEST_INITIAL_TRAIN_DAYS,
    step_days: int = config.BACKTEST_STEP_DAYS,
    lgbm_params: dict | None = None,
) -> dict[str, Any]:
    """Replay history: retrain at each step, forecast the next window.

    Returns a JSON-serializable dict with per-fold and pooled metrics for the
    model and both baselines.
    """
    params = lgbm_params or LGBM_PARAMS
    X, y = build_supervised(raw)
    if X.empty:
        raise ValueError("No supervised rows — not enough history to backtest.")

    start = X.index.min()
    end = X.index.max()
    cutoff = start + pd.Timedelta(days=initial_train_days)
    if cutoff >= end:
        raise ValueError(
            f"initial_train_days={initial_train_days} leaves no evaluation data "
            f"(history spans {start.date()} → {end.date()})."
        )

    folds: list[dict[str, Any]] = []
    pooled: dict[str, list] = {"y": [], "model": [], "persistence": [], "seasonal": []}

    fold_no = 0
    while cutoff < end:
        window_end = cutoff + pd.Timedelta(days=step_days)
        train_mask = X.index <= cutoff
        test_mask = (X.index > cutoff) & (X.index <= window_end)
        cutoff = window_end
        if not test_mask.any():
            continue
        fold_no += 1

        X_tr, y_tr = X.loc[train_mask], y.loc[train_mask]
        X_te, y_te = X.loc[test_mask], y.loc[test_mask]

        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr)
        model_pred = model.predict(X_te)

        persist_pred = persistence_prediction(raw, X_te.index)
        seasonal_pred = seasonal_naive_prediction(raw, X_te.index)
        seasonal_valid = seasonal_pred.notna()

        fold = {
            "fold": fold_no,
            "train_rows": int(len(X_tr)),
            "test_window": [str(X_te.index.min().date()), str(X_te.index.max().date())],
            "test_rows": int(len(X_te)),
            "model": _metrics(y_te, model_pred),
            "persistence": _metrics(y_te, persist_pred.values),
            "seasonal_naive": _metrics(y_te[seasonal_valid], seasonal_pred[seasonal_valid].values)
            if seasonal_valid.any() else None,
        }
        folds.append(fold)
        logger.info(
            "Fold %2d  train=%4d  test %s → %s  model MAE=%.3f  persistence MAE=%.3f",
            fold_no, len(X_tr), *fold["test_window"],
            fold["model"]["mae"], fold["persistence"]["mae"],
        )

        pooled["y"].extend(np.asarray(y_te).tolist())
        pooled["model"].extend(np.asarray(model_pred).tolist())
        pooled["persistence"].extend(np.asarray(persist_pred).tolist())
        pooled["seasonal"].extend(np.asarray(seasonal_pred).tolist())

    if not folds:
        raise ValueError("Backtest produced no folds — check initial_train_days/step_days.")

    # Pooled (all out-of-sample predictions concatenated) — the headline.
    y_all = pd.Series(pooled["y"])
    seasonal_all = pd.Series(pooled["seasonal"])
    seasonal_ok = seasonal_all.notna()
    pooled_metrics = {
        "model": _metrics(y_all, np.asarray(pooled["model"])),
        "persistence": _metrics(y_all, np.asarray(pooled["persistence"])),
        "seasonal_naive": _metrics(y_all[seasonal_ok], seasonal_all[seasonal_ok].values),
    }
    improvement = {
        "persistence_mae_pct": _pct_improvement(
            pooled_metrics["persistence"]["mae"], pooled_metrics["model"]["mae"]
        ),
        "seasonal_naive_mae_pct": _pct_improvement(
            pooled_metrics["seasonal_naive"]["mae"], pooled_metrics["model"]["mae"]
        ),
    }
    folds_beating_persistence = sum(
        1 for f in folds if f["model"]["mae"] < f["persistence"]["mae"]
    )

    return {
        "scheme": {
            "initial_train_days": initial_train_days,
            "step_days": step_days,
            "history": [str(X.index.min().date()), str(X.index.max().date())],
            "lgbm_params": params,
        },
        "n_folds": len(folds),
        "n_predictions": int(len(y_all)),
        "folds_beating_persistence": folds_beating_persistence,
        "pooled": pooled_metrics,
        "improvement_vs_baseline": improvement,
        "folds": folds,
    }


def _plot_fold_mae(result: dict[str, Any], out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = result["folds"]
    labels = [f["test_window"][0] for f in folds]
    model_mae = [f["model"]["mae"] for f in folds]
    persist_mae = [f["persistence"]["mae"] for f in folds]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(folds))
    ax.plot(x, model_mae, marker="o", label="LightGBM (walk-forward retrain)")
    ax.plot(x, persist_mae, marker="s", linestyle="--", label="Persistence")
    ax.set_xticks(x[:: max(1, len(x) // 12)])
    ax.set_xticklabels(labels[:: max(1, len(x) // 12)], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Fold start date")
    ax.set_ylabel("MAE (µg/m³)")
    ax.set_title("Walk-forward backtest — per-fold MAE")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved backtest plot → %s", out)
    return out


def _write_markdown(result: dict[str, Any], out: Path) -> Path:
    p = result["pooled"]
    imp = result["improvement_vs_baseline"]
    s = result["scheme"]
    lines = [
        "# Walk-forward backtest",
        "",
        f"Retrain-and-forecast replay over {s['history'][0]} → {s['history'][1]}: "
        f"first train on {s['initial_train_days']} days, then repeatedly forecast the "
        f"next {s['step_days']} days and roll forward — **{result['n_folds']} refits**, "
        f"{result['n_predictions']} out-of-sample predictions in total.",
        "",
        "## Pooled out-of-sample metrics",
        "",
        "| Model | MAE | RMSE |",
        "|---|---:|---:|",
        f"| Persistence | {p['persistence']['mae']:.3f} | {p['persistence']['rmse']:.3f} |",
        f"| Seasonal-naive | {p['seasonal_naive']['mae']:.3f} | {p['seasonal_naive']['rmse']:.3f} |",
        f"| **LightGBM** | **{p['model']['mae']:.3f}** | **{p['model']['rmse']:.3f}** |",
        "",
        f"- Improvement vs persistence: **{imp['persistence_mae_pct']:+.1f}% MAE**",
        f"- Improvement vs seasonal-naive: **{imp['seasonal_naive_mae_pct']:+.1f}% MAE**",
        f"- Folds where the model beat persistence: "
        f"**{result['folds_beating_persistence']} / {result['n_folds']}**",
        "",
        "![Per-fold MAE](backtest_mae.png)",
        "",
        "## Per-fold detail",
        "",
        "| Fold | Test window | Train rows | Model MAE | Persistence MAE | Seasonal MAE |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for f in result["folds"]:
        seasonal = f"{f['seasonal_naive']['mae']:.3f}" if f["seasonal_naive"] else "—"
        lines.append(
            f"| {f['fold']} | {f['test_window'][0]} → {f['test_window'][1]} "
            f"| {f['train_rows']} | {f['model']['mae']:.3f} "
            f"| {f['persistence']['mae']:.3f} | {seasonal} |"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote backtest report → %s", out)
    return out


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_DIR / "training_data.parquet")
    parser.add_argument("--initial-days", type=int, default=config.BACKTEST_INITIAL_TRAIN_DAYS)
    parser.add_argument("--step-days", type=int, default=config.BACKTEST_STEP_DAYS)
    args = parser.parse_args(argv)

    if not args.data.exists():
        logger.error("No data at %s — run `python -m src.build_dataset` first.", args.data)
        return 2
    raw = pd.read_parquet(args.data)

    result = walk_forward(
        raw, initial_train_days=args.initial_days, step_days=args.step_days
    )

    (config.REPORTS_DIR / "backtest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    _plot_fold_mae(result, config.REPORTS_DIR / "backtest_mae.png")
    _write_markdown(result, config.REPORTS_DIR / "backtest.md")

    p = result["pooled"]
    logger.info(
        "Backtest done: %d folds, %d predictions — model MAE=%.3f vs persistence %.3f (%+.1f%%).",
        result["n_folds"], result["n_predictions"],
        p["model"]["mae"], p["persistence"]["mae"],
        result["improvement_vs_baseline"]["persistence_mae_pct"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
