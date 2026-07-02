"""
Phase 6.3 — Hyper-parameter tuning with Optuna, logged to MLflow.

Runs a Tree-structured Parzen (TPE) search over LightGBM hyper-parameters,
scoring each trial with the **same** expanding-window ``TimeSeriesSplit`` CV on
the **train pool only** that ``src/train.py`` uses. The held-out test set is
never touched here — it is scored once, later, by ``src/train.py`` after the
winning params are wired into ``LGBM_PARAMS``.

Each trial and the best result are logged to a dedicated MLflow experiment
(``pm25-hpo``) so the study is reproducible and browsable in the MLflow UI.

Run::

    python -m experiments.tune_lgbm                 # default 60 trials
    python -m experiments.tune_lgbm --n-trials 150
    python -m experiments.tune_lgbm --with-wind     # tune on baseline + wind sin/cos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lightgbm import LGBMRegressor  # noqa: E402

import config  # noqa: E402
from experiments._harness import chrono_split, cv_mae, load_raw  # noqa: E402
from experiments.feature_experiments import _supervised, add_wind_vector  # noqa: E402
from src.features import build_supervised  # noqa: E402

HPO_EXPERIMENT = "pm25-hpo"
RANDOM_STATE = config.RANDOM_STATE


def _search_space(trial: optuna.Trial) -> dict:
    """LightGBM search space, deliberately biased toward regularization —
    the 6.1 analysis showed the risk here is over-fitting the noise in the
    day-over-day move, not under-fitting."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1200, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=60)
    parser.add_argument(
        "--with-wind",
        action="store_true",
        help="Tune on baseline features + wind sin/cos (the 6.2 keeper).",
    )
    args = parser.parse_args(argv)

    raw = load_raw()
    if args.with_wind:
        X, y = _supervised(add_wind_vector, raw)
        feature_tag = "baseline+wind_vector"
    else:
        X, y = build_supervised(raw)
        feature_tag = "baseline"
    X_train, y_train, _, _ = chrono_split(X, y)
    print(f"Tuning on {feature_tag}: {X_train.shape[0]} train rows × {X_train.shape[1]} features")

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_experiment(HPO_EXPERIMENT)

    def objective(trial: optuna.Trial) -> float:
        params = _search_space(trial)
        res = cv_mae(lambda: LGBMRegressor(**params), X_train, y_train)
        # Log each trial as a nested MLflow run.
        with mlflow.start_run(nested=True):
            mlflow.log_params({f"lgbm_{k}": v for k, v in params.items()})
            mlflow.set_tag("features", feature_tag)
            mlflow.log_metrics(
                {"cv_mae_mean": res["mae_mean"], "cv_mae_std": res["mae_std"],
                 "cv_rmse_mean": res["rmse_mean"]}
            )
        trial.set_user_attr("cv_mae_std", res["mae_std"])
        return res["mae_mean"]

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    with mlflow.start_run(run_name=f"hpo-{feature_tag}") as parent:
        mlflow.set_tag("features", feature_tag)
        mlflow.log_param("n_trials", args.n_trials)
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

        best = study.best_trial
        mlflow.log_metric("best_cv_mae", best.value)
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})
        print(f"\nMLflow parent run: {parent.info.run_id}")

    # Baseline (untuned) CV for context.
    from src.train import LGBM_PARAMS

    base = cv_mae(lambda: LGBMRegressor(**LGBM_PARAMS), X_train, y_train)

    print("\n" + "=" * 60)
    print(f"Baseline (current LGBM_PARAMS) CV MAE: {base['mae_mean']:.4f} ± {base['mae_std']:.3f}")
    print(f"Best tuned CV MAE:                     {best.value:.4f} "
          f"(±{best.user_attrs.get('cv_mae_std', float('nan')):.3f})")
    print(f"Δ CV MAE: {best.value - base['mae_mean']:+.4f}")
    print("=" * 60)
    print("Best params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")

    # Write a copy-paste-ready params block + record.
    out = config.REPORTS_DIR / "hpo_best_params.md"
    lines = [
        "# Phase 6.3 — Optuna hyper-parameter search",
        "",
        f"- Features: `{feature_tag}`",
        f"- Trials: {args.n_trials} (TPE, seed {RANDOM_STATE})",
        "- Scoring: expanding-window TimeSeriesSplit CV, train pool only",
        f"- MLflow experiment: `{HPO_EXPERIMENT}`",
        "",
        f"**Baseline CV MAE:** {base['mae_mean']:.4f} ± {base['mae_std']:.3f}  ",
        f"**Best tuned CV MAE:** {best.value:.4f}  ",
        f"**Δ:** {best.value - base['mae_mean']:+.4f}",
        "",
        "## Best params",
        "",
        "```python",
        "LGBM_PARAMS = {",
    ]
    full = {**best.params, "subsample_freq": 1, "random_state": RANDOM_STATE,
            "n_jobs": -1, "verbose": -1}
    for k, v in full.items():
        lines.append(f'    "{k}": {v!r},')
    lines += ["}", "```", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
