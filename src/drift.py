"""
Input-feature and prediction-error drift monitoring (Phase 7.4).

Retraining on a schedule is blind: if the upstream data source degrades or the
world changes (new emission sources, sensor swap, El Niño year), the pipeline
would happily keep training and serving without anyone noticing.  This module
makes that visible:

1. **Feature drift** — at training time, ``src/train.py`` persists a
   *reference profile* of every monitored feature's distribution over the
   train pool to ``reports/feature_reference.json``.  A drift check then bins
   the most recent :data:`config.DRIFT_WINDOW_DAYS` of feature values with
   the *same* edges and computes the **Population Stability Index** (PSI)
   per feature:

   - PSI < 0.10          → stable
   - 0.10 ≤ PSI < 0.25   → moderate shift (warn)
   - PSI ≥ 0.25          → significant shift (alert)

   Two deliberate choices keep this from crying wolf on a *seasonal* series:

   - The reference is **month-conditional**: each feature is profiled per
     calendar month (pooled across all training years), and recent values
     are scored against their own months' profiles.  A monsoon June is
     compared with previous Junes, not with the all-year distribution — a
     naive pooled reference flags *every* June as "drifted".
   - **Quintile** (5) bins and a ~60-day window keep PSI sampling noise
     (≈ (bins−1)·(1/n_recent + 1/n_reference)/2) well under the 0.10 warn
     threshold.

   Calendar features (``month``, ``day_of_year``, …) are excluded — a short
   window always "drifts" against a multi-year calendar distribution.

2. **Prediction-error drift** — the committed model's recent error is
   estimated by predicting each of the last N supervised rows and comparing
   with the actual next-day PM2.5.  If that recent MAE exceeds
   :data:`config.DRIFT_ERROR_RATIO_ALERT` × the committed held-out test MAE,
   the model is degrading in the wild → alert.

The CI retrain workflow runs this check on freshly fetched data *before*
training (new data vs the previous run's reference + model) and opens a
GitHub issue when drift is detected; drift alerts do **not** block the
retrain itself.

Exit codes: ``0`` = no drift, ``1`` = drift detected, ``2`` = cannot check
(missing inputs — treated as a skip, not a failure, by the workflow).

Run::

    python -m src.drift                      # use repo-default paths
    python -m src.drift --reference reports/feature_reference.json \
        --data data/training_data.parquet --model models/model.joblib \
        --metrics reports/metrics.json --output reports/drift_report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from src.features import build_supervised, feature_names, make_features

logger = logging.getLogger(__name__)

# Calendar features cycle by construction; a short recent window would always
# look "drifted" against the full-history distribution, so they're not scored.
CALENDAR_FEATURES: set[str] = {"day_of_week", "month", "day_of_year", "is_weekend"}

N_BINS: int = 5                  # quintile bins — robust for ~monthly samples
MIN_MONTH_REF_ROWS: int = 20     # fewer reference rows → fall back to global
MIN_SCORE_ROWS: int = 5          # fewer recent rows in a month → don't score it
_EPS: float = 1e-4               # avoids log(0)/division-by-zero in PSI


def monitored_features() -> list[str]:
    """The features scored for drift (all model features minus calendar)."""
    return [f for f in feature_names() if f not in CALENDAR_FEATURES]


# ── Reference profile (built at training time) ───────────────────────────────
def _distribution(values: np.ndarray) -> dict[str, Any] | None:
    """Quintile bin edges + fractions for one sample, or ``None`` if degenerate."""
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, N_BINS + 1)))
    if len(edges) < 3:  # (near-)constant sample — PSI is meaningless
        return None
    # Outer edges open-ended so future out-of-range values still bin.
    inner = edges[1:-1]
    counts, _ = np.histogram(values, bins=np.concatenate(([-np.inf], inner, [np.inf])))
    return {
        "bin_edges": inner.tolist(),
        "bin_fractions": (counts / counts.sum()).tolist(),
    }


def build_reference_profile(X: pd.DataFrame) -> dict[str, Any]:
    """Summarize each monitored feature's distribution, per calendar month.

    Called by ``src/train.py`` on the train-pool feature matrix.  Each feature
    gets a ``global`` distribution plus one per calendar month with at least
    :data:`MIN_MONTH_REF_ROWS` training rows — scoring then compares a recent
    June against previous Junes instead of against the whole year (PM2.5 is
    strongly seasonal; a pooled reference would alarm every monsoon).  The
    profile is JSON-serializable and intentionally small, so it can live in
    ``reports/`` next to ``metrics.json``.
    """
    months = X.index.month if isinstance(X.index, pd.DatetimeIndex) else None
    features: dict[str, Any] = {}
    for name in monitored_features():
        if name not in X.columns:
            continue
        series = X[name].dropna()
        values = series.to_numpy(dtype=float)
        if len(values) == 0:
            continue
        global_dist = _distribution(values)
        if global_dist is None:
            features[name] = {
                "constant": True,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
            continue
        by_month: dict[str, Any] = {}
        if months is not None:
            for month in range(1, 13):
                month_values = series[series.index.month == month].to_numpy(dtype=float)
                if len(month_values) < MIN_MONTH_REF_ROWS:
                    continue
                dist = _distribution(month_values)
                if dist is not None:
                    by_month[str(month)] = dist
        features[name] = {
            "constant": False,
            "global": global_dist,
            "by_month": by_month,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_rows": int(len(X)),
        "window": [str(X.index.min().date()), str(X.index.max().date())] if len(X) else [],
        "n_bins": N_BINS,
        "features": features,
    }


def save_reference_profile(X: pd.DataFrame, path: Path) -> Path:
    path.write_text(json.dumps(build_reference_profile(X), indent=2), encoding="utf-8")
    logger.info("Wrote feature reference profile → %s", path)
    return path


# ── PSI scoring ───────────────────────────────────────────────────────────────
def population_stability_index(expected: np.ndarray, actual: np.ndarray) -> float:
    """PSI between two aligned bin-fraction vectors (0 = identical)."""
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _psi_against(dist: dict[str, Any], values: np.ndarray) -> float:
    """PSI of ``values`` binned with a stored reference distribution."""
    inner = np.asarray(dist["bin_edges"], dtype=float)
    counts, _ = np.histogram(values, bins=np.concatenate(([-np.inf], inner, [np.inf])))
    return population_stability_index(np.asarray(dist["bin_fractions"]), counts / counts.sum())


def score_feature_drift(
    profile: dict[str, Any],
    X_recent: pd.DataFrame,
    *,
    warn_at: float = config.DRIFT_PSI_WARN,
    alert_at: float = config.DRIFT_PSI_ALERT,
) -> list[dict[str, Any]]:
    """Score every profiled feature's recent window against the reference.

    Seasonality-aware: recent values are grouped by calendar month and each
    group is scored against *that month's* reference distribution (falling
    back to the global one when the training pool had too few rows for the
    month); the per-month PSIs are then combined weighted by group size.
    Months with fewer than :data:`MIN_SCORE_ROWS` recent rows carry no
    distributional signal and are ignored.

    Returns one record per feature: ``{feature, psi, status}`` where status is
    ``stable`` / ``warn`` / ``alert`` (or ``skipped`` for constant features).
    """
    results: list[dict[str, Any]] = []
    for name, ref in profile.get("features", {}).items():
        if name not in X_recent.columns:
            results.append({"feature": name, "psi": None, "status": "missing"})
            continue
        series = X_recent[name].dropna()
        if ref.get("constant") or series.empty:
            results.append({"feature": name, "psi": None, "status": "skipped"})
            continue

        by_month = ref.get("by_month", {})
        weighted: list[tuple[int, float]] = []  # (n_rows, psi) per scored month
        for month, group in series.groupby(series.index.month):
            values = group.to_numpy(dtype=float)
            if len(values) < MIN_SCORE_ROWS:
                continue
            dist = by_month.get(str(month), ref["global"])
            weighted.append((len(values), _psi_against(dist, values)))

        if not weighted:
            results.append({"feature": name, "psi": None, "status": "skipped"})
            continue
        total = sum(n for n, _ in weighted)
        psi = sum(n * p for n, p in weighted) / total
        status = "alert" if psi >= alert_at else "warn" if psi >= warn_at else "stable"
        results.append({"feature": name, "psi": round(psi, 4), "status": status})
    return sorted(results, key=lambda r: -(r["psi"] or 0.0))


# ── Prediction-error drift ───────────────────────────────────────────────────
def recent_prediction_error(
    bundle: dict[str, Any],
    raw: pd.DataFrame,
    *,
    window_days: int = config.DRIFT_WINDOW_DAYS,
) -> dict[str, Any] | None:
    """MAE of the bundled model over the most recent supervised rows.

    Uses the same leakage-safe features/target as training, restricted to the
    last ``window_days`` rows with a known next-day actual.  Returns ``None``
    when there are no usable rows.
    """
    X, y = build_supervised(raw)
    if X.empty:
        return None
    X_recent = X.tail(window_days)[list(bundle["feature_names"])]
    y_recent = y.tail(window_days)
    preds = bundle["model"].predict(X_recent)
    mae = float(np.mean(np.abs(np.asarray(y_recent) - np.asarray(preds))))
    return {
        "window_days": int(len(X_recent)),
        "window": [str(X_recent.index.min().date()), str(X_recent.index.max().date())],
        "recent_mae": mae,
    }


# ── Orchestration + report ───────────────────────────────────────────────────
def run_drift_check(
    *,
    raw: pd.DataFrame,
    profile: dict[str, Any] | None,
    bundle: dict[str, Any] | None,
    committed_metrics: dict[str, Any] | None,
    window_days: int = config.DRIFT_WINDOW_DAYS,
    error_ratio_alert: float = config.DRIFT_ERROR_RATIO_ALERT,
) -> dict[str, Any]:
    """Run both drift checks and assemble a JSON-serializable report.

    Any missing input (no reference profile yet, no model, no metrics) skips
    that check rather than failing — the first ever retrain has nothing to
    compare against.
    """
    report: dict[str, Any] = {
        "checked_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "window_days": window_days,
        "feature_drift": None,
        "error_drift": None,
        "drift_detected": False,
        "alerts": [],
    }

    # Feature drift: recent window vs training-time reference.
    if profile is not None:
        feats = make_features(raw).dropna()
        recent = feats.tail(window_days)
        scores = score_feature_drift(profile, recent)
        n_alert = sum(1 for s in scores if s["status"] == "alert")
        n_warn = sum(1 for s in scores if s["status"] == "warn")
        report["feature_drift"] = {
            "reference_created": profile.get("created_utc"),
            "recent_window": [str(recent.index.min().date()), str(recent.index.max().date())]
            if len(recent) else [],
            "n_alert": n_alert,
            "n_warn": n_warn,
            "features": scores,
        }
        if n_alert:
            drifted = [s["feature"] for s in scores if s["status"] == "alert"]
            report["alerts"].append(
                f"{n_alert} feature(s) with PSI ≥ {config.DRIFT_PSI_ALERT}: {', '.join(drifted)}"
            )
    else:
        logger.warning("No reference profile — skipping feature-drift check (first run?).")

    # Error drift: committed model's recent live MAE vs its committed test MAE.
    if bundle is not None and committed_metrics is not None:
        err = recent_prediction_error(bundle, raw, window_days=window_days)
        if err is not None:
            test_mae = float(committed_metrics["model"]["test"]["mae"])
            ratio = err["recent_mae"] / test_mae if test_mae > 0 else float("inf")
            err.update(
                committed_test_mae=test_mae,
                ratio=round(ratio, 3),
                alert=bool(ratio > error_ratio_alert),
            )
            report["error_drift"] = err
            if err["alert"]:
                report["alerts"].append(
                    f"Recent MAE {err['recent_mae']:.3f} is {ratio:.2f}× the committed "
                    f"test MAE {test_mae:.3f} (alert threshold {error_ratio_alert}×)"
                )
    else:
        logger.warning("No committed model/metrics — skipping error-drift check.")

    report["drift_detected"] = bool(report["alerts"])
    return report


def write_report(report: dict[str, Any], out_stem: Path) -> tuple[Path, Path]:
    """Write ``<stem>.json`` and a human-readable ``<stem>.md``."""
    json_path = out_stem.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Drift report",
        "",
        f"Checked: {report['checked_utc']}  •  window: last {report['window_days']} days",
        "",
        f"**Drift detected: {'YES' if report['drift_detected'] else 'no'}**",
        "",
    ]
    for alert in report["alerts"]:
        lines.append(f"- ⚠️ {alert}")
    if report["alerts"]:
        lines.append("")

    fd = report.get("feature_drift")
    if fd:
        lines += [
            "## Feature drift (PSI vs training reference)",
            "",
            f"Reference built: {fd['reference_created']}  •  "
            f"alerts: {fd['n_alert']}  •  warnings: {fd['n_warn']}",
            "",
            "| Feature | PSI | Status |",
            "|---|---:|---|",
        ]
        for s in fd["features"]:
            psi = f"{s['psi']:.4f}" if s["psi"] is not None else "—"
            lines.append(f"| {s['feature']} | {psi} | {s['status']} |")
        lines.append("")
    ed = report.get("error_drift")
    if ed:
        lines += [
            "## Prediction-error drift",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Recent MAE ({ed['window_days']} days, {ed['window'][0]} → {ed['window'][1]}) "
            f"| {ed['recent_mae']:.3f} |",
            f"| Committed test MAE | {ed['committed_test_mae']:.3f} |",
            f"| Ratio | {ed['ratio']:.2f}× (alert > {config.DRIFT_ERROR_RATIO_ALERT}×) |",
            "",
        ]

    md_path = out_stem.with_suffix(".md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote drift report → %s / %s", json_path, md_path)
    return json_path, md_path


def _load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=config.DATA_DIR / "training_data.parquet",
                        help="Daily training parquet with the freshest data.")
    parser.add_argument("--reference", type=Path, default=config.REPORTS_DIR / "feature_reference.json",
                        help="Reference profile written by the previous training run.")
    parser.add_argument("--model", type=Path, default=config.MODELS_DIR / "model.joblib",
                        help="The currently committed (champion) model bundle.")
    parser.add_argument("--metrics", type=Path, default=config.REPORTS_DIR / "metrics.json",
                        help="The currently committed metrics.json (for the test MAE).")
    parser.add_argument("--output", type=Path, default=config.REPORTS_DIR / "drift_report",
                        help="Output stem; writes <stem>.json and <stem>.md.")
    args = parser.parse_args(argv)

    if not args.data.exists():
        logger.error("No data at %s — run `python -m src.build_dataset` first.", args.data)
        return 2
    raw = pd.read_parquet(args.data)

    profile = _load_json(args.reference)
    committed_metrics = _load_json(args.metrics)
    bundle = joblib.load(args.model) if args.model.exists() else None

    if profile is None and (bundle is None or committed_metrics is None):
        logger.warning("Nothing to compare against (no reference, no model+metrics) — skipping drift check.")
        return 2

    report = run_drift_check(
        raw=raw, profile=profile, bundle=bundle, committed_metrics=committed_metrics
    )
    write_report(report, args.output)

    if report["drift_detected"]:
        for alert in report["alerts"]:
            logger.error("DRIFT: %s", alert)
        return 1
    logger.info("No drift detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
