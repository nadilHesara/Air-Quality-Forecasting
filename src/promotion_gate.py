"""
Champion/challenger promotion gate (Phase 7.2).

The weekly retrain must not blindly commit whatever model it produced — a bad
data week or a silent dependency change could make the new model *worse* than
the one already serving.  This gate compares the freshly trained model
(**challenger**, the new ``reports/metrics.json``) against the previously
committed one (**champion**, the ``metrics.json`` snapshotted before training)
and only lets the retrain workflow commit/promote when the challenger's
held-out test MAE is not worse than the champion's by more than
:data:`config.PROMOTION_MAX_MAE_REGRESSION_PCT` percent.

Why a tolerance instead of "must be strictly better": each weekly retrain
evaluates on a *different* (shifted) 90-day test window, so MAE wobbles by a
few percent on identical code.  Requiring strict improvement would reject
almost every honest retrain; the tolerance only blocks genuine regressions.

Exit codes (consumed by ``.github/workflows/retrain.yml``):

- ``0`` — promote: challenger is good (or there is no champion yet).
- ``1`` — REJECT: challenger regressed beyond tolerance; keep the old model.
- ``2`` — usage error (challenger metrics missing/unreadable).

Run::

    python -m src.promotion_gate CHAMPION_METRICS CHALLENGER_METRICS \
        [--max-regression-pct 5.0] [--summary-file $GITHUB_STEP_SUMMARY]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Outcome of one champion-vs-challenger comparison."""

    promote: bool
    reason: str
    champion_mae: float | None
    challenger_mae: float
    delta_pct: float | None  # +x% = challenger MAE is x% higher (worse)

    def markdown(self) -> str:
        verdict = "✅ PROMOTE" if self.promote else "❌ REJECT (keeping champion)"
        lines = [
            "## Champion/challenger gate",
            "",
            f"**Verdict: {verdict}** — {self.reason}",
            "",
            "| Model | Held-out test MAE |",
            "|---|---:|",
            f"| Champion (committed) | {self.champion_mae:.3f} |" if self.champion_mae is not None
            else "| Champion (committed) | — none — |",
            f"| Challenger (new) | {self.challenger_mae:.3f} |",
        ]
        if self.delta_pct is not None:
            lines.append(f"| Δ MAE | {self.delta_pct:+.2f}% |")
        return "\n".join(lines) + "\n"


def _test_mae(metrics: dict) -> float:
    """Extract the held-out test MAE from a metrics.json structure."""
    return float(metrics["model"]["test"]["mae"])


def evaluate_gate(
    champion: dict | None,
    challenger: dict,
    *,
    max_regression_pct: float = config.PROMOTION_MAX_MAE_REGRESSION_PCT,
) -> GateResult:
    """Decide whether the challenger may replace the champion.

    Parameters
    ----------
    champion :
        Parsed previous ``metrics.json``, or ``None`` when no committed
        champion exists yet (bootstrap: the challenger is promoted).
    challenger :
        Parsed freshly produced ``metrics.json``.
    max_regression_pct :
        Largest tolerated MAE increase, in percent of the champion's MAE.
    """
    challenger_mae = _test_mae(challenger)

    if champion is None:
        return GateResult(
            promote=True,
            reason="no champion metrics found — bootstrapping with the challenger",
            champion_mae=None,
            challenger_mae=challenger_mae,
            delta_pct=None,
        )

    champion_mae = _test_mae(champion)
    delta_pct = (challenger_mae - champion_mae) / champion_mae * 100.0
    limit = champion_mae * (1.0 + max_regression_pct / 100.0)

    if challenger_mae <= limit:
        reason = (
            f"challenger MAE {challenger_mae:.3f} vs champion {champion_mae:.3f} "
            f"({delta_pct:+.2f}%) is within the {max_regression_pct:.1f}% tolerance"
        )
        promote = True
    else:
        reason = (
            f"challenger MAE {challenger_mae:.3f} regressed {delta_pct:+.2f}% vs "
            f"champion {champion_mae:.3f}, beyond the {max_regression_pct:.1f}% tolerance"
        )
        promote = False

    return GateResult(
        promote=promote,
        reason=reason,
        champion_mae=champion_mae,
        challenger_mae=challenger_mae,
        delta_pct=delta_pct,
    )


def _load_metrics(path: Path) -> dict | None:
    """Load a metrics.json; ``None`` when absent (bootstrap-friendly)."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("champion", type=Path, help="Previous committed metrics.json (may not exist yet).")
    parser.add_argument("challenger", type=Path, help="Freshly produced metrics.json.")
    parser.add_argument(
        "--max-regression-pct",
        type=float,
        default=config.PROMOTION_MAX_MAE_REGRESSION_PCT,
        help="Largest tolerated MAE increase vs the champion, in percent.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=None,
        help="Append a Markdown verdict here (e.g. $GITHUB_STEP_SUMMARY).",
    )
    args = parser.parse_args(argv)

    challenger = _load_metrics(args.challenger)
    if challenger is None:
        logger.error("Challenger metrics not found at %s — did training run?", args.challenger)
        return 2

    champion = _load_metrics(args.champion)
    if champion is None:
        logger.warning("No champion metrics at %s — treating this as the first model.", args.champion)

    try:
        result = evaluate_gate(champion, challenger, max_regression_pct=args.max_regression_pct)
    except (KeyError, TypeError, ValueError) as exc:
        logger.error("Could not compare metrics files (%s) — malformed metrics.json?", exc)
        return 2

    (logger.info if result.promote else logger.error)("Gate verdict: %s — %s",
                                                      "PROMOTE" if result.promote else "REJECT", result.reason)

    if args.summary_file is not None:
        with args.summary_file.open("a", encoding="utf-8") as fh:
            fh.write(result.markdown())

    return 0 if result.promote else 1


if __name__ == "__main__":
    sys.exit(main())
