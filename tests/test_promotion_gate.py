"""
Tests for the champion/challenger promotion gate (src/promotion_gate.py, 7.2).

The contract: a retrain that produces a worse model (beyond tolerance) must
NOT be promoted — the CLI exits 1 so the CI workflow fails before committing.
Equal-or-better challengers, small within-tolerance wobbles, and the very
first model (no champion yet) all pass.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.promotion_gate import evaluate_gate, main


def _metrics(mae: float) -> dict:
    return {"model": {"test": {"mae": mae, "rmse": mae * 1.4}}}


# ── Decision logic ────────────────────────────────────────────────────────────
def test_better_challenger_promotes() -> None:
    result = evaluate_gate(_metrics(4.0), _metrics(3.5), max_regression_pct=5.0)
    assert result.promote
    assert result.delta_pct is not None and result.delta_pct < 0


def test_within_tolerance_promotes() -> None:
    # +4% regression against a 5% tolerance → still promoted (test windows
    # shift week to week; small wobbles are expected noise).
    result = evaluate_gate(_metrics(4.0), _metrics(4.16), max_regression_pct=5.0)
    assert result.promote


def test_regression_beyond_tolerance_rejects() -> None:
    result = evaluate_gate(_metrics(4.0), _metrics(4.5), max_regression_pct=5.0)
    assert not result.promote
    assert "regressed" in result.reason


def test_no_champion_bootstraps() -> None:
    result = evaluate_gate(None, _metrics(4.0))
    assert result.promote
    assert result.champion_mae is None


def test_markdown_renders_both_verdicts() -> None:
    good = evaluate_gate(_metrics(4.0), _metrics(3.5))
    bad = evaluate_gate(_metrics(4.0), _metrics(9.0))
    assert "PROMOTE" in good.markdown()
    assert "REJECT" in bad.markdown()


# ── CLI exit codes (what retrain.yml consumes) ───────────────────────────────
def _write(path: Path, metrics: dict) -> Path:
    path.write_text(json.dumps(metrics), encoding="utf-8")
    return path


def test_cli_promotes_and_writes_summary(tmp_path: Path) -> None:
    champion = _write(tmp_path / "champion.json", _metrics(4.0))
    challenger = _write(tmp_path / "challenger.json", _metrics(3.8))
    summary = tmp_path / "summary.md"

    code = main([str(champion), str(challenger), "--summary-file", str(summary)])

    assert code == 0
    assert "PROMOTE" in summary.read_text(encoding="utf-8")


def test_cli_rejects_regression(tmp_path: Path) -> None:
    champion = _write(tmp_path / "champion.json", _metrics(4.0))
    challenger = _write(tmp_path / "challenger.json", _metrics(6.0))

    assert main([str(champion), str(challenger)]) == 1


def test_cli_missing_champion_bootstraps(tmp_path: Path) -> None:
    challenger = _write(tmp_path / "challenger.json", _metrics(4.0))

    assert main([str(tmp_path / "nope.json"), str(challenger)]) == 0


def test_cli_missing_challenger_is_usage_error(tmp_path: Path) -> None:
    champion = _write(tmp_path / "champion.json", _metrics(4.0))

    assert main([str(champion), str(tmp_path / "nope.json")]) == 2


def test_cli_malformed_challenger_is_usage_error(tmp_path: Path) -> None:
    champion = _write(tmp_path / "champion.json", _metrics(4.0))
    challenger = _write(tmp_path / "challenger.json", {"model": {}})

    assert main([str(champion), str(challenger)]) == 2
