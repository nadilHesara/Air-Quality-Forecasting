"""Tests for the PM2.5 → air-quality-category helper and the alert builder."""

from __future__ import annotations

from src.alert import build_alert
from src.aqi import classify


# ── AQI classification ───────────────────────────────────────────────────────
def test_good_air_is_not_unhealthy() -> None:
    cat = classify(8.0)
    assert cat.name == "Good"
    assert cat.unhealthy is False


def test_moderate_band() -> None:
    cat = classify(20.0)
    assert cat.name == "Moderate"
    assert cat.unhealthy is False


def test_sensitive_group_band_is_unhealthy() -> None:
    cat = classify(42.0)
    assert cat.name == "Unhealthy for Sensitive Groups"
    assert cat.unhealthy is True


def test_boundary_value_stays_in_lower_band() -> None:
    # 12.0 is the top of "Good" (inclusive), so it must not spill into Moderate.
    assert classify(12.0).name == "Good"


def test_extreme_value_is_hazardous() -> None:
    assert classify(999.0).name == "Hazardous"


def test_negative_value_treated_as_clean_air() -> None:
    # The model can predict slightly negative on very clean days.
    assert classify(-3.0).name == "Good"


# ── Alert builder ────────────────────────────────────────────────────────────
def test_alert_fires_on_unhealthy_forecast() -> None:
    result = build_alert("2024-04-02", 80.0, "Colombo")
    assert result.alert is True
    assert "⚠️" in result.message
    assert "Colombo" in result.message


def test_no_alert_on_clean_forecast() -> None:
    result = build_alert("2024-04-02", 9.0, "Colombo")
    assert result.alert is False
    assert "✅" in result.message
