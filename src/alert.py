"""
Warn when tomorrow's predicted PM2.5 crosses into unhealthy air.

This is deliberately small and side-effect-free: :func:`check_alert` runs the
normal prediction, asks :mod:`src.aqi` whether the result is unhealthy, and
returns a plain result you can act on however you like — show it on the
dashboard, print it, or (in CI) turn it into an email/Slack message.

We do **not** hard-wire an email or push provider here, because that needs
secrets and a running service.  Instead the CLI prints the message and exits
with code ``1`` when an alert fires, so a scheduled GitHub Action (or cron) can
trigger any notifier it likes off that exit code::

    python -m src.alert        # prints today's outlook; exit 1 if unhealthy
"""

from __future__ import annotations

from dataclasses import dataclass

from src.aqi import AqiCategory, classify
from src.inference import predict_next_day


@dataclass
class AlertResult:
    """The outcome of an alert check for one day's forecast."""

    prediction_date: str
    predicted_pm25: float
    category: AqiCategory
    alert: bool          # True when the air is unhealthy (worth warning about)
    message: str         # a ready-to-send, human-readable line


def build_alert(prediction_date: str, predicted_pm25: float, city: str) -> AlertResult:
    """Turn a (date, PM2.5) pair into an :class:`AlertResult` — no network.

    Kept separate from :func:`check_alert` so it can be unit-tested without
    fetching live data.
    """
    category = classify(predicted_pm25)
    if category.unhealthy:
        message = (
            f"⚠️ {city}: tomorrow ({prediction_date}) PM2.5 is forecast at "
            f"{predicted_pm25:.1f} µg/m³ — {category.name}. {category.advice}"
        )
    else:
        message = (
            f"✅ {city}: tomorrow ({prediction_date}) PM2.5 is forecast at "
            f"{predicted_pm25:.1f} µg/m³ — {category.name}. No alert."
        )
    return AlertResult(
        prediction_date=prediction_date,
        predicted_pm25=predicted_pm25,
        category=category,
        alert=category.unhealthy,
        message=message,
    )


def check_alert() -> AlertResult:
    """Run the live prediction and build the alert for tomorrow."""
    result = predict_next_day()
    return build_alert(
        prediction_date=result.prediction_date.isoformat(),
        predicted_pm25=result.predicted_pm25,
        city=result.city,
    )


if __name__ == "__main__":
    import sys

    outcome = check_alert()
    print(outcome.message)
    # Non-zero exit lets a scheduler notify only when the air is unhealthy.
    sys.exit(1 if outcome.alert else 0)
