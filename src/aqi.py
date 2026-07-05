"""
Turn a PM2.5 number into a plain-English air-quality category + health advice.

This uses the **US EPA** PM2.5 breakpoints (the same 0–500 AQI scale most apps
show).  It is a small, dependency-free lookup so both the dashboard and the
alert helper can share one source of truth for "how bad is this number?".

The main entry point is :func:`classify`, which takes a PM2.5 value in µg/m³
(a daily mean, as this project predicts) and returns an :class:`AqiCategory`
with a name, a colour, a short health message, and whether it counts as
"unhealthy" (used by the alerting in :mod:`src.alert`).

Run a quick check::

    python -m src.aqi 42
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AqiCategory:
    """One air-quality band: what to call it and what to do about it."""

    name: str            # e.g. "Moderate"
    color: str           # a hex colour for the dashboard badge
    advice: str          # short, plain-English health guidance
    unhealthy: bool      # True once the air is bad for sensitive groups or worse


# EPA PM2.5 bands as (upper_limit_inclusive, category).  A value is placed in
# the first band whose upper limit it does not exceed.  The final band has no
# real upper limit (we use a very large number so anything extreme lands here).
_BANDS: list[tuple[float, AqiCategory]] = [
    (12.0, AqiCategory(
        "Good", "#009966",
        "Air quality is fine. Enjoy your normal outdoor activities.",
        unhealthy=False,
    )),
    (35.4, AqiCategory(
        "Moderate", "#ffde33",
        "Air quality is acceptable. Very sensitive people may want to take it easy outdoors.",
        unhealthy=False,
    )),
    (55.4, AqiCategory(
        "Unhealthy for Sensitive Groups", "#ff9933",
        "People with heart or lung problems, older adults, and children should limit long or intense outdoor activity.",
        unhealthy=True,
    )),
    (150.4, AqiCategory(
        "Unhealthy", "#cc0033",
        "Everyone may feel effects. Sensitive groups should avoid outdoor exertion; others should cut it down.",
        unhealthy=True,
    )),
    (250.4, AqiCategory(
        "Very Unhealthy", "#660099",
        "Health warning. Everyone should avoid outdoor activity and stay indoors where possible.",
        unhealthy=True,
    )),
    (float("inf"), AqiCategory(
        "Hazardous", "#7e0023",
        "Emergency conditions. Everyone should stay indoors and keep activity to a minimum.",
        unhealthy=True,
    )),
]


def classify(pm25: float) -> AqiCategory:
    """Return the air-quality category for a PM2.5 value (µg/m³, daily mean).

    Negative values are treated as 0 (the model can occasionally predict a
    slightly negative number for very clean air).
    """
    value = max(0.0, float(pm25))
    for upper, category in _BANDS:
        if value <= upper:
            return category
    # Unreachable because the last band's upper limit is +inf, but return the
    # worst band defensively rather than None.
    return _BANDS[-1][1]


if __name__ == "__main__":
    import sys

    pm = float(sys.argv[1]) if len(sys.argv) > 1 else 42.0
    cat = classify(pm)
    print(f"PM2.5 = {pm} µg/m³")
    print(f"Category: {cat.name}")
    print(f"Advice:   {cat.advice}")
    print(f"Unhealthy: {cat.unhealthy}")
