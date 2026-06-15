"""
Fetch daily air-quality data from the Open-Meteo Air Quality API.

The air-quality endpoint returns **hourly** PM2.5 / PM10 values.  This
module aggregates them to daily means so the output aligns with the daily
weather data produced by :mod:`src.fetch_weather`.

Usage
-----
>>> from src.fetch_air_quality import fetch_daily_air_quality
>>> df = fetch_daily_air_quality(6.9271, 79.8612, "2023-01-01", "2023-12-31")
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Default timeout for HTTP requests (seconds)
_REQUEST_TIMEOUT: int = 120


def fetch_daily_air_quality(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    *,
    api_url: str = "https://air-quality-api.open-meteo.com/v1/air-quality",
    hourly_variables: list[str] | None = None,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetch hourly air-quality data and aggregate to daily means.

    Parameters
    ----------
    latitude : float
        Decimal latitude of the target location.
    longitude : float
        Decimal longitude of the target location.
    start_date : str
        Start date in ISO-8601 format (``YYYY-MM-DD``), inclusive.
    end_date : str
        End date in ISO-8601 format (``YYYY-MM-DD``), inclusive.
    api_url : str, optional
        Base URL for the air-quality API.
    hourly_variables : list[str] | None, optional
        Pollutant variables to request.  Defaults to ``["pm2_5", "pm10"]``.
    timezone : str, optional
        Timezone for the returned timestamps (default ``"UTC"``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``date`` (datetime64) with columns named
        ``<variable>_mean`` (e.g. ``pm2_5_mean``, ``pm10_mean``).

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code.
    KeyError
        If the response JSON is missing expected fields.
    """
    if hourly_variables is None:
        hourly_variables = ["pm2_5", "pm10"]

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(hourly_variables),
        "timezone": timezone,
    }

    logger.info(
        "Fetching hourly air quality for (%.4f, %.4f) from %s to %s …",
        latitude,
        longitude,
        start_date,
        end_date,
    )

    response = requests.get(api_url, params=params, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    hourly_data: dict[str, Any] = data["hourly"]
    df = pd.DataFrame(hourly_data)
    df["time"] = pd.to_datetime(df["time"])

    # ── Aggregate hourly → daily mean ────────────────────────────────────
    df = df.set_index("time")
    daily = df.resample("D").mean()
    daily.index.name = "date"

    # Rename columns to make aggregation explicit (e.g. pm2_5 → pm2_5_mean)
    daily = daily.rename(columns={col: f"{col}_mean" for col in hourly_variables})

    logger.info(
        "Air-quality data: %d daily rows, columns=%s", len(daily), list(daily.columns)
    )
    return daily
