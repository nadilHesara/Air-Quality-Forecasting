"""
Fetch daily weather data from the Open-Meteo Historical Weather API.

The archive API (archive-api.open-meteo.com) provides gap-free reanalysis
data from 1940 onwards.  We request **daily** aggregations directly from the
API (temperature_2m_mean, wind_speed_10m_max, etc.) so no hourly→daily
aggregation is needed on our side.

Usage
-----
>>> from src.fetch_weather import fetch_daily_weather
>>> df = fetch_daily_weather(6.9271, 79.8612, "2023-01-01", "2023-12-31")
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Default timeout for HTTP requests (seconds)
_REQUEST_TIMEOUT: int = 120


def fetch_daily_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    *,
    api_url: str = "https://archive-api.open-meteo.com/v1/archive",
    daily_variables: list[str] | None = None,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetch daily weather data from the Open-Meteo archive API.

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
        Base URL for the archive API.
    daily_variables : list[str] | None, optional
        Weather variables to request.  Defaults to a standard set suitable
        for air-quality forecasting.
    timezone : str, optional
        Timezone for the returned timestamps (default ``"UTC"``).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``date`` (datetime64) with one column per
        requested variable.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-2xx status code.
    KeyError
        If the response JSON is missing expected fields.
    """
    if daily_variables is None:
        daily_variables = [
            "temperature_2m_mean",
            "wind_speed_10m_max",
            "wind_direction_10m_dominant",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "surface_pressure_mean",
        ]

    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(daily_variables),
        "timezone": timezone,
    }

    logger.info(
        "Fetching daily weather for (%.4f, %.4f) from %s to %s …",
        latitude,
        longitude,
        start_date,
        end_date,
    )

    response = requests.get(api_url, params=params, timeout=_REQUEST_TIMEOUT)
    response.raise_for_status()
    data: dict[str, Any] = response.json()

    daily_data: dict[str, Any] = data["daily"]
    df = pd.DataFrame(daily_data)
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date"}).set_index("date")

    logger.info("Weather data: %d rows, columns=%s", len(df), list(df.columns))
    return df
