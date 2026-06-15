# PM2.5 Air-Quality Forecasting — Data Ingestion

Data pipeline that builds a clean daily training table for **next-day PM2.5
prediction** in Colombo, Sri Lanka.  Designed to be easily re-targeted to any
city by editing `config.py`.

## Data Sources

| Dataset | API | Resolution | Variables |
|---------|-----|------------|-----------|
| Historical weather | [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) (`archive-api.open-meteo.com`) | Daily, 0.1–0.25° | Temperature, wind speed/direction, humidity, precipitation, surface pressure |
| Air quality (PM2.5 / PM10) | [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (`air-quality-api.open-meteo.com`) | Hourly → aggregated to daily | PM2.5, PM10 (CAMS model) |

Both APIs are **free** and require **no API key** for non-commercial use.

## Project Structure

```
Air-Quality-Forecasting/
├── config.py               # Location, dates, API URLs
├── requirements.txt
├── src/
│   ├── fetch_weather.py    # Reusable weather data fetcher
│   ├── fetch_air_quality.py# Reusable air-quality fetcher
│   └── build_dataset.py    # Join + feature engineering + save
├── scripts/
│   └── eda.py              # Time-series plot of PM2.5
├── data/
│   └── training_data.parquet   (generated)
└── reports/
    └── pm25_timeseries.png     (generated)
```

## Quick Start

```bash
# 1. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fetch data & build training table
python -m src.build_dataset

# 4. Generate the EDA plot
python scripts/eda.py
```

## Output

- **`data/training_data.parquet`** — ~1 090 rows × 13 columns.  Each row is
  one day with weather features, air-quality features, engineered lag/rolling
  features, and the target (`pm25_next_day`).

- **`reports/pm25_timeseries.png`** — time-series plot of daily PM2.5 with a
  30-day rolling-mean overlay for eyeballing seasonality.

## Changing the Target City

Edit `config.py`:

```python
CITY_NAME  = "Mumbai"
LATITUDE   = 19.0760
LONGITUDE  = 72.8777
START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"
```

Then re-run steps 3–4 above.

## Engineered Features

| Feature | Description |
|---------|-------------|
| `pm25_lag1` | Previous day's mean PM2.5 |
| `pm25_rolling7` | 7-day rolling mean PM2.5 |
| `day_of_week` | 0 (Monday) – 6 (Sunday) |
| `month` | 1 – 12 |
| `pm25_next_day` | **Target** — next day's mean PM2.5 |
