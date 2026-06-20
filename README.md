# PM2.5 Air-Quality Forecasting

End-to-end pipeline for **next-day PM2.5 prediction** in Colombo, Sri Lanka:
data ingestion → feature engineering → training & evaluation → serving (API +
dashboard) → MLOps (experiment tracking + automated retraining).  Designed to
be easily re-targeted to any city by editing `config.py`.

## Data Sources

| Dataset | API | Resolution | Variables |
|---------|-----|------------|-----------|
| Historical weather | [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) (`archive-api.open-meteo.com`) | Daily, 0.1–0.25° | Temperature, wind speed/direction, humidity, precipitation, surface pressure |
| Air quality (PM2.5 / PM10) | [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) (`air-quality-api.open-meteo.com`) | Hourly → aggregated to daily | PM2.5, PM10 (CAMS model) |

Both APIs are **free** and require **no API key** for non-commercial use.

## Project Structure

```
Air-Quality-Forecasting/
├── config.py               # Location, dates, API URLs, schedule, MLflow settings
├── requirements.txt
├── src/
│   ├── fetch_weather.py    # Reusable weather data fetcher
│   ├── fetch_air_quality.py# Reusable air-quality fetcher
│   ├── build_dataset.py    # Join + feature engineering + save
│   ├── features.py         # Single source of truth for features (train + serve)
│   ├── train.py            # End-to-end: fetch → features → train → evaluate → save (+ MLflow)
│   ├── inference.py        # Shared serving path (fetch → features → predict)
│   ├── api.py              # FastAPI app (/health, /predict)
│   └── dashboard.py        # Streamlit dashboard
├── scripts/
│   └── eda.py              # Time-series plot of PM2.5
├── .github/workflows/
│   └── retrain.yml         # Weekly cron + manual retrain, commits model back
├── models/
│   └── model.joblib            (committed; refreshed by src.train / retrain CI)
├── mlruns/                     # MLflow local tracking store (committed)
├── data/
│   └── training_data.parquet   (generated)
└── reports/
    ├── metrics.json            (generated)
    ├── results.md              (generated)
    ├── shap_summary.png        (generated)
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

# 3. Train end-to-end — fetches latest data, builds features, trains,
#    evaluates vs baselines, saves the model, and logs an MLflow run.
python -m src.train

# (optional) re-run on the already-fetched parquet, no network:
python -m src.train --use-cached

# (optional) generate the EDA plot (needs the parquet from a prior run)
python scripts/eda.py
```

`src/train.py` is the **single reproducible entry point**. One command takes
you from nothing to a fresh, evaluated, persisted model:

1. **fetch** latest weather + air-quality for the configured city/date range,
2. **build features + target** (the same `src/features.py` used at serving time),
3. **train** a LightGBM regressor (time-series CV on the train pool only),
4. **evaluate** once on a held-out test set vs persistence & seasonal-naive baselines,
5. **save** `models/model.joblib`, `reports/metrics.json`, `reports/results.md`,
   `reports/shap_summary.png`, and log everything to MLflow.

`python -m src.build_dataset` still exists if you only want to (re)build the
training parquet without training.

## Serving (API + Dashboard)

The serving layer reuses the **same** fetch functions and the **same**
`src/features.py` feature builder used in training, so the schema the model
sees in production can't drift.  Both surfaces share one prediction path in
`src/inference.py` (fetch live data → build one feature row → predict
tomorrow's PM2.5).

> Prerequisite: `models/model.joblib` must exist — run `python -m src.train`
> first.  The API reports `"degraded"` on `/health` and returns `503` from
> `/predict` if the model file is missing.

### REST API (FastAPI + Uvicorn)

```bash
uvicorn src.api:app --reload --port 8000
```

- `GET http://localhost:8000/health` — liveness/readiness (model present & loadable).
- `GET http://localhost:8000/predict` — tomorrow's predicted PM2.5, the exact
  input features, and a model version string.
- Interactive docs: `http://localhost:8000/docs`

Example:

```bash
curl http://localhost:8000/predict
```

### Dashboard (Streamlit)

```bash
streamlit run src/dashboard.py
```

Shows tomorrow's predicted PM2.5 and a chart of the last ~30 days of actual
PM2.5 with the latest prediction overlaid.

## Output

- **`data/training_data.parquet`** — ~1 090 rows × 13 columns.  Each row is
  one day with weather features, air-quality features, engineered lag/rolling
  features, and the target (`pm25_next_day`).

- **`reports/pm25_timeseries.png`** — time-series plot of daily PM2.5 with a
  30-day rolling-mean overlay for eyeballing seasonality.

## MLOps — Experiment Tracking & Automated Retraining

### Experiment tracking (MLflow)

Every `python -m src.train` run is wrapped in an [MLflow](https://mlflow.org/)
run that logs:

- **Params** — model type & hyper-parameters, city/lat/lon, date range, feature
  count, CV/test configuration, and the trained-through date.
- **Metrics** — the model's MAE/RMSE **and** the persistence / seasonal-naive
  baseline numbers, the cross-validation MAE/RMSE (mean ± std), and the
  % improvement over each baseline.
- **Artifacts** — the SHAP summary plot, `metrics.json`, `results.md`, and the
  serialized `model.joblib`.
- **Model** — logged via `mlflow.sklearn.log_model` with an inferred signature
  and input example.

The tracking store is **local and file-based**, committed to the repo under
[`mlruns/`](mlruns/), so experiment history travels with the code and CI runs
append to it. `src/train.py` enables the file store automatically; to browse
runs with the MLflow UI you set the same opt-in (MLflow 3.x gates the file
backend behind it):

```bash
# Windows PowerShell
$env:MLFLOW_ALLOW_FILE_STORE = "true"; mlflow ui --backend-store-uri mlruns
# bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns
# then open http://localhost:5000
```

#### Pointing MLflow at a remote tracking server

Nothing in the code is hardcoded to the local store. To log to a remote
[MLflow Tracking Server](https://mlflow.org/docs/latest/tracking/server.html)
(or a managed one such as Databricks), set the `MLFLOW_TRACKING_URI`
environment variable — it overrides the local default in `config.py`:

```bash
# Local / self-hosted tracking server
export MLFLOW_TRACKING_URI="http://my-mlflow-server:5000"
python -m src.train

# Or a managed backend, e.g. Databricks
export MLFLOW_TRACKING_URI="databricks"
```

In CI, set it as the repository variable **`MLFLOW_TRACKING_URI`** (Settings →
Secrets and variables → Actions → Variables); the retrain workflow already
passes it through. When a remote store is used you can stop committing
`mlruns/` and remove it from the retrain commit step.

### Automated retraining (GitHub Actions)

[`.github/workflows/retrain.yml`](.github/workflows/retrain.yml) retrains the
model on a schedule and on demand:

- **Schedule** — weekly cron (the canonical schedule lives in `config.py` as
  `RETRAIN_CRON`; a guard step in the workflow fails if the YAML cron drifts
  from it).
- **Manual** — the *Run workflow* button (`workflow_dispatch`).

Each run sets up Python, installs `requirements.txt`, prints the run
configuration (city / dates / schedule read from `config.py`), runs
`python -m src.train` end-to-end, and **commits the refreshed
`models/model.joblib`, `reports/`, and `mlruns/` back to the branch** — but only
if something actually changed, so re-runs don't create empty commits. The
commit message carries `[skip ci]` to avoid retrigger loops.

The workflow needs `contents: write` permission (already declared) so the
`github-actions[bot]` can push. If you'd rather **not** commit binaries back,
swap the final commit step for an
[`actions/upload-artifact`](https://github.com/actions/upload-artifact) step to
publish the model + reports as downloadable build artifacts instead.

## Changing the Target City

Edit `config.py`:

```python
CITY_NAME  = "Mumbai"
LATITUDE   = 19.0760
LONGITUDE  = 72.8777
START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"
```

Then re-run `python -m src.train`.

## Engineered Features

| Feature | Description |
|---------|-------------|
| `pm25_lag1` | Previous day's mean PM2.5 |
| `pm25_rolling7` | 7-day rolling mean PM2.5 |
| `day_of_week` | 0 (Monday) – 6 (Sunday) |
| `month` | 1 – 12 |
| `pm25_next_day` | **Target** — next day's mean PM2.5 |
