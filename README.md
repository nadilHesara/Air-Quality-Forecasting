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
├── experiments/           # Phase 6 model-improvement studies (offline, not in serving)
│   ├── error_analysis.py  # 6.1 where the model loses to persistence
│   ├── feature_experiments.py # 6.2 leakage-guarded feature candidates
│   ├── tune_lgbm.py       # 6.3 Optuna HPO (logged to MLflow)
│   ├── model_comparison.py# 6.4 LGBM vs XGBoost vs Ridge vs SARIMAX
│   └── multi_horizon.py   # 6.6 +1/+2/+3-day forecasts (stretch)
├── .github/workflows/
│   └── retrain.yml         # Weekly cron + manual retrain, commits model back
├── models/
│   └── model.joblib            (committed; point + p10/p50/p90 quantile models)
├── mlruns/                     # MLflow local tracking store (gitignored; uploaded as CI artifact)
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
- `GET http://localhost:8000/predict` — tomorrow's predicted PM2.5 **with an 80%
  prediction interval** (`pm25_lower` / `pm25_upper` / `interval_coverage`), the
  exact input features, and a model version string.
- Interactive docs: `http://localhost:8000/docs`

Example:

```bash
curl http://localhost:8000/predict
```

### Dashboard (Streamlit)

```bash
streamlit run src/dashboard.py
```

Shows tomorrow's predicted PM2.5 with its 80% prediction interval and a chart
of the last ~30 days of actual PM2.5 with the latest prediction (and its
lower/upper band) overlaid.

## Output

- **`data/training_data.parquet`** — ~1 060 rows × 27 columns.  Each row is
  one day with the raw weather + air-quality columns, the 26 engineered
  features (lags, rolling stats, calendar), and the target (`pm25_next_day`).

- **`reports/pm25_timeseries.png`** — time-series plot of daily PM2.5 with a
  30-day rolling-mean overlay for eyeballing seasonality.

## Model Performance & Prediction Intervals

On the held-out test the LightGBM point forecast beats persistence by **+4.5%
MAE / +7.5% RMSE** and seasonal-naive by ~39%. That persistence gap is *small
on purpose*: daily-mean PM2.5 has lag-1 autocorrelation ≈ **0.83**, so "tomorrow
≈ today" is a genuinely strong baseline. The Phase 6 study
([`reports/phase6_summary.md`](reports/phase6_summary.md)) shows that a linear
model, XGBoost, and SARIMAX all land within a ~5% MAE band of each other and of
persistence — the remaining gap is largely **intrinsic** to the series, not a
modelling shortfall.

The real value-add is honest **uncertainty**: the model ships three LightGBM
quantile regressors (p10/p50/p90) with **split-conformal calibration**, so the
80% band actually covers ~80% of held-out days (uncalibrated quantile LGBM
under-covers at ~68%). `GET /predict` and the dashboard surface this range.

Reproduce any study offline (needs `requirements-experiments.txt`):

```bash
pip install -r requirements.txt -r requirements-experiments.txt
python -m experiments.error_analysis      # 6.1
python -m experiments.feature_experiments  # 6.2
python -m experiments.tune_lgbm            # 6.3  (logs to MLflow experiment pm25-hpo)
python -m experiments.model_comparison     # 6.4
python -m experiments.multi_horizon        # 6.6
```

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

The tracking store is **local and file-based** (`mlruns/`), created on first
run and gitignored (MLflow's file store embeds the absolute artifact path at
creation time, so a `mlruns/` built on Windows would break on a Linux CI runner
and vice versa). `src/train.py` enables the file store automatically; to browse
runs with the MLflow UI you set the same opt-in (MLflow 3.x gates the file
backend behind it):

```bash
# Windows PowerShell
$env:MLFLOW_ALLOW_FILE_STORE = "true"; mlflow ui --backend-store-uri mlruns
# bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns
# then open http://localhost:5000
```

Each CI retrain run uploads its `mlruns/` as a downloadable build artifact
(retained 30 days) under **Actions → run → Artifacts**.

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

All features are built by `src/features.py` (`make_features`), the **single
source of truth** used identically at training and serving time.  There are
**26 features** in four groups, plus the target.  Every feature for day *t* uses
only information available up to and including day *t* (lags use `shift(k≥1)`;
rolling windows are trailing / non-centred), so there is no leakage — see the
leakage guarantee at the top of `src/features.py`.

**Pollution context (today, day *t*)**

| Feature | Description |
|---------|-------------|
| `pm2_5_mean` | Today's mean PM2.5 (µg/m³) |
| `pm10_mean` | Today's mean PM10 (µg/m³) |

**Weather (today, day *t*)**

| Feature | Description |
|---------|-------------|
| `temperature_2m_mean` | Daily mean 2 m air temperature |
| `wind_speed_10m_max` | Daily max 10 m wind speed |
| `wind_direction_10m_dominant` | Dominant 10 m wind direction (degrees) |
| `relative_humidity_2m_mean` | Daily mean 2 m relative humidity |
| `precipitation_sum` | Daily total precipitation |
| `surface_pressure_mean` | Daily mean surface pressure |

**Lagged & rolling PM2.5 (past only)**

| Feature | Description |
|---------|-------------|
| `pm25_lag1`, `pm25_lag2`, `pm25_lag3`, `pm25_lag7`, `pm25_lag14` | PM2.5 from *k* days ago (1/2/3/7/14; 7 = same weekday last week) |
| `pm25_rolling3_mean`, `pm25_rolling7_mean`, `pm25_rolling14_mean`, `pm25_rolling30_mean` | Trailing rolling mean of PM2.5 over 3/7/14/30 days |
| `pm25_rolling3_std`, `pm25_rolling7_std`, `pm25_rolling14_std`, `pm25_rolling30_std` | Trailing rolling std of PM2.5 over 3/7/14/30 days |
| `pm25_diff1` | Day-over-day change: yesterday's PM2.5 minus the day before |

**Calendar**

| Feature | Description |
|---------|-------------|
| `day_of_week` | 0 (Monday) – 6 (Sunday) |
| `month` | 1 – 12 |
| `day_of_year` | 1 – 366 |
| `is_weekend` | 1 if Saturday/Sunday, else 0 |

**Target**

| Column | Description |
|--------|-------------|
| `pm25_next_day` | **Target** — next day's mean PM2.5 (the only future-looking column; produced by `add_target`, never used at inference) |
