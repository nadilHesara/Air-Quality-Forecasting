# PM2.5 Air-Quality Forecasting

Predicts **tomorrow's air pollution (PM2.5)** for Colombo, Sri Lanka. It pulls
live weather and air data, builds features, trains a model, and shows the
forecast on a web page. You can point it at any city by editing `config.py`.

## 🌐 See it live

**👉 https://colombo-air-forecast.streamlit.app/**

Open that link in any browser to see tomorrow's forecast and a chart of the
last 30 days. Click **🔄 Refresh** for the latest numbers. No install needed.

> The app sleeps after a few days with no visitors. If you see a "waking up"
> screen, wait about 30 seconds and it loads.

## What's inside

The full flow: get data → build features → train a model → check it → serve it
(web page + API) → retrain it every week on its own, safely.

- **Model:** LightGBM, predicts next-day mean PM2.5.
- **Data:** free [Open-Meteo](https://open-meteo.com/) weather + air-quality
  APIs (no API key). Weather is daily; air quality is hourly, rolled up to
  daily. Both are joined on the date.
- **Uncertainty:** the model also gives a low/high range (an 80% band), not
  just a single number.

## Run it on your own computer

**Easiest way — Docker** (needs [Docker Desktop](https://www.docker.com/products/docker-desktop/), no Python):

```bash
docker compose up --build
```

Then open:
- Web page → **http://localhost:8501**
- API docs → **http://localhost:8000/docs**

Press **Ctrl + C** to stop.

**With Python instead:**

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux
pip install -r requirements.txt

python -m src.train             # train the model (fetches data, saves model)
streamlit run src/dashboard.py  # open the web page
uvicorn src.api:app --port 8000 # or run the API
```

`src/train.py` is the one command that does everything: fetch → check data →
build features → train → test against baselines → save the model and reports →
log to MLflow.

## The two ways to use it

Both share the **same** feature code and the **same** prediction path
(`src/inference.py`), so the model always sees the same kind of input.

**Web page (Streamlit)** — `src/dashboard.py`. Shows tomorrow's PM2.5, its
low/high range, and the last 30 days as a chart. It also shows:
- a **health badge** — the air-quality category (Good, Moderate, Unhealthy…)
  with a short health tip (`src/aqi.py`), and
- **"Why this number?"** — the top features that pushed tomorrow's forecast up
  or down, using SHAP, so the prediction isn't a black box.

**API (FastAPI)** — `src/api.py`:
- `GET /health` and `GET /ready` — is the model loaded and ready?
- `GET /predict` — tomorrow's PM2.5 with the low/high range, the air-quality
  category + health advice, and the input features. Docs at `/docs`.

The API caches each prediction for 15 minutes (the data only changes daily). If
the data source goes down, it serves the last good answer instead of failing.
Logs are one JSON line each, so hosts can read them easily.

**Unhealthy-day alert** — `python -m src.alert` prints tomorrow's outlook and
exits with code `1` if the air will be unhealthy, so a scheduled job can send an
email or Slack message off that exit code.

## Change the city

You don't need to touch the code — set environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CITY_NAME` | `Colombo` | City name to show |
| `LATITUDE` | `6.9271` | City latitude |
| `LONGITUDE` | `79.8612` | City longitude |
| `MLFLOW_TRACKING_URI` | local `mlruns/` | Remote MLflow server (optional) |
| `MODEL_SOURCE` | `local` | `local` = saved file, `registry` = MLflow model |

```bash
CITY_NAME=Mumbai LATITUDE=19.076 LONGITUDE=72.8777 docker compose up
```

To also change the training date range, edit `config.py` (`START_DATE` /
`END_DATE`) and re-run `python -m src.train`.

## Deploy it yourself

The live app runs on **Streamlit Community Cloud** (free):

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. **Create app** → pick this repo, branch `main`, main file `src/dashboard.py`.
4. Pick a URL name and click **Deploy**. First build takes a few minutes.

The saved model (`models/model.joblib`) is in the repo, so the app runs with no
extra setup. Every push to `main` redeploys it.

Want the API online too? It ships as one Docker image (`Dockerfile`), so any
container host works — **Render**, **Railway**, or **Fly.io**. It listens on
port `8000`; use `/ready` as the readiness check.

## How good is the model?

Next-day PM2.5 is a hard thing to beat, because "tomorrow ≈ today" is already a
strong guess (day-to-day PM2.5 is highly correlated). So the model only edges
past that simple baseline on average — and we're honest about it.

Over a full walk-forward backtest (24 rounds, 693 predictions):

| Method | MAE | RMSE |
|--------|-----|------|
| Model (LightGBM) | 4.23 | 5.82 |
| Persistence ("tomorrow = today") | 4.16 | 5.90 |
| Seasonal-naive | ~7.5 | — |

The real value is: it clearly beats the seasonal-naive baseline (~44%), has
slightly better RMSE (fewer big misses), and — most useful — it gives a
**calibrated low/high range** (three quantile models, p10/p50/p90, tuned so the
80% band really covers ~80% of days). See
[`reports/phase6_summary.md`](reports/phase6_summary.md) and
[`reports/backtest.md`](reports/backtest.md).

## MLOps — what keeps it safe and fresh

- **Data check** (`src/validate.py`) — before any training, checks the data:
  right columns, enough rows, sane values, no big date gaps. Bad data stops
  the run.
- **Retrain every week** (`.github/workflows/retrain.yml`) — GitHub Actions
  fetches fresh data and retrains on a schedule (or on a button click).
- **Quality gate** (`src/promotion_gate.py`) — a new model is only saved if it
  isn't clearly worse than the current one. A bad retrain can't replace a good
  model.
- **Drift watch** (`src/drift.py`) — checks if the incoming data has shifted
  (season-aware) or if recent errors are rising, and opens a GitHub issue if so.
- **Backtest** (`src/backtest.py`) — replays history the way production runs, so
  the reported numbers are honest.
- **Experiment tracking** (MLflow) — every training run logs its params,
  metrics, and model. Local by default; point `MLFLOW_TRACKING_URI` at a remote
  server to enable the Model Registry (see `scripts/start_mlflow_server.*`).

## Project layout

```
Air-Quality-Forecasting/
├── config.py               # All settings (city, dates, APIs) — env-overridable
├── Dockerfile              # One image for the API + web page
├── docker-compose.yml      # Runs the API (8000) and web page (8501) together
├── src/
│   ├── fetch_weather.py / fetch_air_quality.py   # Get the data
│   ├── build_dataset.py    # Join data + save
│   ├── features.py         # Feature code (used by both train and serve)
│   ├── validate.py         # Data checks before training
│   ├── train.py            # Fetch → check → train → test → save (+ MLflow)
│   ├── promotion_gate.py   # Only keep a new model if it's not worse
│   ├── drift.py            # Watch for data / error drift
│   ├── backtest.py         # Walk-forward history replay
│   ├── registry.py         # MLflow Model Registry helpers
│   ├── inference.py        # Shared predict path (fetch → features → predict + explain)
│   ├── aqi.py              # PM2.5 → health category + advice
│   ├── alert.py            # Warn when tomorrow is unhealthy
│   ├── api.py              # FastAPI app (/health, /ready, /predict)
│   └── dashboard.py        # Streamlit web page
├── docs/architecture.md    # Diagram + how the pieces fit together
├── experiments/            # Offline model-improvement studies
├── scripts/                # EDA plot + MLflow server helpers
├── .github/workflows/      # ci.yml (tests) + retrain.yml (weekly retrain)
├── models/model.joblib     # The saved model (committed)
├── LICENSE                 # MIT
├── CONTRIBUTING.md
└── reports/                # Metrics, plots, backtest, drift (generated)
```

For a diagram of how data flows from the APIs to the forecast, see
[`docs/architecture.md`](docs/architecture.md).

## Features used by the model

All features come from `src/features.py`, used the same way in training and
serving. There are **26 features**, plus the target. Every feature for day *t*
only uses info up to and including day *t* (lags shift back, rolling windows
look backward), so there's no leakage.

**Today's values (day *t*)**

| Feature | Meaning |
|---------|---------|
| `pm2_5_mean`, `pm10_mean` | Today's mean PM2.5 / PM10 (µg/m³) |
| `temperature_2m_mean` | Mean air temperature |
| `wind_speed_10m_max` | Max wind speed |
| `wind_direction_10m_dominant` | Main wind direction (degrees) |
| `relative_humidity_2m_mean` | Mean humidity |
| `precipitation_sum` | Total rain |
| `surface_pressure_mean` | Mean surface pressure |

**Past PM2.5 patterns**

| Feature | Meaning |
|---------|---------|
| `pm25_lag1` … `pm25_lag14` | PM2.5 from 1/2/3/7/14 days ago |
| `pm25_rolling{3,7,14,30}_mean` | Backward rolling mean of PM2.5 |
| `pm25_rolling{3,7,14,30}_std` | Backward rolling std of PM2.5 |
| `pm25_diff1` | Change from two days ago to yesterday |

**Calendar**

| Feature | Meaning |
|---------|---------|
| `day_of_week` | 0 (Mon) – 6 (Sun) |
| `month` | 1 – 12 |
| `day_of_year` | 1 – 366 |
| `is_weekend` | 1 on Sat/Sun, else 0 |

**Target**

| Column | Meaning |
|--------|---------|
| `pm25_next_day` | Next day's mean PM2.5 — the only future-looking column, never used as input |

## Contributing & License

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for how to set
up and what to run before a pull request. Released under the
[MIT License](LICENSE).
