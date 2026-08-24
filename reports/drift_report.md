# Drift report

Checked: 2026-08-24T03:46:56+00:00  •  window: last 60 days

**Drift detected: YES**

- ⚠️ 21 feature(s) with PSI ≥ 0.25: pm25_rolling30_std, pm25_rolling14_mean, pm25_rolling14_std, pm25_rolling30_mean, surface_pressure_mean, pm10_mean, pm25_rolling3_mean, pm25_rolling7_mean, relative_humidity_2m_mean, wind_speed_10m_max, pm2_5_mean, pm25_rolling7_std, pm25_lag1, temperature_2m_mean, pm25_lag2, pm25_rolling3_std, pm25_lag3, pm25_lag14, pm25_lag7, wind_direction_10m_dominant, precipitation_sum

## Feature drift (PSI vs training reference)

Reference built: 2026-08-17T03:42:50+00:00  •  alerts: 21  •  warnings: 1

| Feature | PSI | Status |
|---|---:|---|
| pm25_rolling30_std | 5.4043 | alert |
| pm25_rolling14_mean | 4.7837 | alert |
| pm25_rolling14_std | 4.1545 | alert |
| pm25_rolling30_mean | 3.0727 | alert |
| surface_pressure_mean | 2.9829 | alert |
| pm10_mean | 2.8816 | alert |
| pm25_rolling3_mean | 2.2472 | alert |
| pm25_rolling7_mean | 2.0645 | alert |
| relative_humidity_2m_mean | 1.9622 | alert |
| wind_speed_10m_max | 1.9057 | alert |
| pm2_5_mean | 1.7681 | alert |
| pm25_rolling7_std | 1.5563 | alert |
| pm25_lag1 | 1.4507 | alert |
| temperature_2m_mean | 1.2274 | alert |
| pm25_lag2 | 1.0990 | alert |
| pm25_rolling3_std | 0.8496 | alert |
| pm25_lag3 | 0.8278 | alert |
| pm25_lag14 | 0.8024 | alert |
| pm25_lag7 | 0.7958 | alert |
| wind_direction_10m_dominant | 0.5934 | alert |
| precipitation_sum | 0.5616 | alert |
| pm25_diff1 | 0.1463 | warn |

## Prediction-error drift

| Metric | Value |
|---|---:|
| Recent MAE (60 days, 2026-04-14 → 2026-06-12) | 2.688 |
| Committed test MAE | 3.426 |
| Ratio | 0.79× (alert > 1.5×) |
