# Phase 6.3 — Optuna hyper-parameter search

- Features: `baseline+wind_vector`
- Trials: 60 (TPE, seed 42)
- Scoring: expanding-window TimeSeriesSplit CV, train pool only
- MLflow experiment: `pm25-hpo`

**Baseline CV MAE:** 4.9421 ± 2.139  
**Best tuned CV MAE:** 4.6671  
**Δ:** -0.2750

## Best params

```python
LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.0189756973497472,
    "num_leaves": 44,
    "max_depth": 3,
    "min_child_samples": 18,
    "subsample": 0.8898705283830923,
    "colsample_bytree": 0.9664329660605866,
    "reg_alpha": 0.2594797777901213,
    "reg_lambda": 0.0892155202631994,
    "subsample_freq": 1,
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}
```
