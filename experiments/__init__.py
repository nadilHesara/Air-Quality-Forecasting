"""Phase 6 model-improvement experiments (analysis, feature/HP search, model comparison).

These scripts are runnable offline against ``data/training_data.parquet`` and are
kept separate from the production ``src/`` pipeline. They import the split,
baseline, and feature helpers from ``src`` so there is no drifting second copy.
"""
