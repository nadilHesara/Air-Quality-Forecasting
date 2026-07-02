# Start a persistent, registry-capable MLflow tracking server (Windows).
#
# Unlike the ad-hoc mlruns/ file store, this backend is a SQLite database —
# which is what enables the MLflow **Model Registry** (versions + aliases)
# that src/registry.py uses. Runs and artifacts survive across machines and
# reclones because nothing lives inside the repo's gitignored mlruns/.
#
# Usage:
#   .\scripts\start_mlflow_server.ps1          # http://127.0.0.1:5000
#
# Then point the pipeline at it (per shell, or set it permanently):
#   $env:MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
#   python -m src.train
#
# In CI, set the repository variable MLFLOW_TRACKING_URI to your server's URL
# (Settings -> Secrets and variables -> Actions -> Variables); retrain.yml
# already passes it through and promotes registered models after the gate.

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

mlflow server `
    --backend-store-uri "sqlite:///mlflow.db" `
    --artifacts-destination "./mlflow-artifacts" `
    --host 127.0.0.1 `
    --port 5000
