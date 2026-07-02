"""
Tests for the MLflow Model Registry integration (src/registry.py, Phase 7.3).

The registry needs a database-backed store, so the round-trip test spins up a
throwaway SQLite tracking backend in tmp_path — still fully offline, no
server process involved.  The contracts under test:

- backend capability detection (plain file store → no registry);
- register → @staging alias → promote → @production alias;
- loading the production bundle returns the full serving dict that
  ``src/inference.py`` expects (not just the bare registered model).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import mlflow
import pytest
from sklearn.dummy import DummyRegressor

from src.registry import (
    load_production_bundle,
    promote_staging_to_production,
    register_run_model,
    registry_capable,
)


def test_registry_capability_detection() -> None:
    # The plain file store has no registry tables.
    assert not registry_capable("file:///somewhere/mlruns")
    assert not registry_capable("/somewhere/mlruns")
    # Database-backed and remote stores do.
    assert registry_capable("sqlite:///mlflow.db")
    assert registry_capable("postgresql://user:pw@host/db")
    assert registry_capable("http://mlflow.internal:5000")
    assert registry_capable("databricks")


def test_register_is_a_noop_on_file_store(monkeypatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///nowhere/mlruns")
    assert register_run_model("does-not-matter") is None


@pytest.fixture
def sqlite_store(tmp_path: Path, monkeypatch):
    """A throwaway registry-capable tracking backend (SQLite in tmp_path)."""
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    previous = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(previous)


def test_register_promote_load_roundtrip(sqlite_store: str, tmp_path: Path) -> None:
    exp_id = mlflow.create_experiment(
        "registry-test", artifact_location=(tmp_path / "artifacts").as_uri()
    )

    # A minimal stand-in for what train.py produces: a logged sklearn model
    # (the registered artifact) + the full serving bundle on the same run.
    est = DummyRegressor(strategy="constant", constant=5.0).fit([[0.0]], [5.0])
    bundle_path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": est,
            "feature_names": ["x"],
            "target": "y",
            "trained_through": "2024-01-01",
        },
        bundle_path,
    )

    with mlflow.start_run(experiment_id=exp_id) as run:
        mlflow.log_artifact(str(bundle_path), artifact_path="model_bundle")
        mlflow.sklearn.log_model(est, name="model", serialization_format="cloudpickle")

    # Register → version 1 carries the @staging alias.
    version = register_run_model(run.info.run_id, name="test-pm25")
    assert version is not None
    assert version.run_id == run.info.run_id

    # Promote → @production now points at the same version.
    promoted = promote_staging_to_production(name="test-pm25")
    assert promoted == version.version

    # Serving loads the FULL bundle from the production version's run.
    bundle = load_production_bundle(name="test-pm25")
    assert bundle["target"] == "y"
    assert bundle["feature_names"] == ["x"]
    assert float(bundle["model"].predict([[0.0]])[0]) == 5.0
