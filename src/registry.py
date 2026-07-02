"""
MLflow Model Registry integration (Phase 7.3).

Gives every trained model a registered version with lifecycle aliases instead
of provenance-by-git-commit:

- ``python -m src.train`` registers each new model under
  :data:`config.MLFLOW_REGISTERED_MODEL_NAME` and points the **staging**
  alias at it (see :func:`register_run_model`, called from ``src/train.py``).
- After the champion/challenger gate passes, ``python -m src.registry promote``
  moves the **production** alias to the staging version.
- Serving (``src/inference.py``) loads the current production bundle when the
  environment selects the registry (``MODEL_SOURCE=registry``), and otherwise
  keeps using the committed ``models/model.joblib`` — so the repo still works
  out of the box with zero infrastructure.

Aliases, not stages: MLflow 3.x removed the classic ``Staging``/``Production``
*stages*; registered-model **aliases** are their successor and behave the same
way (one mutable pointer per environment).

The registry needs a database-backed or remote tracking server — the plain
``mlruns/`` file store has no registry tables.  Point ``MLFLOW_TRACKING_URI``
at e.g. ``sqlite:///mlflow.db`` (see ``scripts/start_mlflow_server.*``) or an
HTTP tracking server.  On the file store every function here degrades to a
clear log message instead of crashing the training run.

What gets registered vs what gets served: training logs the point model via
``mlflow.sklearn.log_model`` (that is the registered artifact, giving the
registry a signature and input example) **and** uploads the full serving
bundle — point + quantile models + conformal offsets — as the
``model_bundle/model.joblib`` artifact *on the same run*.  Loading a version
therefore resolves the alias → run, then downloads that run's bundle, so the
serving path keeps its prediction intervals.

CLI::

    python -m src.registry status    # list versions + aliases
    python -m src.registry promote   # production alias ← staging version
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME: str = config.MLFLOW_REGISTERED_MODEL_NAME
STAGING_ALIAS: str = "staging"
PRODUCTION_ALIAS: str = "production"

# Artifact path (within a run) where train.py uploads the full serving bundle.
BUNDLE_ARTIFACT_PATH: str = "model_bundle/model.joblib"


def _resolve_tracking_uri() -> str:
    """The tracking URI serving/registry code should talk to.

    Environment wins (that's how CI and deployments point at a real server);
    otherwise fall back to the local default in :mod:`config`.
    """
    return os.environ.get("MLFLOW_TRACKING_URI") or config.MLFLOW_TRACKING_URI


def registry_capable(tracking_uri: str | None = None) -> bool:
    """Whether the tracking backend can host a Model Registry.

    The plain file store (``file://`` / bare path) cannot; database URIs
    (``sqlite://``, ``postgresql://``, …) and remote servers can.
    """
    uri = tracking_uri or _resolve_tracking_uri()
    if uri.startswith(("http://", "https://", "databricks")):
        return True
    # Database-backed stores expose a scheme like sqlite:/// or postgresql://.
    scheme = uri.split("://", 1)[0].lower() if "://" in uri else ""
    return scheme in {"sqlite", "postgresql", "mysql", "mssql"}


def register_run_model(
    run_id: str,
    *,
    name: str = REGISTERED_MODEL_NAME,
    alias: str = STAGING_ALIAS,
) -> Any | None:
    """Register the model logged by ``run_id`` and point ``alias`` at it.

    Called from ``src/train.py`` after a successful training run.  Returns the
    new ``ModelVersion``, or ``None`` when the tracking backend has no
    registry (file store) — training must not fail because of that.
    """
    if not registry_capable():
        logger.warning(
            "Model Registry unavailable: tracking URI %r is a plain file store. "
            "Point MLFLOW_TRACKING_URI at sqlite:///mlflow.db or a tracking "
            "server to enable registration (see README).",
            _resolve_tracking_uri(),
        )
        return None
    try:
        version = mlflow.register_model(f"runs:/{run_id}/model", name)
        client = MlflowClient()
        client.set_registered_model_alias(name, alias, version.version)
        logger.info(
            "Registered model '%s' version %s (run %s) and set alias @%s.",
            name, version.version, run_id, alias,
        )
        return version
    except MlflowException as exc:
        logger.warning("Could not register model in the MLflow registry (%s); continuing.", exc)
        return None


def promote_staging_to_production(*, name: str = REGISTERED_MODEL_NAME) -> str:
    """Point the production alias at the current staging version.

    Returns the promoted version number.  Raises :class:`MlflowException`
    when there is no staging version or no registry — promotion is an explicit
    lifecycle action, so unlike registration it fails loudly.
    """
    client = MlflowClient()
    staging = client.get_model_version_by_alias(name, STAGING_ALIAS)
    client.set_registered_model_alias(name, PRODUCTION_ALIAS, staging.version)
    logger.info(
        "Promoted '%s' version %s: alias @%s → @%s.",
        name, staging.version, STAGING_ALIAS, PRODUCTION_ALIAS,
    )
    return staging.version


def load_production_bundle(*, name: str = REGISTERED_MODEL_NAME) -> dict[str, Any]:
    """Load the serving bundle for the current production model version.

    Resolves the ``@production`` alias to a model version, then downloads the
    ``model_bundle/model.joblib`` artifact from that version's training run —
    the full dict (point model + quantile models + offsets) that
    ``src/inference.py`` expects.
    """
    uri = _resolve_tracking_uri()
    if not registry_capable(uri):
        raise MlflowException(
            f"MODEL_SOURCE=registry but tracking URI {uri!r} has no Model "
            "Registry. Set MLFLOW_TRACKING_URI to a database-backed or remote "
            "tracking server, or unset MODEL_SOURCE to serve models/model.joblib."
        )
    mlflow.set_tracking_uri(uri)
    client = MlflowClient(tracking_uri=uri)
    version = client.get_model_version_by_alias(name, PRODUCTION_ALIAS)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path=BUNDLE_ARTIFACT_PATH, tracking_uri=uri
    )
    bundle = joblib.load(local_path)
    logger.info(
        "Loaded production bundle: model '%s' version %s (run %s).",
        name, version.version, version.run_id,
    )
    return bundle


def _cmd_status(name: str) -> int:
    client = MlflowClient()
    try:
        model = client.get_registered_model(name)
    except MlflowException as exc:
        print(f"No registered model '{name}': {exc.message}")
        return 1
    aliases = dict(model.aliases or {})
    print(f"Registered model: {name}")
    for mv in client.search_model_versions(f"name = '{name}'"):
        alias_str = ", ".join(f"@{a}" for a, v in aliases.items() if v == mv.version)
        print(f"  version {mv.version}  run={mv.run_id}  {alias_str}")
    if not aliases:
        print("  (no aliases set)")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "promote"])
    parser.add_argument("--name", default=REGISTERED_MODEL_NAME)
    args = parser.parse_args(argv)

    mlflow.set_tracking_uri(_resolve_tracking_uri())

    if args.command == "status":
        return _cmd_status(args.name)

    # promote
    if not registry_capable():
        logger.error(
            "Cannot promote: tracking URI %r has no Model Registry backend.",
            _resolve_tracking_uri(),
        )
        return 1
    try:
        version = promote_staging_to_production(name=args.name)
    except MlflowException as exc:
        logger.error("Promotion failed: %s", exc)
        return 1
    print(f"Promoted '{args.name}' version {version} to @{PRODUCTION_ALIAS}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
