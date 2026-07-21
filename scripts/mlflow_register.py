# /// script
# requires-python = ">=3.10"
# dependencies = ["mlflow-skinny"]
# ///
"""Register the mythic-voice adapter: run + artifacts + model version + aliases.

  GEN=v3 ADAPTER_DIR=out/voice-adapter3 BATTERY_JSON=out/voice-eval-v4.json \
  LEAKAGE_JSON=out/voice-leakage-v4.json PROVENANCE_JSON=out/provenance-final.json \
  uv run scripts/mlflow_register.py
"""
import json
import os

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import RestException

TRACKING = "https://mlflow.knuteson.io"
NAME = "mythic-voice-9b"
GEN = os.environ["GEN"]
ADAPTER_DIR = os.environ["ADAPTER_DIR"]
BATTERY_JSON = os.environ.get("BATTERY_JSON")
LEAKAGE_JSON = os.environ.get("LEAKAGE_JSON")
PROVENANCE_JSON = os.environ.get("PROVENANCE_JSON")


def main() -> None:
    mlflow.set_tracking_uri(TRACKING)
    mlflow.set_experiment("mythic-voice")
    c = MlflowClient(TRACKING)
    try:
        c.create_registered_model(NAME, description=(
            "World-agnostic 'mythic voice' model — synthetic voice data, "
            "two-teacher generation, decontaminated against a private "
            "reference corpus + blocklist. Beta only; see version "
            "descriptions for the ship gate (GuardedTeacher requirement)."))
    except RestException as e:
        if "RESOURCE_ALREADY_EXISTS" not in str(e):
            raise
    with mlflow.start_run(run_name=f"qwen3.5-9b-mythic-voice-{GEN}") as run:
        mlflow.log_artifacts(ADAPTER_DIR, artifact_path="adapter")
        if BATTERY_JSON:
            report = json.load(open(BATTERY_JSON))["report"]
            mlflow.log_metrics({f"battery_{k}": v for k, v in report.items()
                                if isinstance(v, (int, float))})
            mlflow.log_artifact(BATTERY_JSON)
        if LEAKAGE_JSON:
            report = json.load(open(LEAKAGE_JSON))["report"]
            mlflow.log_metrics({f"leakage_unguarded_{k}": v for k, v in report.items()
                                if isinstance(v, (int, float))})
            mlflow.log_artifact(LEAKAGE_JSON)
        if PROVENANCE_JSON:
            prov = json.load(open(PROVENANCE_JSON))
            mlflow.log_metric("leakage_guarded_pass_rate",
                              prov["guarded_leakage_verified"]["pass_rate"])
            mlflow.log_artifact(PROVENANCE_JSON)
    # create_model_version OUTSIDE the run context — mlflow 3.14 register_model
    # needs a LoggedModel, and a failure inside start_run marks the run FAILED.
    mv = c.create_model_version(
        NAME, source=run.info.artifact_uri + "/adapter", run_id=run.info.run_id)
    c.set_model_version_tag(NAME, mv.version, "local_name", GEN)
    c.update_model_version(NAME, mv.version, description=(
        f"World-agnostic mythic-voice model, generation {GEN}. Base: Qwen/Qwen3.5-9B. "
        f"Trained via staged CPT->SFT->DPO on synthetic two-teacher voice data, "
        f"decontaminated against a private reference corpus + blocklist. "
        f"MUST be served through personaforge.guard.GuardedTeacher for the leakage "
        f"guarantee to hold (raw weights reach ~0.89 on the leakage eval, not 1.0). "
        f"Beta only — no public HF release without a separate explicit go-ahead. "
        f"Local: voice-adapter3, voice-merged3, voice-v1-q8.gguf, voice-mlx-q4."))
    c.set_registered_model_alias(NAME, f"gen-{GEN}", mv.version)
    c.set_registered_model_alias(NAME, "current", mv.version)
    print(f"registered version {mv.version} = @gen-{GEN} = @current, run {run.info.run_id}")


if __name__ == "__main__":
    main()
