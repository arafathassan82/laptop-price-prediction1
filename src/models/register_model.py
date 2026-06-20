import os, sys
import json
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.logger import get_logger
from src.utils.config import get_mlflow_cfg

logger = get_logger("model_registration")


# ── load env ─────────────────────────────
load_dotenv()

token = os.getenv("DAGSHUB_PAT")
if not token:
    raise EnvironmentError("DAGSHUB_PAT not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = token
os.environ["MLFLOW_TRACKING_PASSWORD"] = token

# ── same tracking URI as training ────────
mlflow.set_tracking_uri(
    "https://dagshub.com/shahriar0999/mlops-small-project.mlflow"
)

client = MlflowClient()


# ── get latest run automatically ─────────
def get_latest_run_id(experiment_name: str):
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError("Experiment not found")
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError("No runs found")
    return runs[0].info.run_id



# ── register model ───────────────────────
def register_model(model_name: str, run_id: str):

    model_uri = f"runs:/{run_id}/model"

    logger.info(f"Registering model from {model_uri}")

    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    logger.info(f"Model registered: {model_name} v{version}")

    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Staging",
        archive_existing_versions=True
    )

    logger.info("Model moved to Staging ✔")


# ── MAIN ────────────────────────────────
def main():

    mlf_cfg = get_mlflow_cfg()
    experiment_name = mlf_cfg["experiment_train"]

    run_id = get_latest_run_id(experiment_name)

    register_model(
        model_name="laptop_price_model",
        run_id=run_id
    )


if __name__ == "__main__":
    main()