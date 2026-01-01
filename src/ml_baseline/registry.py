from pathlib import Path
import joblib


REGISTRY_DIR = Path("models/registry")
RUNS_DIR = Path("models/runs")


def get_latest_run_id() -> str:
    latest_path = REGISTRY_DIR / "latest.txt"
    if not latest_path.exists():
        raise FileNotFoundError("No latest model found in registry")
    return latest_path.read_text(encoding="utf-8").strip()


def load_latest_model():
    run_id = get_latest_run_id()
    model_path = RUNS_DIR / run_id / "model" / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found for run_id {run_id}")
    return joblib.load(model_path)
