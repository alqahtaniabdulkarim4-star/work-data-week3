from pathlib import Path
import joblib
import pandas as pd

def run_predict(data_path: str, output_path: str) -> None:
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    registry_path = Path("models/registry/latest.txt")
    run_id = registry_path.read_text(encoding="utf-8").strip()

    model_path = Path("models/runs") / run_id / "model" / "model.joblib"
    model = joblib.load(model_path)

    X = df.copy()
    if "target" in X.columns:
        X = X.drop(columns=["target"])

    preds = model.predict(X)

    result = df.copy()
    result["prediction"] = preds

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        result["probability"] = proba

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print("Run ID:", run_id)
    print("Wrote:", out_path)


