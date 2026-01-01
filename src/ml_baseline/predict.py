from pathlib import Path
import pandas as pd
import joblib

from ml_baseline.schema import default_schema


def run_predict(
    data_path: str = "data/processed/features.csv",
    output_path: str = "data/predictions/predictions.csv",
    run_id: str | None = None,
) -> None:
    registry_path = Path("models/registry/latest.txt")

    if run_id is None:
        run_id = registry_path.read_text(encoding="utf-8").strip()

    model_path = Path("models/runs") / run_id / "model" / "model.joblib"
    model = joblib.load(model_path)

    df = pd.read_csv(Path(data_path))

    schema = default_schema()
    schema.validate(df)

    if "target" in df.columns:
        features = df.drop(columns=["target"])
    else:
        features = df

    preds = model.predict(features)

    result = df.copy()
    result["prediction"] = preds

    if hasattr(model, "predict_proba"):
        result["probability"] = model.predict_proba(features)[:, 1]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)

    print("Run ID:", run_id)
    print("Wrote:", out_path)


