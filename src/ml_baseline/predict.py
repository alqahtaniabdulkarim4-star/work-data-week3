from pathlib import Path
import typer
import pandas as pd

from ml_baseline.registry import load_latest_model


def run_predict(
    data_path: str = "data/processed/features.csv",
    output_path: str = "data/predictions/predictions.csv",
) -> None:
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    model = load_latest_model()

    X = df.copy()
    if "target" in X.columns:
        X = X.drop(columns=["target"])

    preds = model.predict(X)

    out_df = df.copy()
    out_df["prediction"] = preds

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    run_id = (Path("models/registry/latest.txt")).read_text(encoding="utf-8").strip()
    print("Run ID:", run_id)
    print("Wrote:", out_path)



