from pathlib import Path
import json
import time

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from ml_baseline.schema import default_schema
from ml_baseline.metrics import compute_classification_metrics


def run_train(
    target: str,
    data_path: str = "data/processed/features.csv",
) -> None:
    data_path = Path(data_path)
    df = pd.read_csv(data_path)

    schema = default_schema()
    schema.validate(df)

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]

    metrics = compute_classification_metrics(y_test, preds, proba)

    run_id = str(int(time.time()))
    run_dir = Path("models/runs") / run_id

    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (run_dir / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "schema").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model" / "model.joblib")

    (run_dir / "metrics" / "holdout_metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    X_test.to_csv(
        run_dir / "tables" / "holdout_input.csv",
        index=False,
    )

    schema_payload = {
        "required_columns": schema.required_columns,
        "dtypes": schema.dtypes,
    }
    (run_dir / "schema" / "input_schema.json").write_text(
        json.dumps(schema_payload, indent=2),
        encoding="utf-8",
    )

    meta = {
        "run_id": run_id,
        "target": target,
        "data_path": str(data_path),
        "created_at": int(time.time()),
    }
    (run_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )

    pred_df = X_test.copy()
    pred_df[target] = y_test.values
    pred_df["prediction"] = preds
    pred_df.to_csv(
        run_dir / "predictions" / "holdout_predictions.csv",
        index=False,
    )

    Path("models/registry").mkdir(parents=True, exist_ok=True)
    Path("models/registry/latest.txt").write_text(
        run_id,
        encoding="utf-8",
    )

    print("Run ID:", run_id)
    print("Metrics:", metrics)
    print("Wrote:", run_dir)

