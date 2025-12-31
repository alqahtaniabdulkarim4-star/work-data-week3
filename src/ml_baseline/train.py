from pathlib import Path
import json
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_train(target: str):
    data_path = Path("data/processed/features.csv")
    df = pd.read_csv(data_path)

    if target not in df.columns:
        raise ValueError(f"Target column not found: {target}")

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    run_id = str(int(time.time()))
    run_dir = Path("models/runs") / run_id
    (run_dir / "model").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, run_dir / "model" / "model.joblib")

    metrics = {"accuracy": float(acc)}
    (run_dir / "metrics" / "holdout_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    Path("models/registry").mkdir(parents=True, exist_ok=True)
    Path("models/registry/latest.txt").write_text(run_id, encoding="utf-8")

    print("Run ID:", run_id)
    print("Accuracy:", acc)
