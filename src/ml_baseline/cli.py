from pathlib import Path
import typer
import pandas as pd
import numpy as np

app = typer.Typer(help="Week 3 ML baseline system CLI")

@app.callback()
def main():
    pass

@app.command()
def helpme():
    typer.echo("ml-baseline CLI is working")

@app.command("make-sample-data")
def make_sample_data(
    out: str = typer.Option("data/processed/features.csv"),
    rows: int = typer.Option(200),
    seed: int = typer.Option(7),
):
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "x1": rng.normal(0, 1, size=rows),
            "x2": rng.normal(0, 1, size=rows),
            "target": rng.integers(0, 2, size=rows),
        }
    )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    typer.echo(f"Wrote {len(df)} rows to {out_path}")

from ml_baseline.train import run_train

@app.command()
def train(
    target: str = typer.Option(..., "--target"),
):
    run_train(target=target)




