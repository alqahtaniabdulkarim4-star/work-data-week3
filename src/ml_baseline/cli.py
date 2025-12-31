from pathlib import Path
import typer
import pandas as pd
import numpy as np

app = typer.Typer(help="Week 3 ML baseline system CLI")


@app.callback()
def main():
    """Root command for ml-baseline."""
    pass


@app.command()
def helpme():
    """Sanity command to confirm the CLI runs."""
    typer.echo("ml-baseline CLI is working ✅")


@app.command("make-sample-data")
def make_sample_data(
    out: str = typer.Option("data/processed/features.csv", help="Output CSV path"),
    rows: int = typer.Option(200, help="Number of rows"),
    seed: int = typer.Option(7, help="Random seed"),
):
    """
    Create a tiny sample dataset and write it to data/processed/features.csv.
    """
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

