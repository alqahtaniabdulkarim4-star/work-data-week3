work-data-week3

Setup

Install dependencies and run commands using uv.

Commands

1) Make sample data

Command:
uv run ml-baseline make-sample-data

Output:
 • data/processed/features.csv

2) Train baseline model

Command:
uv run ml-baseline train –target target

Output:
 • models/runs/<run_id>/
 • models/registry/latest.txt

3) Run predictions

Command:
uv run ml-baseline predict

Output:
 • data/predictions/predictions.csv