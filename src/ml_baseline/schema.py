from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import pandas as pd


@dataclass(frozen=True)
class DataSchema:
    required_columns: List[str]
    dtypes: Dict[str, str]

    def validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns {missing}")

    def validate_dtypes(self, df: pd.DataFrame) -> None:
        kind_map = {
            "float": "f",
            "int": "i",
            "bool": "b",
            "str": "O",
            "object": "O",
        }

        for col, expected in self.dtypes.items():
            if col in df.columns:
                expected_kind = kind_map.get(expected)
                actual_kind = df[col].dtype.kind
                if expected_kind and actual_kind != expected_kind:
                    raise ValueError(
                        f"Column {col} has dtype {df[col].dtype} expected {expected}"
                    )

    def validate(self, df: pd.DataFrame) -> None:
        self.validate_columns(df)
        self.validate_dtypes(df)


def default_schema() -> DataSchema:
    required = ["id", "x1", "x2", "target"]
    dtypes = {
        "id": "int",
        "x1": "float",
        "x2": "float",
        "target": "int",
    }
    return DataSchema(required_columns=required, dtypes=dtypes)


