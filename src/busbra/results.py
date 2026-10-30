"""Read access to the stored result tables (reports/tables) for the analysis and plot scripts, and the comparison of recomputed tables with stored ones; no computation of results."""
from __future__ import annotations

import numpy as np
import pandas as pd

from busbra import config


def table(name: str, **read_csv_options) -> pd.DataFrame:
    """reports/tables/<name>.csv with exact float round trip; IDs and cases stay strings."""
    path = config.TABLES_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"result table not found: {path}")
    return pd.read_csv(path, dtype={"ID": str, "Case": str, **read_csv_options.pop("dtype", {})}, float_precision="round_trip", **read_csv_options)


def same_as_stored(new: pd.DataFrame, path, ignore=()) -> bool:
    """Equality with a stored table: text and integers exactly, floats to 1e-12, undefined equal to undefined. `ignore`: descriptive text columns that may be worded differently."""
    stored = pd.read_csv(path, dtype=str, keep_default_na=False).drop(columns=list(ignore), errors="ignore")
    mine = pd.read_csv(pd.io.common.StringIO(new.to_csv(index=False, lineterminator="\n", na_rep="")), dtype=str, keep_default_na=False).drop(columns=list(ignore), errors="ignore")
    if list(stored.columns) != list(mine.columns) or len(stored) != len(mine):
        return False
    for column in stored.columns:
        a, b = stored[column], mine[column]
        if a.equals(b):
            continue
        try:
            if not np.allclose(pd.to_numeric(a.replace("", "nan")), pd.to_numeric(b.replace("", "nan")), atol=1e-12, rtol=0, equal_nan=True):
                return False
        except (ValueError, TypeError):
            return False
    return True
