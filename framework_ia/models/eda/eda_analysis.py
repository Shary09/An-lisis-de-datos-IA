"""Reusable exploratory data analysis functions."""
import pandas as pd


def analyze_dataframe(data: pd.DataFrame) -> dict:
    """Return a compact, serializable EDA summary based on actual data."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    numeric = data.select_dtypes(include="number")
    return {
        "shape": {"rows": int(data.shape[0]), "columns": int(data.shape[1])},
        "column_names": [str(c) for c in data.columns],
        "dtypes": {str(c): str(t) for c, t in data.dtypes.items()},
        "missing_count": {str(c): int(v) for c, v in data.isna().sum().items()},
        "missing_percent": {str(c): round(float(v), 2) for c, v in (data.isna().mean() * 100).items()},
        "duplicate_rows": int(data.duplicated().sum()),
        "numeric_describe": numeric.describe().replace({float("inf"): None, float("-inf"): None}).to_dict(),
        "numeric_correlations": numeric.corr(numeric_only=True).to_dict() if numeric.shape[1] > 1 else {},
    }
