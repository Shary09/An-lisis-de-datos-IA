"""Dataset loading utilities."""
from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and return a DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)
