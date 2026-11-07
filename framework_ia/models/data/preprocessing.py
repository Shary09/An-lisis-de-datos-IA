"""Configurable, reproducible dataframe preparation utilities."""
from dataclasses import dataclass, asdict
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@dataclass
class PreparationReport:
    rows_before: int
    columns_before: int
    rows_after: int
    columns_after: int
    missing_before: int
    missing_after: int
    duplicates_before: int
    duplicates_after: int
    dropped_columns: list
    duplicates_removed: int
    imputation: str
    scaling: str

    def to_dict(self):
        return asdict(self)

def prepare_dataframe(data: pd.DataFrame, *, drop_columns=None, remove_duplicates=False,
                      imputation='none', scaling='none'):
    """Return (prepared_copy, report); never mutates the source dataframe.

    Imputation choices: none, mean, median, most_frequent. Scaling: none,
    standardize, normalize. Categorical values are only imputed when requested.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame')
    out = data.copy()
    before = out.copy()
    dropped = [c for c in (drop_columns or []) if c in out.columns]
    out = out.drop(columns=dropped)
    duplicates_before = int(out.duplicated().sum())
    if remove_duplicates:
        out = out.drop_duplicates().copy()
    numeric = out.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in out.columns if c not in numeric]
    if imputation not in {'none', 'mean', 'median', 'most_frequent'}:
        raise ValueError('Unsupported imputation strategy.')
    if imputation != 'none':
        for col in numeric:
            if imputation == 'mean': value = out[col].mean()
            elif imputation == 'median': value = out[col].median()
            else:
                mode = out[col].mode(dropna=True); value = mode.iloc[0] if not mode.empty else None
            if value is not None: out[col] = out[col].fillna(value)
        for col in categorical:
            if out[col].isna().any():
                mode = out[col].mode(dropna=True)
                if not mode.empty: out[col] = out[col].fillna(mode.iloc[0])
    if scaling not in {'none', 'standardize', 'normalize'}:
        raise ValueError('Unsupported scaling strategy.')
    if scaling != 'none' and numeric:
        # Keep missing values as missing if imputation was explicitly disabled.
        cols = [c for c in numeric if out[c].notna().all()]
        if cols:
            scaler = StandardScaler() if scaling == 'standardize' else MinMaxScaler()
            out.loc[:, cols] = scaler.fit_transform(out[cols])
    report = PreparationReport(
        rows_before=int(before.shape[0]), columns_before=int(before.shape[1]),
        rows_after=int(out.shape[0]), columns_after=int(out.shape[1]),
        missing_before=int(before.isna().sum().sum()), missing_after=int(out.isna().sum().sum()),
        duplicates_before=duplicates_before, duplicates_after=int(out.duplicated().sum()),
        dropped_columns=[str(c) for c in dropped],
        duplicates_removed=int(before.shape[0] - len(out)) if remove_duplicates else 0,
        imputation=imputation, scaling=scaling)
    return out, report
