import pandas as pd
from framework_ia.models.data.preprocessing import prepare_dataframe

def test_preparation_reports_changes_without_mutating_source():
    source = pd.DataFrame({'value':[1.0, None, 1.0], 'group':['a','b','a'], 'dropme':[1,2,1]})
    prepared, report = prepare_dataframe(source, drop_columns=['dropme'], remove_duplicates=True, imputation='mean', scaling='standardize')
    assert source.shape == (3, 3)
    assert prepared.shape == (2, 2)
    assert prepared.isna().sum().sum() == 0
    assert report['rows_before'] == 3 if isinstance(report, dict) else report.rows_before == 3
