import pandas as pd
from framework_ia.controllers.eda_controller import EDAController


def test_eda_controller_detects_missing_values_and_duplicates():
    data = pd.DataFrame({"age": [10, None, 10], "score": [2, 3, 2]})
    result = EDAController().run(data)
    assert result["analysis"]["shape"] == {"rows": 3, "columns": 2}
    assert result["analysis"]["missing_count"]["age"] == 1
    assert result["analysis"]["duplicate_rows"] == 1
    assert result["interpretation"]["findings"]
