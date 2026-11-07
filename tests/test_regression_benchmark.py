import pandas as pd
from framework_ia.models.supervised.supervised_models import RegressionModels

def test_required_regression_estimators_are_available():
    factories = [RegressionModels.linear_regression, RegressionModels.lasso,
                 RegressionModels.lasso_cv, RegressionModels.ridge, RegressionModels.ridge_cv,
                 RegressionModels.svr, RegressionModels.decision_tree, RegressionModels.random_forest]
    assert len([factory() for factory in factories]) == 8
