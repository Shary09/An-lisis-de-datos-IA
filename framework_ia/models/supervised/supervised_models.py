"""Central registry for supervised algorithm families.

Keep classification and regression methods independent while satisfying the
project convention that supervised algorithms live in one Python module.
"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostClassifier
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, RidgeCV
from sklearn.svm import SVR


class ClassificationModels:
    """Factory methods for classification estimators."""
    @staticmethod
    def knn(**kwargs): return KNeighborsClassifier(**kwargs)
    @staticmethod
    def decision_tree(**kwargs): return DecisionTreeClassifier(random_state=42, **kwargs)
    @staticmethod
    def random_forest(**kwargs): return RandomForestClassifier(random_state=42, **kwargs)
    @staticmethod
    def adaboost(**kwargs): return AdaBoostClassifier(random_state=42, **kwargs)


class RegressionModels:
    """Factory methods for the eight regression algorithms required in Lab 3."""
    @staticmethod
    def linear_regression(**kwargs): return LinearRegression(**kwargs)
    @staticmethod
    def lasso(**kwargs): return Lasso(**kwargs)
    @staticmethod
    def lasso_cv(**kwargs): return LassoCV(**kwargs)
    @staticmethod
    def ridge(**kwargs): return Ridge(**kwargs)
    @staticmethod
    def ridge_cv(**kwargs): return RidgeCV(**kwargs)
    @staticmethod
    def svr(**kwargs): return SVR(**kwargs)
    @staticmethod
    def decision_tree(**kwargs): return DecisionTreeRegressor(random_state=42, **kwargs)
    @staticmethod
    def random_forest(**kwargs): return RandomForestRegressor(random_state=42, **kwargs)
