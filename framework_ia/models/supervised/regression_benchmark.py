"""Leakage-aware regression benchmarking for the Lab 3 required estimators."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from framework_ia.models.supervised.supervised_models import RegressionModels

REGRESSION_FACTORIES = {
    'LinearRegression': RegressionModels.linear_regression,
    'Lasso': RegressionModels.lasso,
    'LassoCV': RegressionModels.lasso_cv,
    'Ridge': RegressionModels.ridge,
    'RidgeCV': RegressionModels.ridge_cv,
    'SVR': RegressionModels.svr,
    'DecisionTreeRegressor': RegressionModels.decision_tree,
    'RandomForestRegressor': RegressionModels.random_forest,
}

def benchmark_regression(data: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    if target not in data.columns: raise ValueError(f'Target column not found: {target}')
    frame = data.dropna(subset=[target]).copy()
    y = frame[target]
    if not pd.api.types.is_numeric_dtype(y):
        raise ValueError('Regression target must be numeric. Choose classification for categorical targets.')
    X = frame.drop(columns=[target])
    if X.empty: raise ValueError('At least one predictor is required.')
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers=[]
    if numeric:
        transformers.append(('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric))
    if categorical:
        transformers.append(('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical))
    preprocess = ColumnTransformer(transformers)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    rows=[]; fitted={}
    for name, factory in REGRESSION_FACTORIES.items():
        estimator=factory()
        pipe=Pipeline([('preprocess', clone(preprocess)), ('model', estimator)])
        try:
            pipe.fit(X_train, y_train)
            pred=pipe.predict(X_test)
            rows.append({'model':name,'MAE':float(mean_absolute_error(y_test,pred)), 'MSE':float(mean_squared_error(y_test,pred)), 'RMSE':float(mean_squared_error(y_test,pred)**0.5), 'R2':float(r2_score(y_test,pred))})
            fitted[name]=pipe
        except Exception as exc:
            rows.append({'model':name,'MAE':None,'MSE':None,'RMSE':None,'R2':None,'error':str(exc)})
    results=pd.DataFrame(rows)
    if 'RMSE' in results: results=results.sort_values('RMSE',na_position='last').reset_index(drop=True)
    return {'results':results,'models':fitted,'target':target,'train_rows':len(X_train),'test_rows':len(X_test),'random_state':random_state}
