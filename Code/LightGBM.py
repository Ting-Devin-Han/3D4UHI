from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.model_selection import GridSearchCV


df = pd.read_excel(r'E:\city\center35.xlsx')

X = df.iloc[:, 2:11]#SIF: 5 12
y = df.iloc[:, 1]
name=list(df)[2:11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lgb_model = LGBMRegressor()

param_grid = {

    'num_leaves': [3,5,7],
    'learning_rate': [0.05,0.12],
    'n_estimators': [100,200,300],
    'max_depth': [1,3,5],
    'lambda_l2': [3],
    'subsample': [0.2,0.4],
    'colsample_bytree': [0.3,0.6],
    'min_child_samples':[3,4,6],
    'random_state':[4,8],
    'n_jobs': [2,4],
    'verbosity': [-1]
}

grid_search = GridSearchCV(
    estimator=lgb_model,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    verbose=2,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)

model = grid_search.best_estimator_
y_pred = model.predict(X_test)

r2 = metrics.r2_score(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Optimized r2:', r2)
print('Optimized rmse:', rmse)


explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap_interaction_values = explainer.shap_interaction_values(X)
