import time

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = pd.read_csv('climate.csv')
data = data.drop(['Date Time',"Tpot (K)"], axis=1)


"""
#Data exploration
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(data.corr(), vmin=-1, vmax=1, annot=True)
plt.show()
"""

def pairing(data, period=6):
    x=[]
    y=[]
    for i in range(0,data.shape[0]-period+1,period+1):
        x_period = data.iloc[i:i+period,:]
        x.append(np.array(x_period).flatten())
        y.append(data['T (degC)'][i+period])
    return np.array(x),np.array(y)

x,y = pairing(data)


"""
classifiers = {
  "SVR": SVR(),
  "Decision Tree": DecisionTreeRegressor(),
  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
  "Random Forest": RandomForestRegressor(n_estimators=100),
  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
  "XGBoost":       XGBRegressor(n_estimators=100),
  "LightGBM":      LGBMRegressor(n_estimators=100),
  "CatBoost":      CatBoostRegressor(n_estimators=100)
}
"""
classifiers = {
  "CatBoost":      CatBoostRegressor(n_estimators=100)
}

tscv = TimeSeriesSplit(n_splits=5)
i = 1
results = pd.DataFrame({'Model': [], 'Iteration':[], 'Explained_variance': [], 'MSE': [], 'MAB': [], "R2-score": [], 'Time': []})

for tr_index, val_index in tscv.split(x):
    scaler = StandardScaler()

    X_tr, X_val = x[tr_index], x[val_index]
    y_tr, y_val = y[tr_index], y[val_index]

    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    for model_name, model in classifiers.items():
        start_time = time.time()
        model.fit(X_tr, y_tr)
        total_time = time.time() - start_time

        y_pred = model.predict(X_val)

        results = results.append({"Model": model_name,
                                  "Iteration": i,
                                  "Explained_variance": metrics.explained_variance_score(y_val, y_pred),
                                  "MSE": metrics.mean_squared_error(y_val, y_pred),
                                  "MAB": metrics.mean_absolute_error(y_val, y_pred),
                                  "R2-score": metrics.r2_score(y_val, y_pred),
                                  "Time": total_time},
                                 ignore_index=True)

    i += 1

plt.figure()
plt.plot(np.linspace(1, y_val.shape[0],y_val.shape[0]), y_val, label='real', linewidth=1 )
plt.plot(np.linspace(1, y_val.shape[0],y_val.shape[0]), y_pred, linestyle='dashed', label='prediction',linewidth=0.5 )
plt.legend()
plt.show()

print(results)