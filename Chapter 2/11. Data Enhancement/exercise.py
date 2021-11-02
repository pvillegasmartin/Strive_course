import time

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from imblearn import pipeline
from lightgbm import LGBMRegressor
from sklearn import preprocessing, compose, metrics
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

np.random.seed(0)

df = pd.read_csv('./data/london_merged.csv')
print(df.isnull().sum()) # no null values

#Adding features
df['hour'] = df['timestamp'].apply(lambda row: row.split(':')[0][-2:])

#Split the data in TRAIN/TEST
x, y = df.drop(['cnt','timestamp'], axis=1), df['cnt']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

x_train['cnt'] = y_train

#Increasing train data
def enhancement(train, dev):
    gen_data = train.copy()
    for season in train['season'].unique():
        seasonal_data = gen_data[gen_data['season'] == season]
        hum_std = seasonal_data['hum'].std()
        wind_speed_std = seasonal_data['wind_speed'].std()
        t1_std = seasonal_data['t1'].std()
        t2_std = seasonal_data['t2'].std()

        for row in train[train['season']==season].index:

            if np.random.randint(2)==1:
                gen_data.loc[row, 'hum'] += hum_std/dev
            else:
                gen_data.loc[row, 'hum'] -= hum_std/dev

            if np.random.randint(2)==1:
                gen_data.loc[row, 'wind_speed'] += wind_speed_std/dev
            else:
                gen_data.loc[row, 'wind_speed'] -= wind_speed_std/dev

            if np.random.randint(2)==1:
                gen_data.loc[row, 't1'] += t1_std/dev
            else:
                gen_data.loc[row, 't1'] -= t1_std/dev

            if np.random.randint(2)==1:
                gen_data.loc[row, 't2'] += t2_std/dev
            else:
                gen_data.loc[row, 't2'] -= t2_std/dev

    return gen_data

x_new = enhancement(x_train, 8).sample(x_train.shape[0]//4)
x_train, y_train = pd.concat([x_train, x_new ]).drop('cnt', axis=1), pd.concat([x_train, x_new ])['cnt']



cat_vars = ['season','is_weekend','is_holiday','hour','weather_code']
num_vars = ['t1','t2','hum','wind_speed']

tree_classifiers = {
  "Decision Tree": DecisionTreeRegressor(),
  "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
  "Random Forest": RandomForestRegressor(n_estimators=100),
  "AdaBoost":      AdaBoostRegressor(n_estimators=100),
  "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
  "XGBoost":       XGBRegressor(n_estimators=100),
  "LightGBM":      LGBMRegressor(n_estimators=100),
  "CatBoost":      CatBoostRegressor(n_estimators=100)
}

num_features = pipeline.Pipeline(steps=[
    ('standard', StandardScaler()),
])

cat_features = pipeline.Pipeline(steps=[
    ('ordinal', preprocessing.OrdinalEncoder())
])

tree_prepro = compose.ColumnTransformer(transformers=[
    ('num', num_features, num_vars),
    ('cat', cat_features, cat_vars),
], remainder='drop')

tree_classifiers = {name: pipeline.make_pipeline(tree_prepro, model) for name, model in tree_classifiers.items()}

results = pd.DataFrame({'Model': [],  'Explained_variance': [],'MSE': [], 'MAB': [], "R2-score": [], 'Time': []})

for model_name, model in tree_classifiers.items():
    start_time = time.time()
    model.fit(x_train, y_train)
    total_time = time.time() - start_time

    y_pred = model.predict(x_test)

    results = results.append({"Model": model_name,
                              "Explained_variance": metrics.explained_variance_score(y_test, y_pred),
                              "MSE": metrics.mean_squared_error(y_test, y_pred),
                              "MAB": metrics.mean_absolute_error(y_test, y_pred),
                              "R2-score": metrics.r2_score(y_test,y_pred),
                              "Time": total_time},
                             ignore_index=True)

#Catboost is the best model, higher R2-score and more variance explained