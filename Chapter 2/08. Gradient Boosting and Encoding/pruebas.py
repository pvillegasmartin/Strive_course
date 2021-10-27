import data_handler as dh
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import explained_variance_score, r2_score
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
'''
data = pd.read_csv("./insurance.csv")
ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4] ),('non_transformed','passthrough',[3])] )
x_train = pd.DataFrame(ct.fit_transform(data))


'''
x_train, x_test, y_train, y_test, ct, scaler = dh.get_data("./insurance.csv")

clf_1 = GradientBoostingRegressor(learning_rate=0.01, n_estimators=1000)
clf_2 = XGBRegressor()
clf_3 = CatBoostRegressor(max_depth=4)

clf_1.fit(x_train,y_train)
clf_2.fit(x_train,y_train)
clf_3.fit(x_train,y_train)


y_pred_1 = clf_1.predict(x_test)
y_pred_2 = clf_2.predict(x_test)
y_pred_3 = clf_3.predict(x_test)

print(explained_variance_score(y_test, y_pred_1), r2_score(y_test, y_pred_1))
print(explained_variance_score(y_test, y_pred_2), r2_score(y_test, y_pred_2))
print(explained_variance_score(y_test, y_pred_3), r2_score(y_test, y_pred_3))

#print(x_train.corr())