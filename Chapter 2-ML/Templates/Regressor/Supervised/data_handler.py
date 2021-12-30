import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

def get_data(pth):

    data = pd.read_csv(pth)

    x_train, x_test, y_train, y_test = train_test_split(data.values[:,:-1], data.values[:,-1], test_size=0.2, random_state = 0)

    # could use OrdinalEncoding if your pipeline or model is tree based, else OHE for Multiplicative Models
    ct = ColumnTransformer( [('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4,5] ),('non_transformed','passthrough',[0,2,3])] )

    x_train = ct.fit_transform(x_train)
    x_test = ct.transform(x_test)

    #Its not mandatory in tree algorithms but it helps in computational timing
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.fit_transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, ct, scaler