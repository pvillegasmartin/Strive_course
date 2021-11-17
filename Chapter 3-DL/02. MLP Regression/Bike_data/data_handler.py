import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable

def data_handler():
    df = pd.read_csv('./data/bikes.csv')

    # null_values = df.isnull().sum() #no null values

    x,y = df.drop(['instant', 'dteday', 'registered', 'cnt'], axis=1), df['cnt']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values.reshape(-1, 1), y_test.values.reshape(-1, 1)

    x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor = Variable(torch.tensor(x_train).float()), Variable(torch.tensor(x_test).float()), Variable(torch.tensor(y_train).float()), Variable(torch.tensor(y_test).float())

    return x_train_tensor, x_test_tensor, y_train_tensor, y_test_tensor
