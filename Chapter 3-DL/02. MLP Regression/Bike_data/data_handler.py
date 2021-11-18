import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset


class data_handler(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # null_values = df.isnull().sum() #no null values

        self.x, self.y = df.drop(['instant', 'dteday', 'cnt'], axis=1).values, df['cnt']
        self.features = self.x.shape[1]
        self.n_samples = len(self.x)

    def __getitem__(self, index):
        # we want to be index like dataset[index]
        # to get the index-th batch
        return self.x[index], self.y[index]

    def __len__(self):
        # to retrieve the total samples by doing len(dataset)
        return self.n_samples
