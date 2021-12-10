import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets

df = pd.read_csv('data_akbilgic.csv')

def split_data(stock, lookback):
    data_raw = stock.to_numpy()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])

    data = np.array(data);
    test_set_size = int(np.round(0.15 * data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size, :-1, 1:-1]
    y_train = data[:train_set_size, -1, -1]

    x_test = data[train_set_size:, :-1,1:-1]
    y_test = data[train_set_size:, -1, -1]

    return [x_train, y_train, x_test, y_test]



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # 28 time steps
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out

# Number of steps to unroll
seq_dim = 10

#DATA
x_train, y_train, x_test, y_test = split_data(df, seq_dim)
x_train, y_train, x_test, y_test = x_train.astype('float64'), y_train.astype('float64'), x_test.astype('float64'), y_test.astype('float64')

x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
x_train, y_train = x_train.float().requires_grad_(), y_train.reshape(-1,1).float()

x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
x_test, y_test = x_test.float().requires_grad_(), y_test.reshape(-1,1).float()

#MODEL
n_iters = 10
#Number of features
input_dim = x_train.shape[-1]
hidden_dim = 100
layer_dim = 1
#Number of outputs
output_dim = y_train.shape[-1]

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss(reduction='mean')

iter = 0
for it in range(n_iters):

    model.train()

    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    outputs = model(x_train)

    loss = criterion(outputs, y_train)
    print("Iteration ", it, "MSE: ", loss.item())

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    iter += 1

    if iter % 5 == 0:
        with torch.no_grad():
            model.eval()

            y_test_pred = model(x_test)
            loss_eval = criterion(y_test_pred, y_test)
            print("Iteration_test ", it, "MSE: ", loss_eval.item())