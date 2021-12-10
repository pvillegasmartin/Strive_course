import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as dsets

df = pd.read_csv('data_akbilgic.csv')

df_train = df[:431]
df_test = df[431:]

# OPTION 1: prediction of all y with shift+1 --> y has same shape as n_steps
def next_stock_batch_option1(batch_size, n_steps, df_base):
    t_min = 0
    t_max = df_base.shape[0]

    # The inputs will be formed by 8 sequences taken from
    # 8 time series [ISE.1,SP,DAX,FTSE,NIKKEI,BOVESPA,EU,EM]
    x = np.zeros((batch_size, n_steps, 7))

    # We want to predict the returns of the Istambul stock
    # taken into consideration the previous n_steps days
    y = np.zeros((batch_size, n_steps, 1))

    # We chose batch_size random points from time series x-axis

    starting_points = np.random.randint(0, t_max - n_steps - 1, size=batch_size)
    # print(starting_points)

    # We create the batches for x using all time series (8) between t and t+n_steps
    # We create the batches for y using only one time series between t+1 and t+n_steps+1

    for k in np.arange(batch_size):
        lmat = []
        for j in np.arange(n_steps + 1):
            lmat.append(df_base.iloc[starting_points[k] + j, 2:].values)
            mat = np.array(lmat)
        # The x values include all columns (mat[:n_steps,:]), these are ([ISE.1,SP,DAX,FTSE,NIKKEI,BOVESPA,EU,EM])
        # and TS values in mat between 0 and n_steps
        x[k, :, :] = mat[:n_steps, 1:]

        # The y values include only column 0 (mat[1:n_steps+1,0]), this is ([ISE.1])
        # and TS values in mat between 1 and n_steps+1
        y[k, :, 0] = mat[1:n_steps + 1, 0]

    return x, y

# OPTION 2: prediction of last y --> y has shape 1
def next_stock_batch_option2(batch_size, n_steps, df_base):
    t_min = 0
    t_max = df_base.shape[0]

    # The inputs will be formed by 8 sequences taken from
    # 8 time series [ISE.1,SP,DAX,FTSE,NIKKEI,BOVESPA,EU,EM]
    x = np.zeros((batch_size, n_steps, 7))

    # We want to predict the returns of the Istambul stock
    # taken into consideration the previous n_steps days
    y = np.zeros((batch_size, 1, 1))

    # We chose batch_size random points from time series x-axis

    starting_points = np.random.randint(0, t_max - n_steps - 1, size=batch_size)
    # print(starting_points)

    # We create the batches for x using all time series (8) between t and t+n_steps
    # We create the batches for y using only one time series between t+1 and t+n_steps+1

    for k in np.arange(batch_size):
        lmat = []
        for j in np.arange(n_steps + 1):
            lmat.append(df_base.iloc[starting_points[k] + j, 2:].values)
            mat = np.array(lmat)
        # The x values include all columns (mat[:n_steps,:]), these are ([ISE.1,SP,DAX,FTSE,NIKKEI,BOVESPA,EU,EM])
        # and TS values in mat between 0 and n_steps
        x[k, :, :] = mat[:n_steps, 1:]

        # The y values include only column 0 (mat[1:n_steps+1,0]), this is ([ISE.1])
        # and TS values in mat between 1 and n_steps+1
        y[k, 0, 0] = mat[n_steps, 0]

    return x, y


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your RNN
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, input_dim)
        # batch_dim = number of samples per batch
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()

        # We need to detach the hidden state to prevent exploding/vanishing gradients
        # This is part of truncated backpropagation through time (BPTT)
        out, hn = self.rnn(x, h0.detach())

        # Index hidden state of last time step
        # out.size() --> batch_size, n_steps, output
        # out[:, -1, :] just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> batch_size, output
        return out




batch_size = 100
n_iters = 50
#Number of features
input_dim = 7
hidden_dim = 100
layer_dim = 1
# Number of steps to unroll
seq_dim = 25
#OPTION 1
'''
output_dim = seq_dim
'''
#OPTION 2
output_dim = 1

model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss(reduction='mean')



iter = 0
for it in range(n_iters):

    model.train()

    #OPTION 1
    '''
    x_batch, y_batch = next_stock_batch_option1(batch_size, seq_dim, df_train)
    x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch).squeeze(-1)
    x_batch, y_batch = x_batch.float().requires_grad_(), y_batch.float()
    '''

    # OPTION 2
    x_batch, y_batch = next_stock_batch_option2(batch_size, seq_dim, df_train)
    x_batch, y_batch = torch.from_numpy(x_batch), torch.from_numpy(y_batch)
    x_batch, y_batch = x_batch.float().requires_grad_(), y_batch.float()


    # Clear gradients w.r.t. parameters
    optimizer.zero_grad()

    outputs = model(x_batch)

    loss = criterion(outputs, y_batch)
    print("Iteration ", it, "MSE: ", loss.item())

    # Getting gradients w.r.t. parameters
    loss.backward()

    # Updating parameters
    optimizer.step()

    iter += 1

    if iter % 5 == 0:
        with torch.no_grad():
            model.eval()

            # OPTION 1
            '''
            x_test, y_test = next_stock_batch_option1(batch_size, seq_dim, df_test)
            x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test).squeeze(-1)
            x_test, y_test = x_test.float().requires_grad_(), y_test.float()
            '''

            # OPTION 2
            x_test, y_test = next_stock_batch_option2(batch_size, seq_dim, df_test)
            x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)
            x_test, y_test = x_test.float().requires_grad_(), y_test.float()


            y_test_pred = model(x_test)
            loss_eval = criterion(y_test_pred, y_test)
            print("Iteration_test ", it, "MSE: ", loss_eval.item())