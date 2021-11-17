import models
import data_handler
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR



if __name__ == '__main__':
    x_train, x_test, y_train, y_test = data_handler.data_handler()
    model = models.MLP_2_hidden(x_train.shape[1], 12, 12, 1)

    epochs = 0
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    loss_anterior = 0
    best_loss = float(np.inf)

    while True:
        model.train()
        optimizer.zero_grad()
        output = model.forward(x_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        print(round(loss.item(), 3), loss_anterior)
        model.eval()
        with torch.no_grad():
            test_output = model.forward(x_test)
            loss_test = criterion(test_output, y_test)
        print(f'epoch {epochs + 1} and LR {scheduler.get_last_lr()} done: loss of {loss.item()} and loss_test of {loss_test.item()}')
        model.train()
        scheduler.step()
        epochs += 1


        if round(loss.item(), 2) == loss_anterior:
            break
        else:
            loss_anterior = round(loss.item(), 2)

        if loss.item()<best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'model_2hidden.pth')