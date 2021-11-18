from torch.utils.data import DataLoader

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

    dataset = data_handler.data_handler('./data/bikes.csv')
    samples_train = int(len(dataset) * 0.8)
    train_set, val_set = torch.utils.data.random_split(dataset, [samples_train, len(dataset) - samples_train])

    dataloader_train = DataLoader(dataset=train_set, batch_size=15, drop_last=True, shuffle=True)
    dataloader_test = DataLoader(dataset=val_set, batch_size=5,drop_last=True, shuffle=True)

    model = models.MLP_1_hidden(dataset.features, 16, 1)

    epochs = 0
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.5)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)
    loss_anterior = 0
    best_loss = float(np.inf)

    while True:
        loss_train=0
        for x_train,y_train in iter(dataloader_train):
            x_train,y_train = x_train.float() ,y_train.float()
            model.train()
            optimizer.zero_grad()
            output = model.forward(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
        model.eval()
        with torch.no_grad():
            loss_test = 0
            for x_test, y_test in iter(dataloader_test):
                x_test, y_test = x_test.float(), y_test.float()
                test_output = model.forward(x_test)
                loss_test += criterion(test_output, y_test).item()
        print(f'epoch {epochs + 1} and LR {scheduler.get_last_lr()} done: loss of {loss_train} and loss_test of {loss_test}')
        scheduler.step()
        epochs += 1


        if loss_train == loss_anterior:
            break
        else:
            loss_anterior = loss_train

        if loss_train<best_loss:
            best_loss = loss_train
            torch.save(model.state_dict(), 'model_1_hidden.pth')