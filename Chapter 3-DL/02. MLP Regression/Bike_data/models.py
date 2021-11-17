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

class MLP_2_hidden(nn.Module):
    def __init__(self, num_features,num_hidden_1, num_hidden_2, labels_output=1):
        super().__init__()
        self.cf1 = nn.Linear(num_features, num_hidden_1)
        self.cf2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.cf3 = nn.Linear(num_hidden_2, labels_output)
    def forward(self, x):
        x = self.cf1(x)
        x = F.relu(x)
        x = self.cf2(x)
        x = F.relu(x)
        x = self.cf3(x)
        return x
