from torch import nn

model = nn.Sequential(nn.Linear(784, 400),
                      nn.ReLU(),
                      nn.Linear(400, 200),
                      nn.ReLU(),
                      nn.Linear(200, 100),
                      nn.ReLU(),
                      nn.Linear(100, 10),
                      nn.Softmax(dim=1))