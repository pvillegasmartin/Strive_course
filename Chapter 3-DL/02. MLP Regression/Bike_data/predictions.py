import models
import data_handler
import torch

if __name__ == '__main__':
    _, x_test, _, y_test = data_handler.data_handler()
    model = models.MLP_2_hidden(x_test.shape[1], 6, 3, 1)
    model.load_state_dict(torch.load('model_2hidden.pth'))
    model.eval()
    output = model(x_test)
    print('done')