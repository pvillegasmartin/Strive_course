import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):

    # Defining the layers, 128, 64, 10 units each
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    # Forward pass through the network, returns the output logits
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

def preprocess_image(img):
    img = cv2.imread(img)
    img = img[680:1030, 950:1200]
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dst = 255-dst

    return dst

if __name__ == '__main__':

    path = 'C:/Users/Pablo/Desktop/STRIVE AI/Strive_course/Chapter 3-DL/06. Computational vision/DAY 2/'
    """
    #Loading a number done by hand
    img = preprocess_image(path + '3.jpg')
    """
    #Loading numbers done with paint
    img = cv2.imread(path + '0.png')
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



    tensor_gray_img = torch.from_numpy(img)
    tensor_gray_img.resize_(1, 784)

    #PREDICTION
    model = Network()
    model.load_state_dict(torch.load('model_mnist.pth'))
    model.eval()
    output = model(tensor_gray_img/255)
    pred_y = torch.max(output, 1)[1].data.squeeze()
    print(pred_y)
    

    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
