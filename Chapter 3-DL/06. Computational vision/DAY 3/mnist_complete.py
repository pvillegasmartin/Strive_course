import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

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

numbers = ''

# Read image
path = "C:/Users/Pablo/Desktop/STRIVE AI/Strive_course/Chapter 3-DL/06. Computational vision/DAY 3/"
img = cv2.imread(path + 'numbers.jpeg')
img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

# Gray image masked
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
retval,dst = cv2.threshold(gray, 105, 255,cv2.THRESH_BINARY_INV)


# COUNTORS
contours, hierarchy = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
copy = img.copy()
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
left_right_contours, bb = sort_contours(sorted_contours[0:4], method="left-to-right")

#Let some padding to the numbers
margin = 75

model = Network()
model.load_state_dict(torch.load('model_mnist.pth'))
model.eval()

fig, axis = plt.subplots(figsize=(8,12), ncols=len(left_right_contours))

for countur,ax in zip(left_right_contours,axis):
    x, y, w, h = cv2.boundingRect(countur)
    #cv2.rectangle(copy, (x-margin, y-margin), (x + w + margin, y + h + margin), (255, 0, 0), 2)
    number = dst[y-margin:y+h+margin, x-margin:x+w+margin]
    number = cv2.resize(number, (28, 28), interpolation=cv2.INTER_AREA)
    tensor_gray_img = torch.from_numpy(number)
    tensor_gray_img.resize_(1, 784)

    # PREDICTION
    output = model(tensor_gray_img / 255)
    pred_y = torch.max(output, 1)[1].data.squeeze()
    numbers += str(pred_y.item())

    ax.axis('off')
    ax.imshow(cv2.cvtColor(number, cv2.COLOR_BGR2RGB))

print(numbers)
plt.show()