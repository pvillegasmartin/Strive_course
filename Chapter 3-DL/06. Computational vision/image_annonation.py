import cv2
import matplotlib.pyplot as plt
import numpy as np


img = cv2.imread(r'caballos.jpg')

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_copy = rgb_img.copy()
cv2.rectangle(img_copy, (360,200),(490,350),(0,0,0),2)
cv2.putText(img_copy, 'Black horse', (360,200), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0),3)

cv2.rectangle(img_copy, (240,240),(350,340),(255,255,255),2)
cv2.putText(img_copy, 'White horse', (240,240), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),3)


plt.imshow(img_copy)
plt.show()
