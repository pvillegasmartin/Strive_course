import cv2
import matplotlib.pyplot as plt
import numpy as np


text = input('Introduce meme text:')
text = text.upper()
text = text.split()
img = cv2.imread(r'meme_base.png')
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_width = img.shape[1]
img_heigth = img.shape[0]
img_copy = rgb_img.copy()
j = 0

while text!=[]:
    for i in range(len(text),0,-1):
        sentece = ' '.join(text[:i])
        size = cv2.getTextSize(sentece, cv2.FONT_HERSHEY_PLAIN, 2, 5)[0]
        if size[0] < img_width-20:
            cv2.putText(img_copy, sentece, (int((img_width-size[0])/2),int(25+size[1]*j*1.2)), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0),5)
            cv2.putText(img_copy, sentece, (int((img_width-size[0])/2),int(25+size[1]*j*1.2)), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255),2)
            text = text[i:]
            j+=1
            break
if j<4:
    #cv2.imwrite('meme_family.png', cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    plt.imshow(img_copy)
    plt.show()
else:
    print('Too many text')