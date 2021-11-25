import cv2
import matplotlib.pyplot as plt
import numpy as np

def load_image(img_path, width, height):
    bgr_img = cv2.imread(img_path)
    bgr_img = cv2.resize(bgr_img, (width, height), interpolation=cv2.INTER_AREA)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    return bgr_img, rgb_img, gray_img, hsv_img

def mask_background(base_img):
    values, counts = np.unique(base_img.reshape(-1, 3), axis=0, return_counts=True)
    ind = np.argmax(counts)
    H = values[ind][0]
    S = values[ind][1]
    V = values[ind][2]
    lower_h = int(max(H-2,0))
    upper_h = int(min(H+2,180))
    lower_bounds = (lower_h, 0, 0)
    upper_bounds = (upper_h, 255, 255)

    mask = cv2.inRange(base_img, lower_bounds, upper_bounds)
    return mask

def mask_background_2(base_img):

    retval,dst = cv2.threshold(base_img, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Create Kernel
    kernel = np.ones((2, 2), np.uint8)
    erotion = cv2.erode(dst, kernel, iterations=1)
    opening = cv2.morphologyEx(erotion, cv2.MORPH_OPEN, kernel, iterations=1)
    return dst

if __name__=="__main__":

    path = 'C:/Users/Pablo/Desktop/STRIVE AI/Strive_course/Chapter 3-DL/06. Computational vision/DAY 2/'
    """
    #image = input('Image to change background (remember to write the type file too)')

    #LOADING IMAGES IN DIFERENT COLORS SCALES
    image = 'green_image.png'
    bgr_img, rgb_img, gray_img, hsv_img = load_image(path+image)
    """
    new_image = 'new_back.png'
    new_image = 'barcelona.jpg'
    new_image = '80s.png'




    # define a video capture object
    vid = cv2.VideoCapture(0)
    # Capture the video frame by frame
    ret, frame = vid.read()

    bgr_new_img, rgb_new_img, gray_new_img, hsv_new_img = load_image(path + new_image, frame.shape[1], frame.shape[0])


    while True:

        # Capture the video frame by frame
        ret, frame = vid.read()
        if ret:
            #hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            #mask = mask_background(hsv_img)
            """
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            
            
            mask = mask_background_2(gray_img)

            #Change parts of the video with the background image
            frame[mask != 0] = bgr_new_img[mask != 0]
            """
            final = bgr_new_img.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            retval, dst = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
            contours, hierarchy = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            clean_contours = sorted_contours[0:3]
            x, y, w, h = cv2.boundingRect(clean_contours[1])
            final[y:y + h, x:x + w] = frame[y:y + h, x:x + w]

            # Display the resulting frame
            cv2.imshow('frame', final)

        # the 'q' button is set as the quitting button you may use any desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()