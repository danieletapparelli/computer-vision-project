import cv2
import numpy as np
from skimage import exposure
from skimage import feature

cap = cv2.VideoCapture('/Users/dant/Documents/Projects/computer_vision/Project/material/CV_basket.mp4')

hog = cv2.HOGDescriptor() #window size, block size, block stride, cells dimensions, n of histogram beans (64, 128),(16,16),(8,8),(8,8), 9
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while cap.isOpened():
    ret, frame = cap.read()
    frame_copy = frame.copy()

    
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    (rects, weights) =  hog.detectMultiScale(gray)

    #rect(x,y, height, width)
    for rect in rects:
        cv2.rectangle(gray, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 255), thickness=3)

    # (H, hogImage) = feature.hog(frame_copy, orientations=9, pixels_per_cell=(16,16) ,
    #                                 cells_per_block=(2,2), transform_sqrt=True, block_norm='L1', visualize=True)
    
    # hogImage = exposure.rescale_intensity(hogImage,out_range=(50, 255))
    # hogImage = hogImage.astype("uint8")


    cv2.imshow("visualize", gray)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# the following is necessary on the mac,
# maybe not on other platforms:
cv2.waitKey(1)