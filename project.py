import cv2

import numpy as np




img = cv2.imread('car1.jpg',0)

height, width= img.shape

withoutnoise= cv2.medianBlur(img,3)



ret , bw =cv2.threshold(withoutnoise,150,255,cv2.THRESH_BINARY)
cv2.imshow("binary", bw)
edges = cv2.Canny(bw,0,255)

kernal = np.ones((3,3),np.uint8)
#kernal2 = np.ones((7,7),np.uint8)

#dilation=cv2.dilate(edges,kernal,iterations=1)

#erode=cv2.erode(dilation,kernal,iterations=2)

cv2.imshow("edges", edges)
#cv2.imshow("dilation", dilation)

ROI=img
closing = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernal)
cv2.imshow("closing", closing)

contours,_=cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
     area = cv2.contourArea(contour)
     x, y, w, h = cv2.boundingRect(contour)
     plate = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, True), True)
     center_y = y + h / 2

     if area > 2000 and (w > h) and center_y > height / 2:

     #x=plate.ravel()[0]
     #y=plate.ravel()[1]
     #
        #cv2.drawContours(img,[area],0,0,5)
         ROI = img[y:y + h, x:x + w]

cv2.imshow('platee',ROI)

cv2.waitKey(0)
cv2.destroyAllWindows()
