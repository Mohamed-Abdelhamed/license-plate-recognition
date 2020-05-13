import cv2

import numpy as np




img = cv2.imread('car2.jpg',0)



withoutnoise= cv2.medianBlur(img,3)



ret , bw =cv2.threshold(withoutnoise,150,255,cv2.THRESH_BINARY)
# cv2.imshow("binary", bw)
edges = cv2.Canny(bw,0,255)

kernal = np.ones((5,5),np.uint8)
#kernal2 = np.ones((7,7),np.uint8)


#dilation=cv2.dilate(edges,kernal,iterations=2)


#erode=cv2.erode(dilation,kernal,iterations=2)

cv2.imshow("edges", edges)
#cv2.imshow("dilation", dilation)


closing = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernal)
cv2.imshow("closing", closing)

# contours,_=cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
# for contour in contours:
#     plate=cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
#     #x=plate.ravel()[0]
#     #y=plate.ravel()[1]
#     if len(plate)==4:
#         cv2.drawContours(img,[plate],0,0,4)
#
# cv2.imshow('plate',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
