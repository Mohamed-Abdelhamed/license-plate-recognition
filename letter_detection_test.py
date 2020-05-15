import cv2

import numpy as np
import imutils


img = cv2.imread('car2.jpg',0)
imgg = cv2.imread('car2.jpg')

height, width= img.shape

withoutnoise= cv2.medianBlur(img,3)



#
#
#
# def getRmax(hist,num):
#     hist2=hist
#     number=num
#     for intensity in reversed(range(255)):
#         if(intensity in hist2 and hist2[intensity] > number):
#             return intensity
#
# def getRmin(hist,num):
#     hist2=hist
#     number=num
#     for intensity in range(255):
#         if(intensity in hist2 and hist2[intensity] > number):
#             return intensity
#
#
#
#
#
#
#
# image = withoutnoise.copy()
# copy = np.zeros(image.shape, image.dtype)
# height, width = image.shape
#
# #return each intensity with it's count and putting it in a hashmap
# unique, counts = np.unique(image, return_counts=True)
# hist=dict(zip(unique, counts))
#
#
#
# percentage=0.003
# num=round(float(percentage) * (height*width)) #number of counts ommit before.
#
# smin=0
# smax=255
# rmax = getRmax(hist,num)
# rmin=getRmin(hist,num)
#
# for row in range(height):
#     for col in range(width):
#
#         r=image[row,col]
#         s=(r-rmin) * ((smax-smin)/(rmax-rmin)) +smin
#         if(s>255):
#             s=255
#         if(s<0):
#             s=0
#         copy[row,col]=s
#
#
#
# cv2.imshow("after", copy)
# cv2.imshow("before", image)
# cv2.waitKey(0)
#


ret , bw =cv2.threshold(withoutnoise,100,255,cv2.THRESH_BINARY)
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

contours,_=cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
img1 = imgg.copy()
cv2.drawContours(img1, contours, -1, (0,0,0), 3)
cv2.imshow("4- All Contours", img1)



cnts=sorted(contours, key = cv2.contourArea, reverse = True)[:90]
NumberPlateCnt = None #we currently have no Number plate contour

# Top 30 Contours
img2 = imgg.copy()
cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
cv2.imshow("5- Top 30 Contours", img2)
cv2.waitKey(0)
ROI=imgg.copy();


for contour in cnts:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    plate = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    center_y = y + h / 2

    if area > 3000 and (w > h) and center_y > height / 2:

        ROI = imgg[y:y + h, x:x + w]
        break
            #x=plate.ravel()[0]
            #y=plate.ravel()[1]
            #
            #cv2.drawContours(img,[area],0,0,5)

cv2.imshow('platee',ROI)


gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
ret,thresh1 = cv2.threshold(gray, 100, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
cv2.imshow('thresh1', thresh1)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

contourss, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

im2 = ROI.copy()
for cnt in contourss:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
cv2.imshow('final', im2)
cv2.waitKey(0)
cv2.destroyAllWindows()























#
# img = cv2.imread('car2.jpg',0)
# imgg = cv2.imread('car2.jpg')
#
# height, width= img.shape
#
# withoutnoise= cv2.medianBlur(img,3)
#
# cv2.imshow("noise", withoutnoise)
# cv2.waitKey(0)
# ret , bw =cv2.threshold(withoutnoise,150,255,cv2.THRESH_BINARY)
# cv2.imshow("binary", bw)
# #edges = cv2.Canny(withoutnoise,200,255)
#
# edges =cv2.Sobel(bw,cv2.CV_64F,0,1,ksize=3)
# kernal = np.ones((3,3),np.uint8)
# #kernal2 = np.ones((7,7),np.uint8)
#
# dilation=cv2.dilate(edges,kernal,iterations=2)
#
# #erode=cv2.erode(dilation,kernal,iterations=2)
#
# cv2.imshow("edges", edges)
#
# #cv2.imshow("dilation", dilation)
#
# #closing = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernal)
# cv2.imshow("closing", dilation)
# cv2.waitKey(0)
