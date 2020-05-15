import cv2
import numpy as np

def getRmax(hist, num):
    for intensity in reversed(range(255)):
        if intensity in hist and hist[intensity] > num:
            return intensity


def getRmin(hist, num):
    for intensity in range(255):
        if intensity in hist and hist[intensity] > num:
            return intensity

img_path = 'imgs/br/PJI7589.jpg'
gray = cv2.imread(img_path, 0)
image = cv2.imread(img_path)

height, width = gray.shape

withoutnoise = cv2.medianBlur(gray, 3)

image = withoutnoise.copy()
copy = np.zeros(image.shape, image.dtype)
height, width = image.shape

# return each intensity with it's count and putting it in a hashmap
unique, counts = np.unique(image, return_counts=True)
hist = dict(zip(unique, counts))

percentage = 0.005
num = round(float(percentage) * (height * width))  # number of counts ommit before.

smin = 0
smax = 255
rmax = getRmax(hist, num)
rmin = getRmin(hist, num)

for row in range(height):
    for col in range(width):
        r = image[row, col]
        s = (r - rmin) * ((smax - smin) / (rmax - rmin)) + smin
        if s > 255:
            s = 255
        if s < 0:
            s = 0
        copy[row, col] = s

cv2.imshow("after contract stretching", copy)
cv2.imshow("before contract stretching", image)
cv2.waitKey(0)

ret, bw = cv2.threshold(copy, 100, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", bw)

edges = cv2.Canny(bw, 0, 255)
cv2.imshow("edges", edges)

kernal = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernal)
cv2.imshow("closing", closing)
# kernal2 = np.ones((7,7),np.uint8)
# dilation=cv2.dilate(edges,kernal,iterations=1)
# erode=cv2.erode(dilation,kernal,iterations=2)
# cv2.imshow("dilation", dilation)


contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContoursImage = image.copy()
cv2.drawContours(allContoursImage, contours, -1, (0, 0, 0), 3)
cv2.imshow("All Contours", allContoursImage)

cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt = None  # we currently have no Number plate contour

# Top 30 Contours
top30ContoursImage = image.copy()
cv2.drawContours(top30ContoursImage, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 Contours", top30ContoursImage)
cv2.waitKey(0)

ROI = image.copy();
for contour in cnts:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    plate = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    center_y = y + h / 2

    if area > 1000 and (w > h) and center_y > height / 2:
        ROI = image[y:y + h, x:x + w]
        break
        # x=plate.ravel()[0]
        # y=plate.ravel()[1]
        # cv2.drawContours(gray,[area],0,0,5)

cv2.imshow('plate', ROI)

ret, thresh1 = cv2.threshold(ROI, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

contours, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

copyROI = ROI.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(copyROI, (x, y), (x + w, y + h), (0, 255, 0), 1)
cv2.imshow('final', copyROI)

cv2.waitKey(0)
cv2.destroyAllWindows()

#
# gray = cv2.imread('car2.jpg',0)
# image = cv2.imread('car2.jpg')
#
# height, width= gray.shape
#
# withoutnoise= cv2.medianBlur(gray,3)
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
