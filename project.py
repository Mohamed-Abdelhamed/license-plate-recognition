import cv2
import numpy as np

img = cv2.imread('imgs/usimages/car-02.png', 0)
imgg = cv2.imread('imgs/usimages/car-02.png')
height, width = img.shape
withoutnoise = cv2.medianBlur(img, 5)
ret, bw = cv2.threshold(withoutnoise, 150, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary", bw)
edges = cv2.Canny(bw, 0, 255)
cv2.imshow("Edges", edges)
kernel = np.ones((2, 2), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closing", closing)
contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
img1 = imgg.copy()
cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)
cv2.imshow("All Contours", img1)
cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt = None
img2 = imgg.copy()
cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 Contours", img2)
cv2.waitKey(0)
ROI = imgg.copy()
for contour in cnts:
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    plate = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    center_y = y + h / 2
    if area > 1000 and (w > h) and center_y > height / 2:
        ROI = imgg[y:y + h, x:x + w]
        break
            # x = plate.ravel()[0]
            # y = plate.ravel()[1]
            #
            # cv2.drawContours(img, [area], 0, 0, 5)

cv2.imshow('Plate', ROI)
# Drawing the selected contour on the original image
# print(NumberPlateCnt)
# cv2.drawContours(imgg, [NumberPlateCnt], -1, (0,255,0), 3)
cv2.waitKey(0)
cv2.destroyAllWindows()























#
# img = cv2.imread('car2.jpg', 0)
# imgg = cv2.imread('car2.jpg')
#
# height, width = img.shape
#
# withoutnoise = cv2.medianBlur(img, 3)
#
# cv2.imshow("Noise", withoutnoise)
# cv2.waitKey(0)
# ret, bw = cv2.threshold(withoutnoise, 150, 255, cv2.THRESH_BINARY)
# cv2.imshow("Binary", bw)
# #edges = cv2.Canny(withoutnoise, 200, 255)
#
# edges = cv2.Sobel(bw, cv2.CV_64F, 0, 1, ksize=3)
# kernel = np.ones((3, 3), np.uint8)
# #kernel2 = np.ones((7, 7), np.uint8)
#
# dilation=cv2.dilate(edges, kernel, iterations=2)
#
# #erode=cv2.erode(dilation, kernel, iterations=2)
#
# cv2.imshow("Edges", edges)
#
# #cv2.imshow("Dilation", dilation)
#
# #closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# cv2.imshow("Closing", dilation)
# cv2.waitKey(0)
