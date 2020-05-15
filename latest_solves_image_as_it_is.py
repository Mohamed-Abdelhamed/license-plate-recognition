import cv2
import numpy as np

img_path = 'imgs/br/AYO9034.jpg'
gray = cv2.imread(img_path, 0)
image = cv2.imread(img_path)

height, width = gray.shape

withoutnoise = cv2.medianBlur(gray, 3)

_, bw = cv2.threshold(withoutnoise, 120, 255, cv2.THRESH_BINARY)
cv2.imshow("binary", bw)

edges = cv2.Canny(bw, 0, 255)
cv2.imshow("edges", edges)

kernal = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernal)
cv2.imshow("closing", closing)

contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
allContoursImage = image.copy()
cv2.drawContours(allContoursImage, contours, -1, (0, 0, 0), 3)
cv2.imshow("All Contours", allContoursImage)

cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCnt = None

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
    if 4 <= len(plate) <= 6:
        if area > 1000 and (w > h) and center_y > height / 2:
            if len(plate) == 4:
                ROI = image[y:y + h, x:x + w]
                print(len(plate))
                break

cv2.imshow('plate', ROI)

grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
_, threshROI = cv2.threshold(grayROI, 120, 255, cv2.THRESH_BINARY_INV)
cv2.imshow('threshROI', threshROI)

contours, _ = cv2.findContours(threshROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

copyROI = ROI.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(copyROI, (x, y), (x + w, y + h), (0, 255, 0), 1)
cv2.imshow('final ROI', copyROI)
cv2.waitKey(0)
cv2.destroyAllWindows()
