#pytesseract

import cv2
import numpy as np
import pytesseract
import os

for filename in os.listdir('imgs/br'):
    if filename.endswith(".jpg"):
        img_path = 'imgs/br/' + filename
        image = cv2.imread(img_path)
        gray = cv2.imread(img_path, 0)

        height, width = gray.shape

        withoutnoise = cv2.medianBlur(gray, 5)
        _, bw = cv2.threshold(withoutnoise, 150, 255, cv2.THRESH_BINARY)
        # cv2.imshow("Binary", bw)

        edges = cv2.Canny(bw, 0, 255)
        # cv2.imshow("Edges", edges)

        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("Closing", closing)

        contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        allContoursImage = image.copy()
        cv2.drawContours(allContoursImage, contours, -1, (0, 255, 0), 3)
        # cv2.imshow("All Contours", allContoursImage)

        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        NumberPlateCnt = None
        top30ContoursImage = image.copy()
        cv2.drawContours(top30ContoursImage, cnts, -1, (0, 255, 0), 3)
        # cv2.imshow("Top 30 Contours", top30ContoursImage)
        cv2.waitKey(0)

        ROI = image.copy()
        for contour in cnts:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            plate = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            center_y = y + h / 2
            if area > 1000 and (w > h) and center_y > height / 2:
                ROI = image[y:y + h, x:x + w]
                break
                # x = plate.ravel()[0]
                # y = plate.ravel()[1]
                #
                # cv2.drawContours(gray, [area], 0, 0, 5)

        cv2.imwrite('plate_project.jpg', ROI)
        # grayROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        # _, ROI = cv2.threshold(grayROI, 100, 255, cv2.THRESH_BINARY)
        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Fil22es\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(ROI, config='-l eng --oem 1 --psm 7 tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
        print('real plate number is', filename[:-4])
        print("Plate number is: ", text)
        cv2.imshow('Plate', ROI)
        intChar = cv2.waitKey(0)
        if intChar == ord('1'):
            f = open('detect.txt', 'a')
            f.write(filename + "\n")
            f.close()
        if intChar == ord('2'):
            f = open('notdetect.txt', 'a')
            f.write(filename + "\n")
            f.close()
        cv2.destroyAllWindows()























#
# gray = cv2.imread('car2.jpg', 0)
# image = cv2.imread('car2.jpg')
#
# height, width = gray.shape
#
# withoutnoise = cv2.medianBlur(gray, 3)
#
# cv2.imshow("Noise", withoutnoise)
# cv2.waitKey(0)
# _, bw = cv2.threshold(withoutnoise, 150, 255, cv2.THRESH_BINARY)
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
