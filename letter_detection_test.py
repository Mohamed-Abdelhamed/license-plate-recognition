import cv2
import numpy as np

# for filename in os.listdir('imgs/br'):
for filename in ['JPQ9870.jpg']:
    if filename.endswith(".jpg"):
        img = cv2.imread('imgs/br/' + filename, 0)
        imgg = cv2.imread('imgs/br/' + filename)

        height, width = img.shape

        withoutnoise = cv2.medianBlur(img, 3)

        ret, bw = cv2.threshold(withoutnoise, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow("binary", bw)
        edges = cv2.Canny(bw, 0, 255)

        kernal = np.ones((3,3),np.uint8)
        #kernal2 = np.ones((7,7),np.uint8)

        #dilation=cv2.dilate(edges,kernal,iterations=1)

        #erode=cv2.erode(dilation,kernal,iterations=2)

        cv2.imshow("edges", edges)
        #cv2.imshow("dilation", dilation)

        ROI=img
        closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernal)
        cv2.imshow("closing", closing)

        contours, _ = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        img1 = imgg.copy()
        cv2.drawContours(img1, contours, -1, (0,0,0), 3)
        cv2.imshow("All Contours", img1)

        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        NumberPlateCnt = None  #we currently have no Number plate contour

        # Top 30 Contours
        img2 = imgg.copy()
        cv2.drawContours(img2, cnts, -1, (0,255,0), 3)
        cv2.imshow("Top 30 Contours", img2)
        cv2.waitKey(0)
        ROI = imgg.copy();

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

        # gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
        # ret,thresh1 = cv2.threshold(gray, 100, 255,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
        # # cv2.imshow('thresh1', thresh1)
        #
        # rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        #
        # contourss, hierarchy = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        #
        # im2 = ROI.copy()
        # for cnt in contourss:
        #     x, y, w, h = cv2.boundingRect(cnt)
        #     cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.imshow('final', im2)
        intChar = cv2.waitKey(0)
        if intChar == ord('1'):
            f = open('br/full.txt', 'a')
            f.write(filename + "\n")
            f.close()
        elif intChar == ord('2'):
            f = open('br/half.txt', 'a')
            f.write(filename + "\n")
            f.close()
        elif intChar == ord('3'):
            f = open('br/wrong.txt', 'a')
            f.write(filename + "\n")
            f.close()
        else:
            f = open('br/notdetected.txt', 'a')
            f.write(filename + "\n")
            f.close()
        cv2.destroyAllWindows()