import cv2
import numpy as np
import operator
import os

allContoursWithData = []
validContoursWithData = []

# MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30

npaClassifications = np.loadtxt("classifications.txt", np.float32)
npaFlattenedImages = np.loadtxt("flattened_images.txt", np.float32)
npaClassifications = npaClassifications.reshape((npaClassifications.size, 1))
kNearest = cv2.ml.KNearest_create()
kNearest.train(npaFlattenedImages, cv2.ml.ROW_SAMPLE, npaClassifications)
imgTestingNumbers = cv2.imread("plate.jpg")
imgGray = cv2.cvtColor(imgTestingNumbers, cv2.COLOR_BGR2GRAY)
imgBlurred = cv2.GaussianBlur(imgGray, (5, 5), 0)
imgThresh = cv2.adaptiveThreshold(imgBlurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
imgThreshCopy = imgThresh.copy()
imgContours, _ = cv2.findContours(imgThreshCopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for imgContour in imgContours:
    if 100 < cv2.contourArea(imgContour):
        validContoursWithData.append(imgContour)

validContoursWithData.sort(key =lambda ctr: cv2.boundingRect(ctr)[0])

strFinalString = ""
for contourWithData in validContoursWithData:
    [intX, intY, intWidth, intHeight] = cv2.boundingRect(contourWithData)
    cv2.rectangle(imgTestingNumbers, (intX, intY), (intX + intWidth, intY + intHeight), (0, 255, 0), 2)
    imgROI = imgThresh[intY: intY + intHeight, intX: intX + intWidth]
    imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
    npaROIResized = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
    npaROIResized = np.float32(npaROIResized)
    retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized, k=1)
    strCurrentChar = str(chr(int(npaResults[0][0])))
    strFinalString += strCurrentChar

print("\n" + strFinalString + "\n")

cv2.imshow("imgTestingNumbers", imgTestingNumbers)
cv2.waitKey(0)

cv2.destroyAllWindows()