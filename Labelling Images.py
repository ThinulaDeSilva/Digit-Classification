import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv

# This code takes core images and labels them for image segmentation training.
# It identifies the first and last digit labels (in green), and then masks the
# image so that the labels are white and everything else is black.

def processImg(image):
    edges = cv.Canny(image, 50, 150)
    tempCont, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contours = []
    for contour in tempCont:
        if cv.contourArea(contour) >= 10000:
            contours.append(contour)
    #print(contours[0])
    #print(contours[1])
    print(len(tempCont))
    stencil = np.zeros(image.shape).astype(image.dtype)
    cv.drawContours(stencil, tempCont, -1, (255, 255, 255), -1)
    contours.append(contours[1])
    print(len(contours))
    #contours.sort(key=lambda ctr: -cv.contourArea(ctr))
    stencilOne = np.zeros(image.shape).astype(image.dtype)
    cv.drawContours(stencilOne, contours, 0, (255, 255, 255), -1)

    stencilTwo = np.zeros(image.shape).astype(image.dtype)
    cv.drawContours(stencilTwo, contours, 1, (255, 255, 255), -1)

    result = cv.bitwise_or(stencilOne, stencilTwo)
    cv.imwrite("Drawn Contours.png", stencil)
    cv.imwrite("Contours1.png", stencilOne)
    cv.imwrite("Contours2.png", stencilTwo)
    cv.imwrite("Contours.png", result)
    return stencilOne

def calculateImage(image):
    edges = cv.Canny(image, 0, 100)
    tempCont, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in tempCont:
        print(cv.contourArea(contour))

saveFile = os.path.abspath(r".\Final Labels")
stemFile = os.path.abspath(r".\New Labels")
fileList = os.listdir(stemFile)
saveFile += r'\New'
print(fileList[0])
print(len(fileList))

fileList = [file for file in fileList if file.endswith(".png")]
print(len(fileList))

#fileList = ["DDH-RI-09-001-87.75-94.90.png"]

blackColour = np.array([0, 0, 0]).astype('uint8')
whiteColour = np.array([255, 255, 255]).astype('uint8')

for file in [fileList[1]]:
    fileName = str(stemFile) + str(r"\New")[:1] + str(file)
    image = cv.imread(fileName)
    try:
        #cv.imshow("original", image)
        np.putmask(image, image != whiteColour, blackColour)
        cv.imwrite(saveFile[:len(saveFile)-3] + file, image)
        image = processImg(image)
        #cv.imwrite(saveFile[:len(saveFile)-3] + file, image)
        #image = cv.imread("..\Contours.png")
        #plt.imshow(image)
        #plt.show()
        calculateImage(image)
        #cv.imshow("new", image)
        #plt.imshow(image)
        #plt.show()
    except:
        print(file)
