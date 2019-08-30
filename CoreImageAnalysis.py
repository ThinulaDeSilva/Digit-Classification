import numpy as np
import cv2 as cv
import ContourInfo as cont
from imutils import perspective as im


# this function identifies all contours in the image whose area is >= areaThres
def drawContours(image, areaThresh=(1000, 2000)):
    contours = []
    thresh = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    flag, thresh = cv.threshold(thresh, 90, 255, cv.THRESH_BINARY_INV)#cv.threshold(image, 90, 255, cv.THRESH_BINARY_INV)
    # thresh = 255 - thresh
    tempCont, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.imshow("Hello", thresh)
    cv.waitKey()
    # add all the contours whose area is greater than the given threshold
    # tempCont.sort(key=lambda cnt: -cv.contourArea(cnt))
    print("There are originally %d contours" %(len(tempCont)))
    for i in range(len(tempCont)):
        if areaThresh[0] <= cv.contourArea(tempCont[i]) <= areaThresh[1]:
            print(cv.contourArea(tempCont[i]))
            contours.append(tempCont[i])

    stencil = np.zeros(image.shape).astype(image.dtype)
    cv.drawContours(stencil, contours, -1, (255, 255, 255), cv.FILLED)
    result = cv.bitwise_and(image, stencil)
    canny = cv.Canny(result, 100, 200)
    cv.imshow("Hello", canny)

    return thresh, contours


# this function sorts the contours from top to bottom and left to right (currently the most time expensive function)
def sortContours(image, contours):
    NO_CONT = -1
    rows, cols = image.shape[:2]
    arr = np.arange(rows * cols, dtype=np.int32)
    arr = arr.reshape((rows, cols))
    labels = np.full_like(arr, NO_CONT, dtype=np.int32)

    # initializes the dependency tree with the contour information of all the contours
    dependency_tree = {}
    for ind, contour in enumerate(contours):
        cv.drawContours(labels, [contour], -1, ind, -1)
        dependency_tree[ind] = cont.ContourInfo(ind, contour)

    # construct the dependencies tree, processing cols from bottom up
    for c in range(cols):
        lastCont = NO_CONT
        # we scan from bottom up because we want the top contours to depend on the bottom ones
        # so that they can get pruned off the tree first
        for r in range(rows - 1, -1, -1):
            currCont = labels[r][c]
            if currCont != NO_CONT:
                if (lastCont != currCont) and (lastCont != NO_CONT):
                    dependency_tree[lastCont].add_dependency(currCont)
                lastCont = currCont

    return dependency_tree


# this function now extracts the contours from the original image
def extractContours(contours):
    rectangles = []
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        src_pts = im.order_points(box).astype("float32")
        rectangles.append(src_pts)
    return rectangles


def computeHomography(image, src_pts):
    # compute the dimensions of the rectangle
    width = np.linalg.norm(src_pts[1] - src_pts[0])
    height = np.linalg.norm(src_pts[2] - src_pts[1])

    # coordinates of the points in box points after rectangle is horizontal
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # the homography transformation matrix
    M, status = cv.findHomography(src_pts, dst_pts)

    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv.warpPerspective(image, M, (width, height))

    # writes the image to file and adds it to the images array
    cv.imwrite("Hello.png", warped)
    cv.waitKey(0)
