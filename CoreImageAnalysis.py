import numpy as np
import cv2 as cv
import ContourInfo as cont


# this function identifies all contours in the image whose area is >= areaThres
def drawContours(image, areaThres=100*100):
    contours = []
    flag, thresh = cv.threshold(image, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    tempCont, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #cv.drawContours(image, tempCont, -1, (0, 255, 0), 3)
    #cv.waitKey()
    # add all the contours whose area is greater than the given threshold
    for i in range(len(tempCont)):
        if cv.contourArea(tempCont[i]) >= areaThres:
            contours.append(tempCont[i])

    return contours


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

    # sort the dependency tree by removing one leaf at a time
    sorted_contours = []
    while bool(dependency_tree):
        # we construct a list of candidates containing all nodes that are leaves in the dependency tree
        candidates = []
        for node in dependency_tree.values():
            if node.is_leaf():
                candidates.append(node.index)

        # sort the candidates by their depth, which gives precedence to the leftmost contours
        candidates.sort(key=lambda n: dependency_tree[n].depth())

        # add the best contour to our sorted list and completely remove that contour from the dependency tree
        bestContour = dependency_tree.pop(candidates[0])
        sorted_contours.append(contours[bestContour.index])
        for node in dependency_tree.values():
            node.remove_dependency(candidates[0])

    return sorted_contours


# this function now extracts the contours from the original image
def extractContours(image, contours):
    rectangles = []
    for i in range(len(contours)):
        rect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(rect)
        box = np.int0(box)
        rectangles.append(box)
    return rectangles

