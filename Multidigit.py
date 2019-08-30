import cv2 as cv
import numpy as np
from keras.models import load_model
import CoreImageAnalysis as core

# This code is used to label the multi-digit numbers. It takes a look at the core images and tries to identify
# the numbers in the digit labels (mainly the first and last one) so that the files can be automatically named.

model = load_model("Model.h5")
# Read the input image
exampleNum = 2
image = cv.imread(r"C:\Users\GoldSpot_Cloudberry\OneDrive - Goldspot Discoveries Inc\Documents\Goldspot\Core Images\Digit Classification\Example " + str(exampleNum)+".jpg")
print(image.shape)
height = 300
width = image.shape[1]*height//(2*image.shape[0])
print(image.shape)

# Convert to grayscale and apply Gaussian filtering
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)

# this function is used to identify the labels in the middle of the image
# currently doesn't return something, but this should be changed so that it does
def detectMiddleLabels():
    # uses the contour code to identify the location of the cores
    grayCopy = gray.copy()
    imageArea = grayCopy.shape[0] * grayCopy.shape[1]
    maskImg, coreContours = core.drawContours(grayCopy, (int(imageArea / 100), int(imageArea / 50)))
    print("There are %d contours" % (len(coreContours)))

    tree = core.sortContours(maskImg, coreContours)
    while bool(tree):
        # we construct a list of candidates containing all nodes that are leaves in the dependency tree
        candidates = []
        candConts = []

        for node in tree.values():
            if node.is_leaf():
                candidates.append(node.index)
                candConts.append(coreContours[node.index])

        if len(candidates) > 2:
            print("Something is weird")
        elif len(candidates) == 2:
            rectangles = core.extractContours(candConts)
            # writing out each four arrays separately to maintain UMAT type for src_pts
            point1 = [rectangles[0][1][0], rectangles[0][1][1]]
            point2 = [rectangles[1][0][0], rectangles[1][0][1]]
            point3 = [rectangles[1][3][0], rectangles[1][3][1]]
            point4 = [rectangles[0][2][0], rectangles[0][2][1]]
            # since these contours are sorted left to right, we know rectangles[0] is to the left of rectangles[1]
            src_pts = np.array([point1, point2, point3, point4], dtype="float32")

            core.computeHomography(image, src_pts)

        candidates.sort(key=lambda n: tree[n].depth())

        # add the best contour to our sorted list and completely remove that contour from the dependency tree
        bestContour = tree.pop(candidates[0])
        for node in tree.values():
            node.remove_dependency(candidates[0])

# Threshold the image
flag, thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)

# Find contours in the image
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rectangles = []
for ctr in contours:
    area = cv.contourArea(ctr)
    rectangles.append(cv.boundingRect(ctr))
print(len(rectangles))

# For each rectangular region, resize/normalize image then feed into MNIST model
for rect in rectangles:
    # Draw the rectangles
    cv.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    # Make the rectangular region around the digit
    length = int(rect[3] * 1.6)
    point1 = max(int(rect[1] + int(rect[3]/2) - int(length/2)), 0)
    point2 = max(int(rect[0] + int(rect[2]/2) - int(length/2)), 0)
    roi = thresh[point1:point1+length, point2:point2+length]

    # Resize and Normalize the image
    roi = cv.resize(roi, (28, 28), interpolation=cv.INTER_AREA)
    roi = cv.dilate(roi, (3, 3))
    roi = cv.normalize(roi, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    arr = roi.reshape(1, roi.shape[0], roi.shape[1], 1)
    predNum = model.predict(arr)
    # the class of the largest probability prediction value
    number = np.where(predNum[0] == np.amax(predNum[0]))[0][0]
    # round the predictions array for printing
    predNum = np.around(predNum, 3)
    print(predNum, number)
    cv.putText(image, str(number), (rect[0], rect[1]), cv.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
    cv.imwrite("Detected.png", image)

cv.imshow("Resulting Image with Rectangular ROIs", image)
cv.imwrite("Result.png", image)
cv.waitKey()
