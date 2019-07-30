import cv2 as cv
import numpy as np
from keras.models import load_model
import CoreImageAnalysis as core

model = load_model("Model.h5")
# Read the input image
exampleNum = 5
image = cv.imread(r"C:\Users\GoldSpot_Cloudberry\OneDrive - Goldspot Discoveries Inc\Documents\Goldspot\Core Images\Digit Classification\Example " + str(exampleNum)+".png")
print(image.shape)
height = 600
width = image.shape[1]*height//(2*image.shape[0])
image = cv.resize(image, (height, width))

# Convert to grayscale and apply Gaussian filtering
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)

# mser = cv.MSER_create()
# vis = image.copy()
# regions = mser.detectRegions(image)
#
# hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
# print(len(hulls))
#
# newHulls = [hull for hull in hulls if 300 <= cv.contourArea(hull) <= 1000]
# cv.polylines(vis, newHulls, 1, (0, 255, 0))
# print(len(newHulls))
# cv.imshow('img', vis)
# cv.waitKey()

grayCopy = gray.copy()
  # Thresholding the image
#maskImg = 255-thresh
coreContours = core.drawContours(grayCopy, 0)#thresh, 0)
cv.drawContours(grayCopy, coreContours, -1, (0, 255, 0), 3)
cv.waitKey()
print("There are %d contours" %(len(coreContours)))

# # Defining a kernel length
# kernel_length = np.array(image).shape[1]//40
# # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
# verticle_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_length))
# # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
# hori_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length, 1))
# # A kernel of (3 X 3) ones.
# kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
#
# # Morphological operation to detect verticle lines from an image
# img_temp1 = cv.erode(maskImg, verticle_kernel, iterations=3)
# verticle_lines_img = cv.dilate(img_temp1, verticle_kernel, iterations=3)
# cv.imwrite("verticle_lines.png", verticle_lines_img)

# Threshold the image
flag, thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)

# Find contours in the image
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rectangles = []
for ctr in contours:
    area = cv.contourArea(ctr)
    if area >= 1000:
        print(area)
        rectangles.append(cv.boundingRect(ctr))
#rectangles = [cv.boundingRect(ctr) for ctr in contours]
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

cv.imshow("Resulting Image with Rectangular ROIs", image)
cv.waitKey()
