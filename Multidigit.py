import cv2 as cv
import numpy as np
from keras.models import load_model

model = load_model("Model.h5")
# Read the input image
exampleNum = 1
image = cv.imread(r"C:\Users\GoldSpot_Cloudberry\OneDrive - Goldspot Discoveries Inc\Documents\Goldspot\Core Images\Digit Classification\Example " + str(exampleNum)+".jpg")

# Convert to grayscale and apply Gaussian filtering
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)

# Threshold the image
flag, thresh = cv.threshold(gray, 90, 255, cv.THRESH_BINARY_INV)

# Find contours in the image
contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each contour
rectangles = [cv.boundingRect(ctr) for ctr in contours]
print(len(contours))

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
