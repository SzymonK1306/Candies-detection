import cv2
import numpy as np
from pathlib import Path
import os
import pandas as pd

candies = 0

path = Path(f'{os.getcwd()}')
file_list = list(path.glob('images/*.png'))
total_number = 0

data_frame = pd.read_csv('cukierki.csv', header=None)
second_column = data_frame.iloc[:, 1].values  # Access the second column (index 1)

roi_list = [(370, 110, 200, 200), (360, 300, 200, 200), (370, 470, 200, 200), (390, 650, 200, 200),
            (620, 230, 200, 200), (650, 425, 200, 200), (700, 600, 200, 200), (750, 770, 200, 200),
            (910, 320, 200, 200), (960, 480, 200, 200), (1040, 650, 200, 200), (1110, 800, 200, 200),
            (1200, 330, 200, 200), (1280, 480, 200, 200), (1380, 620, 200, 200), (1500, 740, 200, 200)]
#
image = cv2.imread('images/image_335.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.medianBlur(gray, 25)

# gray = cv2.GaussianBlur(gray, (17, 17), 0)

_, gray = cv2.threshold(gray, 203, 255, cv2.THRESH_TOZERO)

gray = cv2.GaussianBlur(gray, (25, 25), 0)

gray = cv2.medianBlur(gray, 25)

_, gray = cv2.threshold(gray, 55, 255, cv2.THRESH_TOZERO)


cv2.imshow('cos', gray)
cv2.waitKey()

for roi_element in roi_list:
    # Example of defining ROI coordinates
    roi_x = roi_element[0]  # X-coordinate of the top-left corner of ROI
    roi_y = roi_element[1]  # Y-coordinate of the top-left corner of ROI
    roi_width = roi_element[2]  # Width of ROI
    roi_height = roi_element[3]  # Height of ROI

    cv2.rectangle(image, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0))

    # Extract the ROI from the grayscale image
    roi = gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    #
    # roi = cv2.GaussianBlur(roi, (21, 21), 0)
    # #
    # _, roi = cv2.threshold(roi, 170, 255, cv2.THRESH_TOZERO)
    # #
    # roi = cv2.GaussianBlur(roi, (13, 13), 0)
    #
    #roi = cv2.medianBlur(roi, 9)
    #
    # _, roi = cv2.threshold(roi, 50, 255, cv2.THRESH_TOZERO)
    #
    # roi = cv2.GaussianBlur(roi, (3, 3), 0)
    #
    # roi = cv2.medianBlur(roi,3)
    #
    # _, roi = cv2.threshold(roi, 29, 255, cv2.THRESH_TOZERO)

    # roi = cv2.medianBlur(roi, 11)

    # cv2.imshow("Result", roi)
    # cv2.waitKey(0)

    # Perform Hough Circles transformation
    circles = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=75, param2=20, minRadius=50,
                               maxRadius=65)

    # If circles are found, draw them on the original image
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x + roi_x, y + roi_y), r, (0, 255, 0), 2)


print('Total number: ', total_number)
cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# for file in file_list:
#     image = cv2.imread(str(file))
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Example of defining ROI coordinates
#     roi_x = 100  # X-coordinate of the top-left corner of ROI
#     roi_y = 100  # Y-coordinate of the top-left corner of ROI
#     roi_width = 200  # Width of ROI
#     roi_height = 200  # Height of ROI
#
#     # Extract the ROI from the grayscale image
#     roi = gray[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

# weights = np.ones(len(file_list)) / len(file_list)
#
# weighted_sum = np.zeros_like(cv2.imread('images/image_19'))
#
# # Press the green button in the gutter to run the script.
# for i, file in enumerate(file_list):
#     if i != 0:
#         image = cv2.imread(str(file))
#         sum_image += image
#     else:
#         sum_image = cv2.imread(str(file))
# sum_image = sum_image/140
# cv2.imshow('Sum Image', sum_image)
# cv2.imwrite('sum.jpg', sum_image)
# cv2.waitKey(0)
