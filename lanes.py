#Cropped Image workings not clear
#30:30
import cv2 as c
import numpy as np
import matplotlib.pyplot as plt


def make_coordinates(img, line_parameters):
    try:
        slope, intercept = line_parameters
    except TypeError:
        slope, intercept = 0.001, 0
    y1 = img.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])


def avg_slope_intercept(image, lines):
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        paramters = np.polyfit((x1, x2), (y1, y2), 1)
        print("P:", paramters)
        slope = paramters[0]
        intercept = paramters[1]
        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))
        left_avg = np.average(left, axis=0)
        right_avg = np.average(right, axis=0)
        left_line = make_coordinates(image, left_avg)
        right_line = make_coordinates(image, right_avg)
        return np.array([left, right])



def canny(image):
    gray = c.cvtColor(image, c.COLOR_RGB2GRAY)
    blur = c.GaussianBlur(gray, (5, 5), 0)
    cany = c.Canny(blur, 50, 150)
    return cany

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            c.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


def area_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(240, height), (600, height), (265, 165)] #3rd point is in (x,y)
        ])
    mask = np.zeros_like(image)
    c.fillPoly(mask, polygons, (255, 255, 255))
    masked_image = c.bitwise_and(image, mask)
    return masked_image


img = c.imread('rd.jpg')
lane_img = np.copy(img)

cnny = canny(lane_img)
ci = area_of_interest(cnny)
lines = c.HoughLinesP(ci, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
avg_line = avg_slope_intercept(lane_img, lines)
line_img = display_lines(lane_img, avg_line)
mimg = c.addWeighted(lane_img, 0.8, line_img, 1, 1)

c.imshow('result', line_img)
c.waitKey(0)

# c.imshow('Result', canny(lane_img)) #displays the image
# c.waitKey(0) # Delays the closing of cv2.imshow(), by specifying 0, it means till any other key is pressed from the keyboard
# plt.imshow(canny(lane_img))
# cani = canny(lane_img)
# c.imshow('Result', cani) #displays the image
# plt.show() #it's working is same as cv2.waitkey()
# plt.imshow(area_of_interest(lane_img))
# cropped_image = area_of_interest(cani)
# lines = c.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# line_image = display(lane_img, lines)
# c.imshow('Result', cropped_image) #displays the image
# plt.show() #it's working is same as cv2.waitkey()
