import cv2 as c
import numpy as np


def makeCoordinates(image, lines_parameters):
    slope, intercept = lines_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])


def averagedSlopeIntercept(image, line_s):
    left_fit = []
    right_fit = []
    for line in line_s:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fitAvg = np.average(left_fit, axis=0)
    right_fitAvg = np.average(right_fit, axis=0)
    leftLine = makeCoordinates(image, left_fitAvg)
    rightLine = makeCoordinates(image, right_fitAvg)
    return np.array([leftLine, rightLine])


def canny(image):
    gray = c.cvtColor(image, c.COLOR_RGB2GRAY)
    blur = c.GaussianBlur(gray, (5, 5), 0)
    canny_img = c.Canny(blur, 50, 150)
    return canny_img


def displayLines(img, lin):
    lineImg = np.zeros_like(img)
    if lin is not None:
        for x1, y1, x2, y2 in lin:
            c.line(lineImg, (x1, y1), (x2, y2), (255, 200, 0), 10)
    return lineImg


def ROI(image):
    height = image.shape[0]
    polygon = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    c.fillPoly(mask, polygon, 255)
    maskedImg = c.bitwise_and(image, mask)
    return maskedImg


"""Image Algo
image = c.imread('test_image.jpg')
laneImage = np.copy(image)
canny_image = canny(laneImage)
croppedImg = ROI(canny_image)
lines = c.HoughLinesP(croppedImg, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
avgLines = averagedSlopeIntercept(laneImage, lines)
line_image = displayLines(laneImage, avgLines)
combination = c.addWeighted(laneImage, 0.8, line_image, 1, 1)
c.imshow("Result", combination)
c.waitKey()
"""


cap = c.VideoCapture("test_video.mp4")
while cap.isOpened():
    _, frame = cap.read()
    canny_image = canny(frame)
    croppedImg = ROI(canny_image)
    lines = c.HoughLinesP(croppedImg, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    avgLines = averagedSlopeIntercept(frame, lines)
    line_image = displayLines(frame, avgLines)
    combination = c.addWeighted(frame, 0.8, line_image, 1, 1)
    c.imshow("Result", combination)
    if c.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
c.destroyAllWindows()
