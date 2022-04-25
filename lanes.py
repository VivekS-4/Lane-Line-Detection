#30:30
import cv2 as c
import numpy as np
import matplotlib.pyplot as plt


def canny(lane_image):
    gray = c.cvtColor(lane_image, c.COLOR_RGB2GRAY)
    blur = c.GaussianBlur(gray, (5, 5), 0)
    cany = c.Canny(blur, 50, 150)
    return cany


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

# c.imshow('Result', canny(lane_img)) #displays the image
# c.waitKey(0) # Delays the closing of cv2.imshow(), by specifying 0, it means till any other key is pressed from the keyboard
# plt.imshow(canny(lane_img))
canny = canny(lane_img)
plt.show() #it's working is same as cv2.waitkey()
plt.imshow(area_of_interest(lane_img))
cropped_image = area_of_interest(canny)
plt.show() #it's working is same as cv2.waitkey()
