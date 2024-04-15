import numpy as np
import cv2


def interpolation(tocki):
    tocki.sort(key=lambda x: x[0])
    x_values = [p[0] for p in tocki]
    y_values = [p[1] for p in tocki]

    if x_values[0] != 0:
        x_values.insert(0, 0)
        y_values.insert(0, 0)
    if x_values[-1] != 255:
        x_values.append(255)
        y_values.append(255)

    return np.interp(np.arange(256), x_values, y_values).astype('uint8')


def contrast_stretching(slika, tocki):
    channels = cv2.split(slika)
    interpolation_curve = interpolation(tocki)
    stretched_channels = [interpolation_curve[channel] for channel in channels]
    stretchedImage = cv2.merge(stretched_channels)
    return stretchedImage


image = cv2.imread('monkey.webp')
# image = cv2.imread('milos_teodosic.jpg')
points = [(50, 20), (100, 50), (150, 200)]
stretched_image = contrast_stretching(image, points)

cv2.imshow('Original Image', image)
cv2.imshow('Contrast Stretched Image', stretched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
