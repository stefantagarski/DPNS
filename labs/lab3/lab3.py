import os
import cv2
import numpy as np


def preprocess_image(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morphed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel=np.ones((5, 5)), iterations=1)
    inverted_image = 255 - morphed_image
    return inverted_image


def find_contours(image):
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_image = cv2.drawContours(np.zeros(image.shape, np.uint8), contours, -1, 255, 1)
    return final_image


def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    images = [os.path.join(input_folder, i)
              for i in os.listdir(input_folder)
              if os.path.isfile(os.path.join(input_folder, i))]

    for img in images:
        image = cv2.imread(img)
        processed_image = preprocess_image(image)
        final_image = find_contours(processed_image)
        cv2.imwrite(os.path.join(output_folder, os.path.basename(img)), 255 - final_image)


if __name__ == "__main__":
    input_folder = './database/'
    output_folder = './result/'
    process_images(input_folder, output_folder)
