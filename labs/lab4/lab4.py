import os
import cv2
import numpy as np


def get_contours(img_path):
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")

    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(greyscale_image, (5, 5), 0)
    _, threshold_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    morphed_image = cv2.morphologyEx(threshold_image, cv2.MORPH_OPEN, kernel, iterations=1)
    inverted_image = 255 - morphed_image

    contours, _ = cv2.findContours(inverted_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        final_image = cv2.drawContours(np.zeros(inverted_image.shape, np.uint8), contours, -1, 255, 1)
        os.makedirs('./results', exist_ok=True)
        cv2.imwrite(os.path.join('./results', os.path.basename(img_path)), final_image)
        return contours[0]
    else:
        raise ValueError(f"No contours found in image '{img_path}'.")


def calculate_similarities(query_image_path, database_folder):
    similarities = {}
    query_contour = get_contours(query_image_path)

    for image_name in os.listdir(database_folder):
        image_path = os.path.join(database_folder, image_name)
        if os.path.isfile(image_path):
            try:
                contours = get_contours(image_path)
                similarity = cv2.matchShapes(query_contour, contours, cv2.CONTOURS_MATCH_I1, 0)
                similarities[image_name] = similarity
            except (FileNotFoundError, ValueError) as e:
                print(e)

    return similarities


def main():
    query_folder = './query/'
    database_folder = './database/'
    query_image_name = input('Enter the name of the query image: ')
    query_image_path = os.path.join(query_folder, query_image_name)

    if not os.path.isfile(query_image_path):
        print(f"Query image '{query_image_name}' not found in folder '{query_folder}'.")
        return

    similarities = calculate_similarities(query_image_path, database_folder)
    sorted_similarities = sorted(similarities.items(), key=lambda x: x[1])

    print("\nImages sorted by similarity:")
    for image_name, similarity in sorted_similarities:
        print(f'{image_name}:\t{similarity}')


if __name__ == '__main__':
    main()
