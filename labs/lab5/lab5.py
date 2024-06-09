import glob
import os

import cv2
import numpy as np


def load_images(dir: str) -> list[np.ndarray]:
    files = glob.glob(f'{dir}/*')
    images = []

    for file in files:
        images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

    return images


def sift(image: np.ndarray) -> tuple:
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def get_matches(desc_1, desc_2) -> list:
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_1, desc_2, k=2)
    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    return good


def main() -> None:
    posters = load_images('Database')
    descriptors = [sift(image) for image in posters]

    img_number = input('Select image number (1, 2, 3): ')

    if img_number not in {'1', '2', '3'}:
        print("Invalid selection. Please enter 1, 2, or 3.")
        return

    image_path = f'hw7_poster_{img_number}.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Error: Image at path {image_path} could not be loaded.")
        return

    kp, desc = sift(image)

    image_descriptors = [get_matches(desc, desc_db[1]) for desc_db in descriptors]
    best_index = np.argmax([len(matches) for matches in image_descriptors])

    poster_keypoints = cv2.drawKeypoints(posters[best_index], descriptors[best_index][0], None,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_keypoints = cv2.drawKeypoints(image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    image_1 = np.concatenate((image, posters[best_index]), axis=1)
    image_2 = np.concatenate((image_keypoints, poster_keypoints), axis=1)

    cv2.imshow('Combined Images', image_1)
    cv2.imshow('Keypoints', image_2)
    cv2.imshow('Matches', cv2.drawMatchesKnn(image, kp, posters[best_index], descriptors[best_index][0],
                                             image_descriptors[best_index], None, flags=2))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
