import cv2
import numpy as np
import matplotlib.pyplot as plt

# import os

# save_dir = 'saved_plots'
# os.makedirs(save_dir, exist_ok=True)

# image = cv2.imread('apple.jpeg', cv2.IMREAD_GRAYSCALE)
image = cv2.imread('nikeBox.jpg', cv2.IMREAD_GRAYSCALE)

filters = [
    np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
    np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]),
    np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
    np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]]),
    np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]),
    np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
]

filtered_images = [cv2.filter2D(image, -1, fillter) for fillter in filters]

combined_edges = np.zeros_like(image)
for filtered in filtered_images:
    combined_edges = np.maximum(combined_edges, filtered)

plt.figure(figsize=(12, 8))
for i in range(len(filters)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(filtered_images[i], cmap='gray')
    plt.title(f"Filter {i + 1}")
    plt.axis('off')
# plt.savefig(os.path.join(save_dir, f"filter_{i + 1}.png"))

# threshold_value = 30
threshold_value = 50
# threshold_value = 80
thresholded_edges = (combined_edges > threshold_value).astype(np.uint8) * 255

plt.figure(figsize=(6, 6))
plt.imshow(combined_edges, cmap='gray')
plt.title("Combined Edges")
plt.axis('off')

# plt.savefig(os.path.join(save_dir, "combined_edges.png"))

plt.figure(figsize=(6, 6))
plt.imshow(thresholded_edges, cmap='gray')
plt.title(f"Threshold = {threshold_value}")
plt.axis('off')

# plt.savefig(os.path.join(save_dir, "thresholded_edges.png"))

plt.show()
