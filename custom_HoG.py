import time
import os

import matplotlib.pyplot as plt
import numpy as np

from skimage import filters
# Use torchvision to install (load) the data
from torchvision import datasets, transforms


def load_MNIST():
    tensor_transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='./MNIST_dataset', transform=tensor_transform, train=True,
                                download=True)
    test_data = datasets.MNIST(root='./MNIST_dataset', transform=tensor_transform, train=False,
                               download=True)

    train_images = train_data.data.numpy()
    train_labels = train_data.targets.numpy()

    test_images = test_data.data.numpy()
    test_labels = test_data.targets.numpy()

    return train_images, train_labels, test_images, test_labels


def pad_image(image, pixels_per_cell):
    # scikit-image HOG does not pad the image by default
    image_height = image.shape[0]
    image_width = image.shape[1]

    patch_height, patch_width = pixels_per_cell
    if image_height % patch_height == 0 and image_width % patch_width == 0:
        return image
    # Calculate the right dimensions for an exact fit of patch size to image dimensions
    padded_height = int(np.ceil(image_height / patch_height) * patch_height)
    padded_width = int(np.ceil(image_width / patch_width) * patch_width)

    # Pad the image to match the right dimensions
    pad_height = padded_height - image_height
    pad_width = padded_width - image_width
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='edge')
    return padded_image.astype(dtype=np.float64)


def divide_matrix_to_blocks(image, block_size=(8, 8), stride=None):
    # If the stride is not specified, then the blocks are not overlapping. Assu
    if stride is None:
        stride = block_size[0]
    # Pad matrix (image) with zeros if block size does not perfectly fit the dimensions
    img = pad_image(image, block_size)
    num_blocks_height = (img.shape[0] - block_size[0]) // stride + 1
    num_blocks_width = (img.shape[1] - block_size[1]) // stride + 1
    blocks = np.zeros((num_blocks_height, num_blocks_width, block_size[0], block_size[1]))

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            y_start = i * stride
            y_end = y_start + block_size[0]
            x_start = j * stride
            x_end = x_start + block_size[1]

            blocks[i, j] = img[y_start:y_end, x_start:x_end]
    # Reshape to 3 dimensions because we do not care about location of block in image
    # blocks = blocks.reshape((-1, block_size[0], block_size[1]))
    return blocks


def get_gradients(image):
    grad_x = filters.sobel_h(image)
    grad_y = filters.sobel_v(image)

    # Calculate the magnitude and angle of the gradients
    grad_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
    # Angle from radians to degrees
    grad_angle = np.arctan2(grad_y, grad_x) * 180 / np.pi

    return grad_magnitude, grad_angle


def create_histogram(block_magnitudes, block_angles):
    # 0 to 180 histogram with 10 degrees step. 180 will be added to 170 and 0
    hist = np.zeros(18)

    num_of_pixels = len(block_magnitudes) * len(block_magnitudes)
    # Stack the values and reshape to pixel,mag,angle because location of pixel in block does not interest us
    pixel_values = np.stack((block_magnitudes, block_angles), axis=-1).reshape(num_of_pixels, 2)
    for pixel in range(pixel_values.shape[0]):
        pixel_mag = pixel_values[pixel][0]
        pixel_angle = pixel_values[pixel][1]
        # Find the index in the histogram by rounding to the closest decade
        ind = round(pixel_angle / 10.0)
        # If angle = 180 degrees, add it to the 0 and 170 degrees elements
        if ind > len(hist) - 1:
            hist[0] += pixel_mag / 2
            hist[-1] += pixel_mag / 2
        else:
            hist[ind] += pixel_mag
    # Make the 18 bins into 9 bins
    actual_hist = np.zeros(9)
    # Add the non-central values
    for i in range(0, 18, 2):
        actual_hist[i // 2] += hist[i]
    # Add the central values to adjacent numbers
    for i in range(1, 18, 2):
        left = i // 2
        # If it is the last element of the initial histogram then add half of it to the first element of the final hist
        right = i // 2 + 1 if i // 2 + 1 < len(actual_hist) - 1 else 0

        actual_hist[left] += hist[i] / 2

        actual_hist[right] += hist[i] / 2

    return actual_hist


def get_feature_vector(hist_per_cell, block_size=(2, 2)):
    num_blocks_height = hist_per_cell.shape[0] // block_size[0]
    num_blocks_width = hist_per_cell.shape[1] // block_size[1]
    # Code snippet I used from stack overflow
    if hist_per_cell.shape[0] % block_size[0] != 0 or hist_per_cell.shape[1] % block_size[1] != 0:
        raise Exception("Number of Cells do not agree with Cells per Block dimensions")
    hists_per_blocks = hist_per_cell.reshape(num_blocks_height, block_size[0], -1, block_size[1]).swapaxes(1,
                                                                                                           2).reshape(
        block_size[0], block_size[1], -1)
    normalized_vectors = np.zeros(hists_per_blocks.shape)
    # Normalize vectors
    for i in range(hists_per_blocks.shape[0]):
        for j in range(hists_per_blocks.shape[1]):
            vector = hists_per_blocks[i][j]
            norm = np.linalg.norm(hists_per_blocks[i][j])
            if norm != 0:
                normalized_vectors[i][j] = vector / norm
            else:
                normalized_vectors[i][j] = vector
    # Flatten the result to a feature vector
    feature_vector = normalized_vectors.reshape(-1)

    return feature_vector


def custom_HoG(images, cell_size=(8, 8), cells_per_block=(2, 2)):
    print("Computing HOG feature vectors. . .")
    start = time.time()
    feature_vectors = []
    for i in range(images.shape[0]):
        img = images[i]
        img_blocks = divide_matrix_to_blocks(img, cell_size)
        # Calculate the histograms per block
        img_histograms_per_cell = np.zeros((img_blocks.shape[0], img_blocks.shape[1]) + (9,))
        for j in range(img_blocks.shape[0]):
            for k in range(img_blocks.shape[1]):
                block = img_blocks[j][k]
                mag, angle = get_gradients(block)
                block_hist = create_histogram(mag, angle)
                img_histograms_per_cell[j, k] = block_hist
        img_vector = get_feature_vector(img_histograms_per_cell, block_size=cells_per_block)
        feature_vectors.append(img_vector)
    print(f'HOG vectors calculated after {time.time() - start:.1f}s')
    return np.array(feature_vectors)


if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = load_MNIST()

    train_vectors = custom_HoG(train_images, cell_size=(14, 14))
    print(train_vectors.shape, train_vectors)
