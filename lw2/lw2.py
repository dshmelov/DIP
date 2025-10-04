import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def equalize(grayscale_image: np.ndarray) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int_)
    for v in grayscale_image.flatten():
        hist[v] += 1

    cdf = hist.cumsum()

    min_cdf = cdf[np.nonzero(cdf)][0]

    num_pixels = grayscale_image.size

    cdf_normalized = ((cdf - min_cdf) * 255 / (num_pixels - min_cdf)).astype(dtype=np.uint8)

    equalized_image = cdf_normalized[grayscale_image]

    return equalized_image


def main() -> None:
    """
    Задание:
    реализуйте данный вид эквализации самостоятельно
    """
    images_dir = 'data'
    images_names = os.listdir(images_dir)

    for image_name in images_names:
        image_path = os.path.join(images_dir, image_name)

        grayscale_image = cv.imread(filename=image_path, flags=cv.IMREAD_GRAYSCALE)
        equalized_image = equalize(grayscale_image=grayscale_image)

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 2, 1)
        plt.title('Исходное изображение')
        plt.imshow(grayscale_image, cmap='gray')

        plt.subplot(2, 2, 2)
        plt.title('Гистограмма (до)')
        plt.hist(grayscale_image.flatten(), bins=256, range=(0, 256), color='gray')

        plt.subplot(2, 2, 3)
        plt.title('Эквализированное')
        plt.imshow(equalized_image, cmap='gray')

        plt.subplot(2, 2, 4)
        plt.title('Гистограмма (после)')
        plt.hist(equalized_image.flatten(), bins=256, range=(0, 256), color='gray')

        plt.show()

        pass

    pass


if __name__ == '__main__':
    main()
