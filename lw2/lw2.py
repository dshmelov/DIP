import os

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    """
    Задание:
    реализуйте данный вид эквализации самостоятельно
    """
    images_dir = 'data'

    images_names = os.listdir(images_dir)

    for image_name in images_names:
        image_path = os.path.join(images_dir, image_name)

        bgr_image = cv.imread(filename=image_path)
        rgb_image = cv.cvtColor(src=bgr_image, code=cv.COLOR_BGR2RGB)



        pass

    pass

if __name__ == '__main__':
    main()