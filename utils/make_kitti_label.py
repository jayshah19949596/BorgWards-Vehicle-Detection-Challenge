import os
import cv2
import time
import shutil
import scipy.misc
import numpy as np
from glob import glob
import utils.constants as cs
import matplotlib.pyplot as plt


def read_image(image_file):
    return scipy.misc.imread(image_file)


def extract_car_pixels(img):
    img[:, :, 0] = 255
    img[:, :, 1] = 0
    img[:, :, 2][img[:, :, 2] != 142] = 0
    return img


def display(img):
    plt.imshow(img)
    plt.show()


def save_car_label_image(img, name, runs_dir=cs.KITTI_LABEL):
    output_dir = os.path.join(runs_dir)
    scipy.misc.imsave(os.path.join(output_dir, name), img)


def iterate_data(data_folder):
    image_paths = glob(os.path.join(data_folder, cs.PNG))
    for image_path in image_paths:
        label_image = read_image(image_path)
        extract_car_pixels(label_image)
        save_car_label_image(label_image, image_path.split("/")[-1], runs_dir=cs.KITTI_LABEL)


# iterate_data(cs.SEM_DATA)
