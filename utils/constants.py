import os

PROJECT_BASE_PATH = os.getcwd()[:len(os.getcwd())-len("utils")]


KITTI_TRAIN = PROJECT_BASE_PATH + "/data/kitti/training/"
KITTI_LABEL = PROJECT_BASE_PATH + "/data/kitti/training_label/"
KITTI_TEST = PROJECT_BASE_PATH + "/data/kitti/testing/"
BORG_WARD_TEST = PROJECT_BASE_PATH + "/data/borgward_test/"
VGG_MODEL = PROJECT_BASE_PATH + "/data/vgg_model"
SSD_PB_FILE = PROJECT_BASE_PATH + "/data/ssd_model/frozen_inference_graph.pb"
BORG_DATA = PROJECT_BASE_PATH + "/data/borgward_test/"

JPG = "*.jpg"
PNG = "*.png"

RES_DIR = PROJECT_BASE_PATH + "/results/"

# print(KITTI_TRAIN)
# print(KITTI_LABEL)
# print(KITTI_TEST)
# print(BORG_WARD_TEST)
# print("VGG_MODEL =", VGG_MODEL)
