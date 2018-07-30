import os
import cv2
import random
import scipy.misc
import numpy as np
from glob import glob
import tensorflow as tf
import utils.constants as cs
from tensorflow.python.tools import freeze_graph


def resize_image(image, image_shape=(160, 576)):
    """
        This function resizes the given image

        :param image: numpy array
                    : image which is to be resized
        :param image_shape : tuple
                             : shape into which you want to resize the image in argument 1
        :return resized_image : numpy array
                               : resized image
    """
    return scipy.misc.imresize(image, image_shape)


def get_image(image_path):
    """
        This function reads the image and returns the read image

        :param image_path: string
                         : path of image which is to be read
        :return read_image : numpy array
                           : image
    """
    return scipy.misc.imread(image_path)


def iterate_folder(data_folder, file_extension):
    """
        This iterates through the data folder
        and stores the files in the data folder
        with given extension

        :param data_folder   : string
                             : data folder which is to be iterated
        :param file_extension: string
                             : file extension whose file's path is to be stored
        :return image_paths : list of strings
                            : list of files path with given extension
    """
    image_paths = glob(os.path.join(data_folder, file_extension))
    return image_paths


def gen_batch_function():
    def get_batches_fn(batch_size, image_shape=(160, 576)):

        image_paths = iterate_folder(cs.KITTI_TRAIN, cs.PNG)

        label_paths = {}
        for path in iterate_folder(cs.KITTI_LABEL, cs.PNG):
            label_paths[os.path.basename(path)] = path

        background_color = np.array([255, 0, 0])
        random.shuffle(image_paths)

        for batch_i in range(0, len(image_paths), batch_size):

            images = []
            labels = []

            for image_file in image_paths[batch_i:batch_i+batch_size]:
                label_file = label_paths[os.path.basename(image_file)]

                image = resize_image(get_image(image_file), image_shape)
                label_image = resize_image(get_image(label_file), image_shape)

                gt_bg = np.all(label_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                label_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                labels.append(label_image)

            yield np.array(images), np.array(labels)

    return get_batches_fn


def freeze_model(sess, logs_path, latest_checkpoint, model, pb_file_name, freeze_pb_file_name):
    """
    :param sess     : tensor-flow session instance which creates the all graph information

    :param logs_path: string
                      directory path where the checkpoint files are stored

    :param latest_checkpoint: string
                              checkpoint file path

    :param model: model instance for extracting the nodes explicitly

    :param pb_file_name: string
                         Name of trainable pb file where the graph and weights will be stored

    :param freeze_pb_file_name: string
                                Name of freeze pb file where the graph and weights will be stored

    """
    print("logs_path =", logs_path)
    tf.train.write_graph(sess.graph.as_graph_def(), logs_path, pb_file_name)
    input_graph_path = os.path.join(logs_path, pb_file_name)
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = latest_checkpoint
    output_graph_path = os.path.join(logs_path, freeze_pb_file_name)
    clear_devices = False
    output_node_names = ",".join(model.nodes)
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    initializer_nodes = ""
    freeze_graph.freeze_graph(input_graph_path,
                              input_saver_def_path,
                              input_binary,
                              input_checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_path,
                              clear_devices,
                              initializer_nodes)


if __name__ == "__main__":
    kitti_train = iterate_folder(cs.KITTI_TRAIN, cs.PNG)
    kitti_label = iterate_folder(cs.KITTI_LABEL, cs.PNG)
    print(kitti_train)
    print(kitti_label)
