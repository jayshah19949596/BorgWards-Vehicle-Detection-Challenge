import os
import time
import shutil
import scipy.misc
import numpy as np
from glob import glob
import tensorflow as tf
from utils import helper
from core.model import Model
from utils import constants as cs


def run():
    segmentation_model = Model(50, 50, 0.0009)
    get_batches = helper.gen_batch_function()
    with tf.Session() as sess:
        correct_label = tf.placeholder(tf.int32, [None, None, None, segmentation_model.no_of_classes],
                                       name='correct_label')
        input_image, keep_prob, last_layer = segmentation_model.create_model_graph(sess, cs.VGG_MODEL,
                                                                                   correct_label)
        sess.run(tf.global_variables_initializer())
        for epoch in range(segmentation_model.no_of_epochs):
            print("EPOCH {} ....".format(epoch+1))

            for image, label in get_batches(segmentation_model.batch_size):

                res, _, loss = sess.run([last_layer,
                                         segmentation_model.train_op,
                                         segmentation_model.cross_entropy_loss],
                                        feed_dict={input_image: image,
                                                   correct_label: label,
                                                   keep_prob: 0.8})
                print("Loss: = {}".format(loss))
            print()

        run_inference_on_test(segmentation_model, sess, keep_prob, input_image)


def run_inference_on_test(model, sess, keep_prob, input_image):
    output_dir = os.path.join(cs.RES_DIR, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print('Training Finished. Saving test images to: {}'.format(output_dir))

    image_outputs = gen_test_output(sess, model.logits, keep_prob, input_image, cs.KITTI_TEST)

    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape=(160, 576)):

    for image_file in glob(os.path.join(data_folder, cs.PNG)):
        image = helper.get_image(image_file)
        image = helper.resize_image(image)

        im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 142]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)


if __name__ == '__main__':
    run()
