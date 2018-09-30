
import utils.image_utils as iutils
import cifar10.networking.cifar10_vgg16_model as cm
import cifar10.constants as constants

from tkinter import *
import os
import numpy as np

from PIL import Image

import tensorflow as tf

import sys

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_image(image_data):
    pass


def cifar10_predict_main(image_path_file):
    var_names = ['conv1_test3', 'conv2_test3', 'conv3_test3', 'conv4_test3', 'fc1_test3', 'fc2_test3',
                 'softmax_linear_test3']
    img = Image.open(image_path_file, mode="r")
    img_bytes = img.tobytes()
    width, height = img.size

    reduced_image = iutils.convolve_1d_rgb(img_bytes, iutils.Dimension2d(width, height), iutils.Dimension2d(32, 32))
    input_img = [None]*cm.BATCH_SIZE
    for i in range(cm.BATCH_SIZE):
        input_img[i] = reduced_image

    print("Shape: ", np.array(input_img).shape)
    model = cm.forward_propagation(input=np.array(input_img, dtype=np.float32), variable_names=var_names)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore variables from disk.
        saver.restore(sess, constants.MODEL_FILE_PREFIX)
        result = sess.run(model)

        print("Result: ", CLASSES[np.argmax(result[0])])


if __name__ == '__main__':
    if sys.argv != 2:
        print("ERROR USAGE %s <image to clasify>" % sys.argv[0])

    cifar10_predict_main(sys.argv[1])
