
import utils.image_utils as iutils
import cifar10.networking.cifar10_vgg16_model as cm

from tkinter import *
import os
import numpy as np

from PIL import Image

import tensorflow as tf
import cifar10.constants as project_constants

import sys

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_image(image_data):
    pass


def cifar10_predict_main(image_path_file):
    img = Image.open(image_path_file, mode="r")
    img_bytes = img.tobytes()
    width, height = img.size

    print("reducing dimension for loaded image [%s]" % image_path_file)
    reduced_image = iutils.reduce_dim_average(img_bytes, iutils.Dimension2d(width, height), iutils.Dimension2d(32, 32))
    print("new image shape: ", np.array(reduced_image).shape)
    parameters = cm.initialize_parameters()

    model = cm.forward_propagation(input=np.array([reduced_image], dtype=np.float32), parameters=parameters)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        model_file = os.path.join(project_constants.MODEL_DIR_PATH, project_constants.MODEL_PREFIX)
        saver.restore(sess, model_file)
        result = sess.run(model)

        print("Result: ", CLASSES[np.argmax(result[0])])


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print("ERROR USAGE %s <image to clasify>" % sys.argv[0])
        exit(1)

    cifar10_predict_main(sys.argv[1])
