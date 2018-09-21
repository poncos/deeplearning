
import main.image_utils as iutils
import cifar10.cifar10_vgg16_model as cm

from tkinter import *
import os
import numpy as np

from PIL import Image

import tensorflow as tf

NUM_TRAINING_RECORDS = 10000

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))

MODEL_FILE_PREFIX = os.path.join(CURRENT_DIR_PATH, '../../resources/model/-980')

print("current dir %s" % CURRENT_DIR_PATH)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def preprocess_image(image_data):
    pass


def cifar10_predict_main():
    var_names = ['conv1_test3', 'conv2_test3', 'conv3_test3', 'conv4_test3', 'fc1_test3', 'fc2_test3',
                 'softmax_linear_test3']
    img = Image.open("/store/datasets/downloaded_images/animals/cat/cat3.jpg", mode="r")
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
        # saver.restore(sess, "/tmp/model/-980")
        saver.restore(sess, MODEL_FILE_PREFIX)
        result = sess.run(model)

        print("Result: ", CLASSES[np.argmax(result[0])])


if __name__ == '__main__':
    cifar10_predict_main()
