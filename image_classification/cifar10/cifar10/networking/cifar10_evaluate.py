
import cifar10.networking.cifar10_vgg16_model as cm
import cifar10.constants as project_constants
import cifar10.dataloader.cifar10_loader as cl

import os
import numpy as np

import tensorflow as tf


def evaluate(session, accuracy, input_placeholder, labels_holder, images, labels):
    accuracy_value = 0
    num_batches = len(images) // cm.BATCH_SIZE
    for i in range(num_batches):
        minibatch = images[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]
        minibatch_labels = labels[cm.BATCH_SIZE * i:cm.BATCH_SIZE + cm.BATCH_SIZE * i]

        accuracy_value += session.run([accuracy],
                                      feed_dict={input_placeholder: minibatch, labels_holder: minibatch_labels})[0]
    return accuracy_value/num_batches


def cifar10_evaluate_main():
    loader_config = cl.LoaderDataSetConfig()
    loader_config.load_evaluate_dataset = True

    print("Reading evaluation dataset")
    labels, images = cl.load_data_set(config=loader_config)
    print("Read [%s] images and [%s] labels." % (len(images), len(labels)))

    x_placeholder, y_placeholder = cm.create_placeholder()
    parameters = cm.initialize_parameters()

    logits = cm.forward_propagation(input=x_placeholder, parameters=parameters)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_placeholder)
    accuracy_fnc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # Restore variables from disk.
        model_file = os.path.join(project_constants.MODEL_DIR_PATH, project_constants.MODEL_PREFIX)
        saver.restore(sess, model_file)

        accuracy_eval = evaluate(sess, accuracy_fnc, x_placeholder, y_placeholder, images, labels)

        print("Accuracy (evaluation): %f" % accuracy_eval)


if __name__ == '__main__':
    cifar10_evaluate_main()
