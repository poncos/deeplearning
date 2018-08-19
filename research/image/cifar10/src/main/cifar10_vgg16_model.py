# Copyright 2018 Esteban Collado.
#
# Licensed under the MIT License

import tensorflow as tf

import numpy as np

DEFAULT_VARIABLE_NAMES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'softmax_linear']

BATCH_SIZE = 200
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10

INPUT_PLACEHOLDER = 'X_INPUT'
LABELS_PLACEHOLDER = 'Y_LABELS'


def create_placeholder():
    x_placeholder = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH], name=INPUT_PLACEHOLDER)
    y_placeholder = tf.placeholder(tf.int32, [BATCH_SIZE], name=LABELS_PLACEHOLDER)

    return x_placeholder, y_placeholder


def forward_propagation(input=input, variable_names=DEFAULT_VARIABLE_NAMES, batch_size=BATCH_SIZE):

    with tf.variable_scope(variable_names[0]) as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 3, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                      )

        conv = tf.nn.conv2d(input, conv_filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)

        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope(variable_names[1]) as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 64, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                      )

        conv = tf.nn.conv2d(conv1, conv_filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)

        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    pool1 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    with tf.variable_scope(variable_names[2]) as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 64, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                      )

        conv = tf.nn.conv2d(pool1, conv_filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)

        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope(variable_names[3]) as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 64, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                      )

        conv = tf.nn.conv2d(conv3, conv_filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)

        conv4 = tf.nn.relu(pre_activation, name=scope.name)

    pool2 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool4')

    with tf.variable_scope(variable_names[4]) as scope:
        flattened_conv = tf.reshape(pool2, [batch_size, -1])

        fc_weights_dim1 = flattened_conv.get_shape()[1].value
        fc_weights = tf.get_variable('weights', shape=[fc_weights_dim1, 384],
                                     initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                     )

        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc1 = tf.nn.relu(tf.matmul(flattened_conv, fc_weights) + biases, name=scope.name)

    with tf.variable_scope(variable_names[5]) as scope:
        fc_weights = tf.get_variable('weights', shape=[384, 192],
                                     initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                     )

        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc_weights) + biases, name=scope.name)

    with tf.variable_scope(variable_names[6]) as scope:
        soft_weights = tf.get_variable('weights', shape=[192, 10],
                                     initializer=tf.truncated_normal_initializer(
                                         stddev=1/192.0, dtype=tf.float32))

        biases = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)

        softmax_linear = tf.add(tf.matmul(fc2, soft_weights), biases, name=scope.name)

    return softmax_linear


def compute_cost(logits, labels):
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return loss


def backward_propagation(cost, learning_rate=0.0001):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    return optimizer


def model(total_loss, global_step):
    pass


def cifar10_main():
    print("train")


