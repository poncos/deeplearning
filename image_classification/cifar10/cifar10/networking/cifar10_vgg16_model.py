# Copyright 2018 Esteban Collado.
#
# Licensed under the MIT License

import tensorflow as tf

DEFAULT_VARIABLE_NAMES = ['conv1', 'conv2', 'conv3', 'conv4', 'fc1', 'fc2', 'softmax_linear']

BATCH_SIZE = 200
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_DEPTH = 3
NUM_CLASSES = 10

INPUT_PLACEHOLDER = 'X_INPUT'
LABELS_PLACEHOLDER = 'Y_LABELS'


def create_placeholder():
    x_placeholder = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH], name=INPUT_PLACEHOLDER)
    y_placeholder = tf.placeholder(tf.int32, [None], name=LABELS_PLACEHOLDER)

    return x_placeholder, y_placeholder


def initialize_parameters():
    parameters = {
        "w1": tf.get_variable("w1", shape=[3, 3, 3, 64],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),
        "w2": tf.get_variable("w2", shape=[3, 3, 64, 64],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),
        "w3": tf.get_variable("w3", shape=[3, 3, 64, 64],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),
        "w4": tf.get_variable("w4", shape=[3, 3, 64, 64],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),
        "w5": tf.get_variable("w5", shape=[4096, 384],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),
        "w6": tf.get_variable("w6", shape=[384, 192],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),
        "w7": tf.get_variable("w7", shape=[192, 10],
                              initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)),

        "b1": tf.get_variable('b1', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        "b2": tf.get_variable('b2', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        "b3": tf.get_variable('b3', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        "b4": tf.get_variable('b4', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32),
        "b5": tf.get_variable('b5', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32),
        "b6": tf.get_variable('b6', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32),
        "b7": tf.get_variable('b7', [10], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    }

    return parameters


def create_conv2d_layer(inputs, name, weight, bias, strides=[1, 1, 1, 1], padding='SAME'):
    with tf.variable_scope(name) as scope:
        conv = tf.nn.conv2d(inputs, weight, strides, padding)
        pre_activation = tf.nn.bias_add(conv, bias)
        activation = tf.nn.relu(pre_activation, name=scope.name)

    return activation


def forward_propagation(input, parameters):

    conv1 = create_conv2d_layer(input, 'conv1', parameters['w1'], parameters['b1'])

    conv2 = create_conv2d_layer(conv1, 'conv2', parameters['w2'], parameters['b2'])

    pool1 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    conv3 = create_conv2d_layer(pool1, 'conv3', parameters['w3'], parameters['b3'])

    conv4 = create_conv2d_layer(conv3, 'conv4', parameters['w4'], parameters['b4'])

    pool2 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    flattened_conv = tf.reshape(pool2, shape=[-1, parameters['w5'].get_shape().as_list()[0]])
    fc1 = tf.nn.relu(tf.matmul(flattened_conv, parameters['w5']) + parameters['b5'], name='fc1')

    fc2 = tf.nn.relu(tf.matmul(fc1, parameters['w6']) + parameters['b6'], name='fc2')

    softmax_linear = tf.add(tf.matmul(fc2, parameters['w7']), parameters['b7'], name='softmax')

    return softmax_linear


def compute_cost(logits, labels):
    # loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return loss


def backward_propagation(cost, learning_rate=LEARNING_RATE):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    return optimizer


