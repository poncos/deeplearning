# Copyright 2018 Esteban Collado.
#
# Licensed under the MIT License

import tensorflow as tf


def forward_propagation(input=input, batch_size=128):


    with tf.variable_scope('conv1') as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 3, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                      )

        conv = tf.nn.conv2d(input, conv_filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)

        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('conv2') as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 64, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                      )

        conv = tf.nn.conv2d(conv1, conv_filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)

        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    pool1 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    with tf.variable_scope('conv3') as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 64, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                      )

        conv = tf.nn.conv2d(pool1, conv_filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)

        conv3 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('conv4') as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 64, 64],
                                      initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                      )

        conv = tf.nn.conv2d(conv3, conv_filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        pre_activation = tf.nn.bias_add(conv, biases)

        conv4 = tf.nn.relu(pre_activation, name=scope.name)

    pool2 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool4')

    with tf.variable_scope('fc1') as scope:
        # TODO change [1, -1] with [dim, -1]
        flattened_conv = tf.reshape(pool2, [1, -1])

        fc_weights_dim1 = flattened_conv.get_shape()[1].value
        fc_weights = tf.get_variable('weights', shape=[fc_weights_dim1, 384],
                                     initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                     )

        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc1 = tf.nn.relu(tf.matmul(flattened_conv, fc_weights) + biases, name=scope.name)

    with tf.variable_scope('fc2') as scope:
        fc_weights = tf.get_variable('weights', shape=[384, 192],
                                     initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32)
                                     )

        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc2 = tf.nn.relu(tf.matmul(fc1, fc_weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        soft_weights = tf.get_variable('weights', shape=[192, 10],
                                     initializer=tf.truncated_normal_initializer(
                                         stddev=1/192.0, dtype=tf.float32))

        biases = tf.get_variable('biases', [10], initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)

        softmax_linear = tf.add(tf.matmul(fc2, soft_weights), biases, name=scope.name)

    return softmax_linear


def get_loss(logits, labels):
    loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

    return loss


def backward_propagation(loss):
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    return train


def model(total_loss, global_step):
    pass


def cifar10_main():
    print("train")


