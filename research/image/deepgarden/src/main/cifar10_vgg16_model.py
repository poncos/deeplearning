# Copyright 2018 Esteban Collado.
#
# Licensed under the MIT License

import argparse

import tensorflow as tf

import re

parser = argparse.ArgumentParser()


parser.add_argument('--batch_size', type=int, default=128,
                    help='Number of images to process in a batch.')

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='Path to the CIFAR-10 data directory.')


FLAGS = parser.parse_args()

NUM_CLASSES = 10


def model(input):
    with tf.variable_scope('conv1') as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 1, 1],
                                      initializer=[[1, 1, 1 ], [1, 1, 1], [1, 1, 1]]
                                      )

        conv = tf.nn.conv2d(input, conv_filter, [1, 1, 1, 1], padding='SAME')

    return conv

'''
def inference(input):
    with tf.variable_scope('conv1') as scope:
        conv_filter = tf.get_variable('weights', shape=[3, 3, 3, 64],
                                      initializer=tf.truncated_normal_initializer(
                                          stddev=5e-2, dtype=tf.float16)
                                      )

        conv = tf.nn.conv2d(input, conv_filter, [1, 1, 1, 1], padding='SAME')
        
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0), dtype=tf.float16)
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        

    return conv

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        filter = tf.get_variable('weights', shape=[5, 5, 3, 64],
                                initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16)
                                )
        conv = tf.nn.conv2d(pool1, filter, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights', shape=[5, 5, 3, 64],
                                initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16)
                                )
        biases = tf.get_variable('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights', shape=[384, 192],
                        initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16)
                        )
        biases = tf.get_variable('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('weights', shape=[192, NUM_CLASSES],
                        initializer=tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float16)
                        )
        biases = tf.get_variable('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear
'''

def loss(logits, labels):
  """Add L2Loss to all the trainable variables.

  Add summary for "Loss" and "Loss/avg".
  Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

  Returns:
    Loss tensor of type float.
  """
  # Calculate the average cross entropy loss across the batch.
  labels = tf.cast(labels, tf.int64)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  # The total loss is defined as the cross entropy loss plus all of the weight
  # decay terms (L2 loss).
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def train():
    print("train")

    with tf.Graph().as_default():
        global_step = tf.train.get_or_create_global_step()


def main(argv=None):  # pylint: disable=unused-argument
    train()


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    tf.app.run()
