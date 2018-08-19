import unittest

import main.cifar10_loader as cl
import main.cifar10_vgg16_model as cm

import numpy as np

import tensorflow as tf


class FixedLengthRecordReaderTestCase(unittest.TestCase):

    def test_model_static(self):
        print("========================= test_model_static")
        labels, images = cl.load_data_set(max_records=1)
        print("Read [%s] images and [%s] labels." % (len(images), len(labels)))

        model = cm.forward_propagation(input=images, batch_size=1)

        assert(model is not None)

        session = tf.Session()
        session.run(tf.global_variables_initializer())
        model_result = session.run(model)

        print("Convolution result: ", model_result)
        print("Result with shape: ", model_result.shape)
        print("labels: ", np.array(labels).shape)
        print(labels)
        print("========================= test_model_static")
        # print("Records: ", len(records))
        # print("labels: ", len(labels))
        # print("Item0: ", records[0])

    def test_model_static_batches(self):
        print("========================= test_model_static_batches")
        labels, images = cl.load_data_set(max_records=50)
        print("Read [%s] images and [%s] labels." % (len(images), len(labels)))
        var_names = ['conv1_test2', 'conv2_test2', 'conv3_test2', 'conv4_test2', 'fc1_test2', 'fc2_test2',
                     'softmax_linear_test2']
        model = cm.forward_propagation(input=images, variable_names=var_names, batch_size=50)

        assert(model is not None)

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        model_result = session.run(model)

        print("Convolution result: ", model_result)
        print("Result with shape: ", model_result.shape)
        print("labels: ", np.array(labels).shape)
        one_hot_labels = tf.one_hot(labels, 10)
        print(labels)
        print(session.run(one_hot_labels))
        print("========================= test_model_static")


    def test_model_static_multiple_batches(self):
        print("========================= test_model_static_batches")
        labels, images = cl.load_data_set(max_records=cm.BATCH_SIZE*2)
        print("Read [%s] images and [%s] labels." % (len(images), len(labels)))
        var_names = ['conv1_test3', 'conv2_test3', 'conv3_test3', 'conv4_test3', 'fc1_test3', 'fc2_test3',
                     'softmax_linear_test3']
        input_placeholder, labels_placeholder = cm.create_placeholder()

        model = cm.forward_propagation(input=input_placeholder, variable_names=var_names)
        cost = cm.compute_cost(model, tf.one_hot(labels_placeholder, 10))
        optimizer = cm.backward_propagation(cost)

        with tf.Session() as session:
            assert (model is not None)
            session.run(tf.global_variables_initializer())

            _, minibatch_cost = session.run([optimizer, cost],
                                       feed_dict={input_placeholder: images[0:cm.BATCH_SIZE],
                                                  labels_placeholder: labels[0:cm.BATCH_SIZE]})

            print("Convolution result: ", minibatch_cost)
            print("labels: ", np.array(labels).shape)
            print(labels)
            print("========================= test_model_static")

            _, minibatch_cost = session.run([optimizer, cost],
                                       feed_dict={input_placeholder: images[cm.BATCH_SIZE:cm.BATCH_SIZE*2],
                                                  labels_placeholder: labels[cm.BATCH_SIZE:cm.BATCH_SIZE*2]})

            print("Convolution result: ", minibatch_cost)
            print("labels: ", np.array(labels).shape)
            print(labels)
            print("========================= test_model_static")


if __name__ == '__main__':
    unittest.main()
