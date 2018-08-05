import unittest

import main.cifar10_loader as cl
import main.cifar10_vgg16_model as cm

import tensorflow as tf


class FixedLengthRecordReaderTestCase(unittest.TestCase):

    def test_model_static(self):
        print("========================= test_model_static")
        labels, images = cl.load_data_set(max_records=1)
        print("Read [%s] images and [%s] labels.", len(images), len(labels))

        model = cm.forward_propagation(input=[images[0]])

        assert(model is not None)

        session = tf.Session()
        session.run(tf.global_variables_initializer())

        model_result = session.run(model)

        print("Convolution result: ", model_result)
        print("Result with shape: ", model_result.shape)
        print("========================= test_model_static")
        # print("Records: ", len(records))
        # print("labels: ", len(labels))
        # print("Item0: ", records[0])


if __name__ == '__main__':
    unittest.main()
