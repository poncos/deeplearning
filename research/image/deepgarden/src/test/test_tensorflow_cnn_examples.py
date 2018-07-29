import unittest

import main.cifar10_loader as cl
import main.tensor_flow_cnn_examples as tfe


class FixedLengthRecordReaderTestCase(unittest.TestCase):

    def test_static_conv2d(self):
        conv_result = tfe.conv_example_static()

        print("Convolution result: ", conv_result)
        print("Result with shape: ", conv_result.shape)
        print(conv_result[0, 0, 0])
        print(conv_result[0, 1, 0])
        print(conv_result[0, 2, 0])

    def test_conv2d_image(self):
        print("test_conv2d_image")
        cifar10_record_list = cl.load_data_set()

        image_data = cifar10_record_list[0].payload.tolist()
        print(image_data)

        conv_result = tfe.conv_example_static(input=[image_data], filter_var_name='image_filter1')

        print("Convolution result: ", conv_result)
        print("Result with shape: ", conv_result.shape)
        print(conv_result[0, 0, 0])
        print(conv_result[0, 1, 0])
        print(conv_result[0, 2, 0])




if __name__ == '__main__':
    unittest.main()
