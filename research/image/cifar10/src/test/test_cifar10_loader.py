import unittest

import main.cifar10_loader as cl


class FixedLengthRecordReaderTestCase(unittest.TestCase):

    def test_load_cifar10(self):
        cifar10_record_list = cl.load_data_set()


if __name__ == '__main__':
    unittest.main()
