import os
import unittest

from cifar10.dataloader.fixed_length_record_reader import FixedLengthRecordReader

TEST_DATA_FILENAME_1 = os.path.join(os.path.dirname(__file__), '../../resources/test/data_batch_5.bin')
TEST_DATA_FILENAME_2 = os.path.join(os.path.dirname(__file__), '../../resources/test/data_batch_4.bin')


class FixedLengthRecordReaderTestCase(unittest.TestCase):

    def setUp(self):
        filenames = [TEST_DATA_FILENAME_1, TEST_DATA_FILENAME_2]
        self.obj_to_test = FixedLengthRecordReader(filenames, 3073)

        self.assertTrue(20000 == self.obj_to_test.count())

    def test_load_c10_data_one_record(self):

        record = self.obj_to_test.read()
        print(record)
        self.assertTrue(record is not None)

    def test_load_c10_data_all_records(self):

        self.obj_to_test.reset()
        record = self.obj_to_test.read()
        counter = 0
        while record is not None:
            counter += 1
            if counter % 100 == 0:
                print("Read record number ", counter)
                print("Record: ", record)

            sequence, record, source = self.obj_to_test.read()

        print("Number of records read ", counter)
        self.assertTrue(20000 == counter)


if __name__ == '__main__':
    unittest.main()
