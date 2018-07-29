import os
import unittest

from main.fixed_length_record_reader import FixedLengthRecordReader

TEST_DATA_FILENAME_1 = os.path.join(os.path.dirname(__file__), '../../resources/test/data_batch_5.bin')
TEST_DATA_FILENAME_2 = os.path.join(os.path.dirname(__file__), '../../resources/test/data_batch_4.bin')


class FixedLengthRecordReaderTestCase(unittest.TestCase):

    def setUp(self):
        filenames = [TEST_DATA_FILENAME_1, TEST_DATA_FILENAME_2]
        self.__obj_to_test = FixedLengthRecordReader(filenames, 3073)
        self.__counter = 0

    def test_load_c10_data_one_record(self):

        record = self.__obj_to_test.read()
        print(record)
        self.assertTrue(record is not None)

    def test_load_c10_data_all_records(self):

        self.__obj_to_test.reset()
        record = self.__obj_to_test.read()

        while record is not None:
            self.__counter += 1
            if self.__counter % 100 == 0:
                print("Read record number ", self.__counter)
                print("Record: ", record)

            record = self.__obj_to_test.read()

        print("Number of records read ", self.__counter)
        self.assertTrue(20000 == self.__counter)



if __name__ == '__main__':
    unittest.main()
