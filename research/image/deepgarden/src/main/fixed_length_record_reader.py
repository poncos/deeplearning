# Copyright 2018 Esteban Collado.
#
# Licensed under the MIT License
import numpy as np


class FixedLengthRecordReader:

    def __init__(self, file_names, record_bytes):
        self.__record_bytes = record_bytes
        self.__file_names = file_names

        self.__current_file = 0
        self.__buffer = None
        self.__offset = 0
        self.__sequence = -1

    def reset_offset(self):
        self.__offset = 0

    def reset(self):
        self.__current_file = 0
        self.__buffer = None
        self.__offset = 0

    def read(self):

        # print("Reading record from memory starting in position ", self.__offset)
        if self.__buffer is None and self.__current_file < len(self.__file_names):
            print("Reading file ", len(self.__file_names), " - ", self.__current_file," - ", self.__file_names[self.__current_file])
            self.__buffer = np.fromfile(self.__file_names[self.__current_file],
                                        dtype=np.ubyte, count=-1)
            self.__current_file += 1
            self.__offset = 0
            self.__sequence = -1

        if self.__buffer is not None and self.__offset <= (len(self.__buffer)-self.__record_bytes):
            # print("Buffer ", len(self.__buffer), " ", self.__buffer)
            # print("Reading record with offset: ", self.__offset)
            # WARNING this function will return a reference to the same data
            record = self.__buffer[self.__offset:self.__offset + self.__record_bytes]
            # print("Record: ", record)
            self.__offset += self.__record_bytes
            self.__sequence += 1

            if self.__offset >= len(self.__buffer):
                self.__buffer = None

            return self.__sequence, record, self.__file_names[self.__current_file-1]

        return None, None, None

