# Copyright 2018 Esteban Collado.
#
# Licensed under the MIT License
import numpy as np
import os


class FixedLengthRecordReader:

    def __init__(self, file_names, record_bytes):
        self.record_bytes = record_bytes
        self.file_names = file_names

        self.current_file = 0
        self.buffer = None
        self.offset = 0
        self.sequence = -1
        self.total_bytes = -1

    def reset_offset(self):
        self.offset = 0

    def reset(self):
        self.current_file = 0
        self.buffer = None
        self.offset = 0

    def read(self):

        # print("Reading record from memory starting in position ", self.__offset)
        if self.buffer is None and self.current_file < len(self.file_names):
            self.buffer = np.fromfile(self.file_names[self.current_file],
                                        dtype=np.ubyte, count=-1)
            self.current_file += 1
            self.offset = 0
            self.sequence = -1

        if self.buffer is not None and self.offset <= (len(self.buffer)-self.record_bytes):
            # WARNING this function will return a reference to the same data
            record = self.buffer[self.offset:self.offset + self.record_bytes]
            self.offset += self.record_bytes
            self.sequence += 1

            if self.offset >= len(self.buffer):
                self.buffer = None

            return self.sequence, record, self.file_names[self.current_file-1]

        return None, None, None

    def count(self):
        if self.total_bytes == -1:
            self.total_bytes = self.__count_bytes()

        return self.total_bytes//self.record_bytes

    def __count_bytes(self):
        total_size = 0
        for file in self.file_names:
            total_size += os.path.getsize(file)

        return total_size
