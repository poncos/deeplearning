# Copyright 2018 Esteban Collado.
#
# Licensed under the MIT License

from main.fixed_length_record_reader import FixedLengthRecordReader
import tarfile
import os
import numpy as np

DEFAULT_CIFAR_DATA_DIR = '/data/deeplearning/input'
DEFAULT_CIFAR_INPUT_FILE = 'cifar-10-binary.tar.gz'

TAR_GZ_FORMAT = 'tar.gz'

CIFAR_10_RECORD_SIZE = 3073
CIFAR_10_LABELS = []

class CIFAR10Record:

    def __init__(self):
        self.height = 32
        self.width = 32
        self.depth = 3
        self.source = "dummy"
        self.sequence = -1
        self.label = -1


def unpack_cifar_file():
    pass


def load_data_set(cifar10_dir = DEFAULT_CIFAR_DATA_DIR,
                  cifar10_format = TAR_GZ_FORMAT,
                  cifar10_input_file = DEFAULT_CIFAR_INPUT_FILE):

    if cifar10_format == TAR_GZ_FORMAT:
        fname = os.path.join(cifar10_dir, cifar10_input_file)
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path='/tmp')
        tar.close()

    cifar10_files = [os.path.join('/tmp/cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in range(1, 6)]

    reader = FixedLengthRecordReader(cifar10_files, CIFAR_10_RECORD_SIZE)

    cifar10_record_list = np.array([None]*50000)
    record_number = 0

    sequence, record, source = reader.read()
    while record is not None:
        # print("record number: ", sequence, " from source: ", source)
        record_obj = CIFAR10Record()
        record_obj.source = source
        record_obj.sequence = sequence
        record_obj.label = record[0]
        record_image = record[1:3073].astype(np.float32, copy=False)

        r = record_image[0:1024].reshape((record_obj.width, record_obj.height))
        b = record_image[1024:2048].reshape((record_obj.width, record_obj.height))
        g = record_image[2048:3072].reshape((record_obj.width, record_obj.height))
        # TODO this line adds one second to the time needed to load the entire cifar10 data set
        record_obj.payload = np.dstack((r, g, b))

        cifar10_record_list[record_number] = record_obj

        sequence, record, source = reader.read()
        record_number += 1

    return cifar10_record_list


def reshape_images():
    pass

