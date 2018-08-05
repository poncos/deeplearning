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
CIFAR_10_WIDTH = 32
CIFAR_10_HEIGHT = 32
CIFAR_10_CHANNELS = 3
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
                  cifar10_input_file = DEFAULT_CIFAR_INPUT_FILE,
                  max_records=-1):

    if cifar10_format == TAR_GZ_FORMAT:
        fname = os.path.join(cifar10_dir, cifar10_input_file)
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path='/tmp')
        tar.close()

    # TODO hardcoded paths
    cifar10_files = [os.path.join('/tmp/cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in range(1, 6)]

    reader = FixedLengthRecordReader(cifar10_files, CIFAR_10_RECORD_SIZE)

    # TODO hardcoded length
    cifar10_image_list = [None]*50000
    cifar10_label_list = [None]*50000
    record_number = 0

    sequence, record, source = reader.read()
    print("Read ", len(record), " records")

    while record is not None:
        if max_records != -1 and record_number == max_records:
            break

        # print("record number: ", sequence, " from source: ", source)
        # record_obj = CIFAR10Record()
        # record_obj.source = source
        # record_obj.sequence = sequence
        record_label = record[0]
        record_image = record[1:3073].astype(np.float32, copy=False)

        r = record_image[0:1024].reshape((CIFAR_10_WIDTH, CIFAR_10_HEIGHT))
        b = record_image[1024:2048].reshape((CIFAR_10_WIDTH, CIFAR_10_HEIGHT))
        g = record_image[2048:3072].reshape((CIFAR_10_WIDTH, CIFAR_10_HEIGHT))
        # TODO this line adds one second to the time needed to load the entire cifar10 data set
        # record_obj.payload = np.dstack((r, g, b))
        # TODO this tolist takes long time in the overall data-set loading
        image = np.dstack((r, g, b)).tolist()

        cifar10_label_list[record_number] = record_label
        cifar10_image_list[record_number] = image

        # cifar10_record_list[record_number] = record_obj
        sequence, record, source = reader.read()
        record_number += 1

    return cifar10_label_list, cifar10_image_list


def reshape_images():
    pass

