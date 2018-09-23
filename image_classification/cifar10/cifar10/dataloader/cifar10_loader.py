# Copyright 2018 Esteban Collado.
#
# Licensed under the MIT License

from cifar10.dataloader.fixed_length_record_reader import FixedLengthRecordReader
import cifar10.constants as constants

import tarfile
import os
import numpy as np

import tempfile

CIFAR_10_RECORD_SIZE = 3073
CIFAR_10_WIDTH = 32
CIFAR_10_HEIGHT = 32


class LoaderDataSetConfig:
    cifar_data_dir = constants.DATA_DIR_PATH
    cifar_input_file = 'cifar-10-binary'
    cifar_input_file_ext = '.tar.gz'
    max_records = -1
    load_evaluate_dataset = False


DEFAULT_CONFIG = LoaderDataSetConfig


def unpack_dataset(config):
    dirpath = tempfile.mkdtemp()
    print("Unpacking the cifar10 dataset in the path %s" % dirpath)

    fname = os.path.join(config.cifar_data_dir, config.cifar_input_file + config.cifar_input_file_ext)
    tar = tarfile.open(fname, "r:gz")
    tar.extractall(path=dirpath)
    tar.close()

    if not config.load_evaluate_dataset:
        cifar10_files = [os.path.join(dirpath, 'cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in range(1, 6)]
    else:
        cifar10_files = [os.path.join(dirpath, 'cifar-10-batches-bin/test_batch.bin')]

    return cifar10_files


def load_data_set(config=DEFAULT_CONFIG):

    cifar10_files = unpack_dataset(config)
    if not cifar10_files:
        raise Exception('Dataset not available in path [%s] and file name [%s]'
                        % (config.cifar_data_dir, config.cifar_input_file + config.cifar_input_file_ext))

    reader = FixedLengthRecordReader(cifar10_files, CIFAR_10_RECORD_SIZE)
    num_records = reader.count()

    cifar10_image_list = [None]*(num_records if config.max_records == -1 else config.max_records)
    cifar10_label_list = [None]*(num_records if config.max_records == -1 else config.max_records)
    record_number = 0

    sequence, record, source = reader.read()
    print("Read ", len(record), " records")

    while record is not None:
        if config.max_records != -1 and record_number == config.max_records:
            break

        record_label = record[0]
        record_image = record[1:3073].astype(np.float32, copy=False)

        r = record_image[0:1024].reshape((CIFAR_10_WIDTH, CIFAR_10_HEIGHT))
        b = record_image[1024:2048].reshape((CIFAR_10_WIDTH, CIFAR_10_HEIGHT))
        g = record_image[2048:3072].reshape((CIFAR_10_WIDTH, CIFAR_10_HEIGHT))
        # TODO this tolist takes long time in the overall data-set loading
        image = np.dstack((r, g, b)).tolist()

        cifar10_label_list[record_number] = record_label
        cifar10_image_list[record_number] = image

        sequence, record, source = reader.read()
        record_number += 1

    return cifar10_label_list, cifar10_image_list


