#!/bin/bash

PYTHON_CMD=python3.5

BASEDIR=$(dirname "$0")
(cd $BASEDIR/..; ${PYTHON_CMD} -m unittest discover cifar10.test.test_fixed_length_record_reader)
