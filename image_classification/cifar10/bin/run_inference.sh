#!/bin/bash

PYTHON_CMD=python3.5

USER_DIR=`pwd`
SCRIPT_DIR=$(dirname "$0")
echo "USER_DIR: ${USER_DIR}"
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
BASE_DIR="${USER_DIR}/${SCRIPT_DIR}"
echo "BASEDIR: ${BASE_DIR}"

export PYTHONPATH="${BASE_DIR}/..:${BASE_DIR}/../../../utils"

echo "PYTHONPATH=${PYTHONPATH}"
(cd ${BASE_DIR}/..; ${PYTHON_CMD} cifar10/networking/cifar10_predict.py $@)
