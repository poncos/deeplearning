#!/bin/bash

# Example usage:
## sh run_inference.sh `pwd`/../resources/test/images/cat3.jpg

PYTHON_CMD=python3.5

USER_DIR=`pwd`
SCRIPT_DIR=$(dirname "$0")
BASE_DIR="${USER_DIR}/${SCRIPT_DIR}"

export PYTHONPATH="${BASE_DIR}/..:${BASE_DIR}/../../../utils"

echo "PYTHONPATH=${PYTHONPATH}"
(cd ${BASE_DIR}/..; ${PYTHON_CMD} cifar10/networking/cifar10_predict.py $@)
