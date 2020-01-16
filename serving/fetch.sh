#!/bin/sh

set -e


BASE_DIR=$(dirname $0)
RESNET_NAME=resnet_v2_fp32_savedmodel_NHWC.tar.gz
RESNET_URL=http://download.tensorflow.org/models/official/20181001_resnet/savedmodels/${RESNET_NAME}
RESNET_UNPACK_DIR=resnet_v2_fp32_savedmodel_NHWC

(cd $BASE_DIR
test -f ${RESNET_NAME} || {
    echo "Cannot find resnet model, downloading" >&2
    curl -LO ${RESNET_URL}
}

test -d ${RESNET_UNPACK_DIR} || {
    echo "Unpack dir does not exist, unpacking" >&2
    tar xvf ${RESNET_NAME}
}

test -e models/1/saved_model.pb || {
    echo "Cannot find unpacked model, creating" >&2
    mkdir -p models/1
    cp -r $(pwd)/resnet_v2_fp32_savedmodel_NHWC/1538687283/* $(pwd)/models/1/
}

test -f example.jpg || {
    echo "Example image does not exist, fetching" >&2
    curl -L -o example.jpg https://upload.wikimedia.org/wikipedia/commons/4/4c/Push_van_cat.jpg
}
)
