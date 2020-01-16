#!/bin/sh

set -e

BASE_DIR=$(dirname $0)
NAME=tensorflow-serving

(cd $BASE_DIR

# if the image is already running then kill it
if [[ $(docker ps -a -f name=${NAME} -q | wc -l) -ne 0 ]]; then
    docker rm -f ${NAME} >/dev/null
fi

docker run -d -p 9000:8500 -p 9001:8501 -v $(pwd)/models:/models/resnet -e MODEL_NAME=resnet -t --name ${NAME} tensorflow/serving:latest >/dev/null

)
