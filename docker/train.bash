#!/bin/bash

docker run \
    -u $(id -u):$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocAligned:/code/DocAligned \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python -c "from DocAligned.model import main_docalign_train; main_docalign_train('$1')"
