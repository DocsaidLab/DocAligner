#!/bin/bash

docker run \
    -u $(id -u):$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocAligned:/code/DocAligned \
    -it --rm doc_align_train python -c "from DocAligned.model.trainer import main; main('$1')"
