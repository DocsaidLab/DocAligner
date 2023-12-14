#!/bin/bash

cat > benchmark.py <<EOF
from fire import Fire
from DocAligned.benchmark import benchmark_smartdoc
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --shm-size=64g \
    --ipc=host --net=host \
    -v $PWD/DocAligned:/code/DocAligned \
    -v $PWD/benchmark.py:/code/benchmark.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python benchmark.py
