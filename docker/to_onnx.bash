#!/bin/bash

cat > torch2onnx.py <<EOF
from fire import Fire
from DocAligner.model import main_docaligner_torch2onnx

if __name__ == '__main__':
    Fire(main_docaligner_torch2onnx)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --shm-size=64g \
    --ipc=host --net=host \
    -v $PWD/DocAligner:/code/DocAligner \
    -v $PWD/torch2onnx.py:/code/torch2onnx.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python torch2onnx.py --cfg_name $1
