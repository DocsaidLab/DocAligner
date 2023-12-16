#!/bin/bash

cat > trainer.py <<EOF
from fire import Fire
from DocAligner.model import main_docaligner_train

if __name__ == '__main__':
    Fire(main_docaligner_train)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/DocAligner:/code/DocAligner \
    -v $PWD/trainer.py:/code/trainer.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python trainer.py --cfg_name $1

# 說明：`/data/Dataset` 是放外部資料的地方，請自行修改成自己的路徑。
