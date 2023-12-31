#!/bin/bash

cat > benchmark.py <<EOF
from fire import Fire

def main(task, model_type, model_cfg):
    if task == 'smartdoc':
        from DocAligner.benchmark import benchmark_smartdoc
        benchmark_smartdoc.main(model_type, model_cfg)
    elif task == 'idcard':
        from DocAligner.benchmark import benchmark_idcard
        benchmark_idcard.main(model_type, model_cfg)
    elif task == 'passport':
        from DocAligner.benchmark import benchmark_passport
        benchmark_passport.main(model_type, model_cfg)

    else:
        print("Invalid task. Please specify 'smartdoc' or 'idcard'.")

if __name__ == "__main__":
    Fire(main)
EOF

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --shm-size=64g \
    --ipc=host --net=host \
    -v $PWD/DocAligner:/code/DocAligner \
    -v $PWD/benchmark.py:/code/benchmark.py \
    -v /data/Dataset:/data/Dataset \
    -it --rm doc_align_train python benchmark.py $1 $2 $3
