#!/usr/bin/env bash
set -e
source ../venv/bin/activate

python ../src/main.py --round=200 --fraction=0.1 --clients=50 --batch_size=50 --n_procs=3 --distribution='imbalance' --dataset='femnist' --local_epoch=5 --weight_decay=1e-04 --lr=0.1 --beta=0 --n_minority_classes=0 --climb=1 --rho=0;

deactivate