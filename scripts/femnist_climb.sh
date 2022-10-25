#!/usr/bin/env bash
set -e
source ../venv/bin/activate

python ../src/main.py --round=200 --clients=100 --batch_size=50 --n_procs=3 --distribution='imbalance' --dataset='femnist' --local_epoch=5 --weight_decay=1e-04 --lr=0.01 --beta=0 --n_minority_classes=0 --climb=1 --rho=0;

deactivate