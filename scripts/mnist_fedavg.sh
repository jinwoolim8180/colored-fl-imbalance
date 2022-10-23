#!/usr/bin/env bash
set -e
source ../venv/bin/activate

python ../src/main.py --round=200 --clients=100 --batch_size=50 --n_procs=5 --distribution='imbalance' --dataset='mnist' --local_epoch=5 --weight_decay=0 --lr=0.01 --beta=0.2 --n_minority_classes=3 --climb=0 --rho=10;

deactivate