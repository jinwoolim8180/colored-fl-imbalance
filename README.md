# Federated Learning with Class Imbalance

## Dataset

### MNIST
Digit Dataset. 10 Classes

### CIFAR10
CIFAR 10. 10 Classes

### FEMNIST
will be added soon...

## Install Requirements
Download Requirements inside your Virtual Env.
Run the command below under this folder. (not in src/)
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Open Linux Screen
Use screen to close or open the terminal anytime.
#### build screen
```
screen -S fl
```
#### reload screen
```
screen -r fl
```
#### view screen list
```
screen -list
```
#### detach screen
```
screen -d fl (outside the screen)
or simply press "Ctrl-A, D" inside screen
```

## How to Run
Run commands below under src/ folder.
### MNIST
```
python main.py --distribution='imbalance' --dataset='mnist' --local_epoch=5 --weight_decay=0 --
lr=0.01 --beta=0.2 --n_minority_classes=0
```

### CIFAR10
```
python main.py --distribution='imbalance' --dataset='cifar10' --local_epoch=5 --weight_decay=5e-4 --
lr=0.01 --beta=0.2 --n_minority_classes=0
```

## Argument Explained
- clients: 10, 50, 100
    - how many clients are paritipating
- dataset: 'mnist', 'cifar10'
    - What dataset to train
- distribution: 'iid', 'imbalance', 'dirichlet'
    - How total dataset is distributed into clients
- n_minority_classes: 0,1,2,...
    - how many classes are minority, minority classes only have 1/rho amount of data
    - "0~n_minority_classes-1" classes are minority classes
- rho: 1, 10, ...
    - minority classes only have 1/rho amount of data
- loss: 'ce', 'bs', 'fl'
    - ce: Cross Entropy Loss (Basic)
    - bs: Balanced Softmax (class imbalance)
    - fl: Focal Loss (class imbalance)
- local_epoch: 1,2,3,...
    - How many epochs each client trains locally per roun
- batch_size: 64, 128, 256,...
    - batch_size per update
- checkpoint_round: 50, ...
    - experiment state is saved at checkpoint_round
- resume_checkpoint: 0,1
    - 0: start from beginning, 1: start from checkpoint
- weighted_avg: 0, 1
    - 0: same weight for every client, 1: sample size as client weight
- lr: 0.01, 0.001, ...
    - Learning rate for SGD optimizer
- weight_decay: 0, 0.0001, 0.00001, ...
    - Weight Decay for SGD optimizer
- beta: 0, 0.1, 0.2, ..., 1, 100
    - if distribution is 'imbalance' (0~1) : beta amount of data is distributed iid, (1-beta) is distributed in order
        - beta = 0, almost only single class per client, beta = 1, iid
    - if distribution is 'dirichlet' (0~) : beta is used in dirichlet distribution
        - smaller beta: non iid, higher beta: iid

## What you need to do
### Common Tasks (for both topics)
1. Fixed Global Imbalance (rho and n_minority_classes)
2. Change Local Imbalance (beta)
3. Compare the result(minority, majority class accuracy) on different local imbalances
4. Repeat (1,2,3) after changing global imbalance
- Observe that under same global imbalance, local imbalance harms the minority class more than majority class

### Additional Tasks for each topics
Run additional tasks below for each topics.
Not ready yet. Will be implemented later.

### How local imbalance impact global imbalance
1. Use CLIMB and Balanced Softmax and Focal Loss and view the result.

### How handling class imbalance helps client performance fairness
1. Use Balanced Softmax and Focal Loss on FEMNIST Dataset and view the result.


## What will be added
1. FEMNIST Dataset
2. CLIMB
