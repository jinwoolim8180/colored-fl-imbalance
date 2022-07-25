import torch
import time
import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../models')
import arguments
import pickle
import torch.nn as nn

import models
from torch.utils.data import DataLoader, Subset
import dataset
from metrics import evaluate
from focalloss import FocalLoss
from gceloss import GCELoss


if __name__ == "__main__":
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
        
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False

    os.environ["OMP_NUM_THREADS"] = "1"

    parser = arguments.parser()
    parser.add_argument('--train_group', default=0, type=int)
    args = parser.parse_args()
    
    print("> Setting:", args)

    lr = args.lr

    train_dataset = dataset.get_dataset(args, 'train')
    test_dataset = dataset.get_dataset(args, 'test')
    group = (train_dataset.protected_variables == args.train_group).nonzero().squeeze()
    train_loader = DataLoader(Subset(train_dataset, group), batch_size=args.batch_size, num_workers=8, shuffle=True)

    train_model = models.get_model(args)
    test_model = models.get_model(args)

    device = devices[0]
    train_model.to(device)
    train_model = nn.DataParallel(train_model)
    train_model.train()

    n_samples = [(train_dataset.labels[group] == 0).count_nonzero(), (train_dataset.labels[group] == 1).count_nonzero()]
    print(n_samples)
    if args.focal_loss != 0:
        weights = [1 for _ in n_samples]
        # weights = [(1/n) for n in n_samples]
        # weights = [w/sum(weights) for w in weights]
        criterion = FocalLoss(gamma=args.focal_loss, alpha=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    acc_list = []
    fair_list = []
    optimizer = torch.optim.Adam(train_model.parameters(), lr=lr)

    for roundIdx in range(1, args.round+1):
        cur_time = time.time()
        print(f"Round {roundIdx}", end=', ')
        for _ in range(args.local_epoch):
            for inputs, labels, indexes in train_loader:
                optimizer.zero_grad()
                if args.dataset in ['compas', 'adult']:
                    inputs_cat, inputs_num, labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
                    output = train_model(inputs_cat, inputs_num)
                else:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = train_model(inputs)
                
                loss: torch.Tensor = criterion(output, labels)   # true loss
                loss.backward()
                optimizer.step()
        if roundIdx % 20 == 0:
            lr *= args.lr_decay
            optimizer = torch.optim.Adam(train_model.parameters(), lr=lr)
        test_model.load_state_dict(train_model.module.state_dict())
        acc, fair = evaluate(test_model, test_dataset, device, args)
        acc_list.append(acc)
        fair_list.append(fair)

        print(f"Elapsed Time : {(time.time()-cur_time):.1f}")
        print("Round: {} / Accuracy: {} / Fairness: {}".format(roundIdx, acc, fair))

    file_name = '../save/results/{}/centralized_R[{}]LR[{}]B[{}]G[{}]F[{}].pkl'.\
        format(args.dataset, args.round, args.lr, args.bias_level, args.train_group, args.focal_loss)

    with open(file_name, 'wb') as f:
        pickle.dump({'acc': acc_list, 'fair': fair_list}, f)

    torch.save(test_model.state_dict(), '../save/models/{}_centralized.pt'.format(args.dataset))