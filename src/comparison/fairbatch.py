from torch.utils.data.sampler import Sampler
import torch
import itertools
import random
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

doPrint = False

class FairBatch(Sampler):
    def __init__(self, model, dataset, batch_size, device, alpha, fairness_type='eqodds'):
        super(FairBatch, self).__init__(data_source=None)
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.fairness_type = fairness_type
        self.device = device
        self.alpha = alpha
        self.batch_num = int(len(self.dataset) / self.batch_size)

        self.z_item = list(set(dataset.protected_variables.tolist()))
        self.y_item = list(set(dataset.labels.tolist()))
        self.yz_tuple = list(itertools.product(self.y_item, self.z_item))

        self.z_index: dict[any, torch.Tensor] = {}
        self.y_index: dict[any, torch.Tensor] = {}
        self.yz_index: dict[any, torch.Tensor] = {}
        for z in self.z_item:
            self.z_index[z] = (dataset.protected_variables == z).nonzero().squeeze()
        
        for y in self.y_item:
            self.y_index[y] = (dataset.labels == y).nonzero().squeeze()
        
        for yz in self.yz_tuple:
            yz_mask = (dataset.labels == yz[0]) & (dataset.protected_variables == yz[1])
            self.yz_index[yz] = yz_mask.nonzero().squeeze()
        
        self.S = {}
        N = len(dataset.protected_variables)
        for yz in self.yz_tuple:
            self.S[yz] = self.batch_size * len(self.yz_index[yz]) / N
        
        self.ld = {}
        for y in self.y_item:
            self.ld[y] = self.S[y,1] / (self.S[y,1] + self.S[y,0])
            # self.ld[y] = 1/2

    def __iter__(self):
        self.adjust_lambda()
        each_size = {}

        for y in self.y_item:
            each_size[(y, 1)] = round(self.ld[y] * (self.S[(y, 0)] + self.S[(y, 1)]))
            each_size[(y, 0)] = round((1 - self.ld[y]) * (self.S[(y, 0)] + self.S[(y, 1)]))
        sort_index = []

        for yz in self.yz_tuple:
            sort_index.append(self.select_batch_replacement(each_size[yz], self.yz_index[yz], self.batch_num))
        for i in range(self.batch_num):
            key_in_fairbatch = np.concatenate([s_i[i].copy() for s_i in sort_index])
            random.shuffle(key_in_fairbatch)
            yield key_in_fairbatch

    def __len__(self):
        return len(self.dataset)


    def select_batch_replacement(self, batch_size, full_index, batch_num):

        tmp_index = full_index.detach().cpu().numpy().copy()
        random.shuffle(tmp_index)

        start_idx = 0
        select_index = []

        tmp_index = full_index.detach().cpu().numpy().copy()
        random.shuffle(tmp_index)
        
        for _ in range(batch_num):
            if start_idx + batch_size > len(full_index):
                select_index.append(np.concatenate((tmp_index[start_idx:], tmp_index[ : batch_size - (len(full_index)-start_idx)])))
                start_idx = len(full_index)-start_idx
            else:
                select_index.append(tmp_index[start_idx:start_idx + batch_size])
                start_idx += batch_size
            
        return select_index

    def adjust_lambda(self):
        yz_losses = {}
        model = self.model
        model.eval()
        for yz in self.yz_tuple:
            criterion = nn.CrossEntropyLoss()
            test_loader = DataLoader(Subset(self.dataset, self.yz_index[yz]), batch_size=self.batch_size*2, num_workers=16)
            total_loss = 0
            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = model(inputs)
                    loss: torch.Tensor = criterion(outputs, labels)   # true loss
                    total_loss += loss * len(labels)
            yz_losses[yz] = total_loss / len(self.yz_index[yz])

        yhat_yz = {}
        for yz in self.yz_tuple:
            yhat_yz[yz] = float(yz_losses[yz])
        
        y1_diff = abs(yhat_yz[(1, 1)] - yhat_yz[(1, 0)])
        y0_diff = abs(yhat_yz[(0, 1)] - yhat_yz[(0, 0)])

        if y1_diff > y0_diff:
            if yhat_yz[(1, 1)] > yhat_yz[(1, 0)]:
                self.ld[1] += self.alpha
            else:
                self.ld[1] -= self.alpha
        else:
            if yhat_yz[(0, 1)] > yhat_yz[(0, 0)]:
                self.ld[0] += self.alpha
            else:
                self.ld[0] -= self.alpha
                
        for i in self.ld.keys():
            if self.ld[i] < 0:
                self.ld[i] = 0
            elif self.ld[i] > 1:
                self.ld[i] = 1

if __name__ == '__main__':
    import numpy as np
    import pickle
    import random
    import itertools

    import torch.nn as nn
    from torch.utils.data.sampler import Sampler
    import torch
    import sys
    import os
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
    sys.path.append('../models')
    import models
    from torch.utils.data import DataLoader
    import dataset
    from metrics import evaluate
    import arguments
    import time


    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
        
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False

    args = arguments.parser()
    
    print("> Setting:", args)

    lr = args.lr

    train_dataset = dataset.get_dataset(args, 'train')
    test_dataset = dataset.get_dataset(args, 'test')

    train_model = models.get_model(args)
    test_model = models.get_model(args)

    device = devices[0]
    train_model.to(device)
    # train_model = nn.DataParallel(train_model)
    alpha = 0.1

    sampler = FairBatch(train_model, train_dataset, args.batch_size, device, alpha)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=32)

    criterion = nn.CrossEntropyLoss()

    acc_list = []
    fair_list = []

    for roundIdx in range(1, args.round+1):
        cur_time = time.time()
        print(f"Round {roundIdx}", end=', ')
        optimizer = torch.optim.SGD(train_model.parameters(), lr=lr, weight_decay=1e-3)
        for _ in range(args.local_epoch):
            for inputs, labels, _ in train_loader:
                train_model.train()
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
        if roundIdx % 5 == 0:
            lr *= args.lr_decay
        test_model.load_state_dict(train_model.state_dict())
        acc, fair = evaluate(test_model, test_dataset, device, args)
        acc_list.append(acc)
        fair_list.append(fair)

        print(f"Elapsed Time : {(time.time()-cur_time):.1f}")
        print("Round: {} / Accuracy: {} / Fairness: {}".format(roundIdx, acc, fair))

    file_name = '../save/results/{}/fairbatch_R[{}]LR[{}]E[{}]LD[{}]B[{}].pkl'.\
        format(args.dataset, args.round, args.lr, args.local_epoch, args.lr_decay, args.bias_level)

    with open(file_name, 'wb') as f:
        pickle.dump({'acc': acc_list, 'fair': fair_list}, f)
