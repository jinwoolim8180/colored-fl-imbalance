from statistics import median, stdev
from typing import Dict
from numpy import count_nonzero
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.utils.data import DataLoader, Subset
from dataset.celeba import CelebADataset
from focalloss import *
from bsloss import *
import random
import numpy as np
from torch.utils.data.sampler import Sampler, WeightedRandomSampler

class Client:
    def __init__(self, nodeID, node_indices, prev_grads: dict, args):
        self.nodeID = nodeID
        self.node_indices = node_indices
        self.args = args
        self.prev_grads = prev_grads
        self.lamb = args.feddyn_lambda

    def train(self, device, lr, model: nn.Module, total_train_dataset: CelebADataset, optimizer):
        model.to(device)
        old_param = copy.deepcopy(model.state_dict())
        labels = total_train_dataset.labels[self.node_indices]

        if self.args.loss == 'ce':
            criterion = nn.CrossEntropyLoss()
        
        elif self.args.loss == 'fl':
            criterion = FocalLoss(gamma=self.args.focal_loss)

        elif self.args.loss == 'bs':
            n_samples = [(labels == i).count_nonzero() for i in labels.unique()]
            criterion = BalancedSoftmax(n_samples)

        train_loader = DataLoader(Subset(total_train_dataset, self.node_indices), self.args.batch_size, shuffle=True, num_workers=0)

        with torch.no_grad():
            origin = {k: v.flatten().detach().clone() for k, v in model.named_parameters()}
            self.prev_grads = {k: v.detach().to(device) for k, v in self.prev_grads.items()}

        for _ in range(self.args.local_epoch):
            for inputs, labels, _ in train_loader:
                model.train()
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                loss: torch.Tensor = criterion(output, labels)   # true loss
                loss.backward()
                optimizer.step()

        # update prev_grads
        with torch.no_grad():
            for k, param in model.named_parameters():
                if param.requires_grad:
                    curr_param = param.detach().flatten().clone()
                    self.prev_grads[k].sub_(torch.sub(curr_param, origin[k]), alpha=self.args.feddyn_alpha * self.lamb)
                self.prev_grads[k] = self.prev_grads[k].to(torch.device('cpu'))

        delta = {k: v.sub(old_param[k]).to(torch.device('cpu')) for k, v in model.state_dict().items()}
        if self.args.weighted_avg == 0:
            weight = len(self.node_indices)
        else:
            weight = 1
        model.to(torch.device('cpu'))
        return delta, weight
