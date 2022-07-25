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


def random_zero_one(theta: list):
    rand = random.random()
    return rand < theta

class Client:
    def __init__(self, nodeID, node_indices, prev_grads: dict, args):
        self.nodeID = nodeID
        self.node_indices = node_indices
        self.args = args
        self.optimizer_params = None
        self.scheduler_params = None
        self.prev_grads = prev_grads
        self.lamb = 1.0

    def train(self, device, lr, model: nn.Module, total_train_dataset: CelebADataset, optimizer):
        model.to(device)
        old_param = copy.deepcopy(model.state_dict())
        labels = total_train_dataset.labels[self.node_indices]
        pv = total_train_dataset.protected_variables[self.node_indices]

        if len(pv.unique()) > 1:
            self.pv_uniq = 10
        else:
            self.pv_uniq = pv.unique()[0].item()

        labels = total_train_dataset.labels[self.node_indices]

        n_samples = [(labels == i).count_nonzero() for i in range(10)]
        sorted_list = copy.deepcopy(n_samples)
        sorted_list.sort()
        subset_indices = [i for i in range(len(labels)) if random_zero_one((sum(n_samples) - n_samples[labels[i]]) / n_samples[labels[i]])]
        
        n_samples = [(labels[subset_indices] == i).count_nonzero() for i in range(10)]
        if self.args.focal_loss != 0:
            criterion = FocalLoss(gamma=self.args.focal_loss)
        else:
            criterion = BalancedSoftmax(n_samples)
            # criterion = nn.CrossEntropyLoss()

        train_loader = DataLoader(Subset(total_train_dataset, self.node_indices[subset_indices]), self.args.batch_size, shuffle=True, num_workers=0)
        
        # train_loader = DataLoader(Subset(total_train_dataset, self.node_indices), self.args.batch_size, shuffle=True, num_workers=0)


        if self.args.feddyn_alpha != 0:
            with torch.no_grad():
                origin = {k: v.flatten().detach().clone()
                          for k, v in model.named_parameters()}
                self.prev_grads = {k: v.detach().to(device)
                                   for k, v in self.prev_grads.items()}

        label_sum = 0
        sample_sum = 0
        for _ in range(self.args.local_epoch):
            for inputs, labels, _ in train_loader:
                model.train()
                optimizer.zero_grad()
                if self.args.dataset in ['compas', 'adult']:
                    inputs_cat, inputs_num, labels = inputs[0].to(
                        device), inputs[1].to(device), labels.to(device)
                    output = model(inputs_cat, inputs_num)
                else:
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model(inputs)
                label_sum += labels.sum()
                sample_sum += len(labels)
                loss: torch.Tensor = criterion(output, labels)   # true loss

                for k, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    curr_param = param.flatten()
                    # linear penalty
                    loss.sub_(torch.dot(curr_param, self.prev_grads[k]), alpha=self.lamb)
                    # quadratic penalty
                    loss.add_(torch.sum(torch.square(
                        torch.sub(curr_param, origin[k]))), alpha=self.lamb * self.args.feddyn_alpha / 2)

                loss.backward()
                optimizer.step()

        # update prev_grads
        with torch.no_grad():
            for k, param in model.named_parameters():
                if param.requires_grad:
                    curr_param = param.detach().flatten().clone()
                    self.prev_grads[k].sub_(torch.sub(curr_param, origin[k]), alpha=self.args.feddyn_alpha * self.lamb)
                self.prev_grads[k] = self.prev_grads[k].to(torch.device('cpu'))

        delta = {k: v.sub(old_param[k]).to(torch.device('cpu'))
                 for k, v in model.state_dict().items()}
        if self.args.feddyn_alpha != 0:
            weight = 1
        else:
            # weight = len(self.node_indices)
            weight = 1
        model.to(torch.device('cpu'))
        return delta, weight
