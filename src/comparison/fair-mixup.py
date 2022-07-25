import os
import argparse
import pandas as pd
import numpy as np
from numpy.random import beta
import pickle
from pprint import pprint

import torch
import torch.nn as nn
from torch import optim

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
import time


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ResNet_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.resnet = models.get_model(args)
        self.resnet.fc = Identity()
        self.resnet.avgpool = Identity()

    def forward(self, x):
        outputs = self.resnet(x)
        # print(outputs.shape)
        return outputs.view(-1, 512, 8, 8)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        x = self.avg(x).view(-1, 512)
        x = self.fc1(x)
        x = self.relu(x)
        outputs = self.fc2(x)
        return outputs

def fit_model(model_linear, model, dataloaders, mode='mixup', lam=10): 

    len_dataloader = min([min(len(dataloader[0]), len(dataloader[1])) for dataloader in dataloaders])
    data_iters = [[iter(dataloader[0]), iter(dataloader[1])] for dataloader in dataloaders]
    model.train()    
    model_linear.train()

    for it in range(len_dataloader - 1):
        inputs = [[data_iter[0].next()[0].cuda(), data_iter[1].next()[0].cuda()] for data_iter in data_iters]
        inputs_by_labels = [[input[i] for input in inputs] for i in range(2)]
        inputs = torch.cat(inputs_by_labels[0] + inputs_by_labels[1], 0)
        target = torch.tensor([0 if i < len(inputs) / 2 else 1 for i in range(len(inputs))]).cuda()
        feat = model(inputs)
        ops = model_linear(feat)

        loss_sup = criterion(ops.squeeze(), target)

        if mode == 'GapReg':
            loss_gap = 0
            for g in range(2):
                inputs_by_label = inputs_by_labels[g]
                ops = [model_linear(model(input)) for input in inputs_by_label]

                loss_gap += torch.abs(ops[0].mean() - ops[1].mean())

            loss = loss_sup + lam*loss_gap

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Gap: {:.8f} ".format(loss_sup, loss_gap))

        elif mode == 'mixup':
            alpha = 1
            loss_grad = 0
            for g in range(2):
                gamma = beta(alpha, alpha)
                inputs_by_label = inputs_by_labels[g]
                inputs_mix = inputs_by_label[0] * gamma + inputs_by_label[1] * (1 - gamma)
                inputs_mix = inputs_mix.requires_grad_(True)

                feat = model(inputs_mix)
                ops = model_linear(feat)
                ops = ops.sum()

                gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
                x_d = (inputs_by_label[1] - inputs_by_label[0]).view(inputs_mix.shape[0], -1)
                grad_inn = (gradx * x_d).sum(1).view(-1)
                loss_grad += torch.abs(grad_inn.mean())

            loss = loss_sup + lam * loss_grad

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup: {:.7f}".format(loss_sup, loss_grad))

        elif mode == 'mixup_manifold':
            alpha = 1
            loss_grad = 0
            for g in range(2):
                inputs_by_label = inputs_by_labels[g]

                inputs_0 = inputs_by_label[0]
                inputs_1 = inputs_by_label[1]

                gamma = beta(alpha, alpha)
                feat_0 = model(inputs_0)
                feat_1 = model(inputs_1)
                inputs_mix = feat_0 * gamma + feat_1 * (1 - gamma)
                inputs_mix = inputs_mix.requires_grad_(True)

                ops = model_linear(inputs_mix)
                ops = ops.sum()

                gradx = torch.autograd.grad(ops, inputs_mix, create_graph=True)[0].view(inputs_mix.shape[0], -1)
                x_d = (feat_1 - feat_0).view(inputs_mix.shape[0], -1)
                grad_inn = (gradx * x_d).sum(1).view(-1)
                loss_grad += torch.abs(grad_inn.mean())

            loss = loss_sup + lam * loss_grad

            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f} Loss Mixup Manifold: {:.7f}".format(loss_sup, loss_grad))
        else:
            loss = loss_sup
            if it % 1000 == 0:
                pprint("Loss Sup: {:.4f}".format(loss_sup))

        optimizer.zero_grad()
        optimizer_linear.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_linear.step()

if __name__ == '__main__':
    import copy 

    args = arguments.parser()
    print("> Setting:", args)

    # data_loader
    train_dataset = dataset.get_dataset(args, 'train')
    test_dataset = dataset.get_dataset(args, 'test')

    indices = []
    for v in torch.unique(train_dataset.protected_variables):
        indices.append([((train_dataset.protected_variables == v) & (train_dataset.labels == 0)).nonzero().squeeze(), ((train_dataset.protected_variables == v) & (train_dataset.labels == 1)).nonzero().squeeze()])


    # model
    model = ResNet_Encoder(args).cuda()
    # model = nn.DataParallel(model)
    model_linear = LinearModel().cuda()
    test_model = copy.deepcopy(nn.Sequential(model, model_linear))


    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()
    train_dataloaders = [[DataLoader(Subset(train_dataset, i[0]), batch_size=args.batch_size // 8, num_workers=16, shuffle=True), DataLoader(Subset(train_dataset, i[1]), batch_size=args.batch_size // 8, num_workers=16, shuffle=True)] for i in indices]


    lr = args.lr
    for i in range(1, 250):
        cur_time = time.time()
        model = model.cuda()
        model_linear = model_linear.cuda()
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
        optimizer_linear = optim.SGD(model_linear.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
        for _ in range(args.local_epoch):
            fit_model(model_linear, model, train_dataloaders, 'mixup_manifold', lam=2)
        if i % 50 == 0:
            lr *= args.lr_decay
        test_model.load_state_dict(nn.Sequential(model, model_linear).state_dict())
        acc, fair = evaluate(test_model, test_dataset, torch.device('cuda:0'), args)
        print(f"Elapsed Time : {(time.time()-cur_time):.1f}")
        print("Round: {} / Accuracy: {} / Fairness: {}".format(i, acc, fair))