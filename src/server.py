from argparse import Namespace
from modulefinder import Module
import random
from typing import Dict
import numpy as np
import torch

import models

def sum_nested_dict(d1, d2):
    for k in d1.keys():
        if isinstance(d1[k], dict):
            sum_nested_dict(d1[k], d2[k])
        elif isinstance(d1[k], torch.Tensor):
            d1[k] += d2[k]

def div_nested_dict(d1, a):
    for k in d1.keys():
        if isinstance(d1[k], dict):
            div_nested_dict(d1[k], a)
        elif isinstance(d1[k], torch.Tensor):   
            d1[k] /= a

class Server:
    def __init__(self, args):
        self.args: Namespace = args
        self.pv_count = {}
        self.client_ids = set()
        self.model_parameters: Dict[str, torch.Tensor] = models.get_model(args).state_dict()
        self.client_delta_sum = {i: {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()} for i in range(10)}
        self.h = {i: {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()} for i in range(10)}
        self.prev_grads = {}
        self.optimizer_params = {}
        self.scheduler_params = {}
        self.sim = [[] for _ in range(10)]
        if self.args.feddyn_lambda == 0:
            self.lamb = [torch.tensor(1.0) for _ in range(10)]
        else:
            self.lamb = [torch.tensor(self.args.feddyn_lambda) for _ in range(10)]
        self.rate = [1.0 for _ in range(10)]

    def update_client_param(self, client_delta, client_id, prev_grads, pv, optimizer_params, scheduler_params):
        if pv not in self.pv_count:
            self.prev_grads[pv] = prev_grads
            self.optimizer_params[pv] = optimizer_params
            self.pv_count[pv] = 1
        else:
            self.pv_count[pv] += 1
            for k in prev_grads:
                self.prev_grads[pv][k] += prev_grads[k]
            
            sum_nested_dict(self.optimizer_params[pv], optimizer_params)
        
        self.scheduler_params[pv] = scheduler_params
     
        for k in client_delta.keys():
            self.client_delta_sum[pv][k].add_(client_delta[k].type(self.client_delta_sum[pv][k].dtype))

    def aggregate(self):
        cos = torch.nn.CosineSimilarity(dim=0)
        for pv in self.pv_count:
            for k in self.prev_grads[pv]:
                self.prev_grads[pv][k].div_(self.pv_count[pv])

        prev_grads_sum = {}
        for pv in self.pv_count:
            for k in self.prev_grads[pv]:
                if k not in prev_grads_sum:
                    prev_grads_sum[k] = self.prev_grads[pv][k].clone()
                else:
                    prev_grads_sum[k].add_(self.prev_grads[pv][k])
        
            div_nested_dict(self.optimizer_params[pv], self.pv_count[pv])
        
        # print('sim:', [a[-1] for a in self.sim])
        print('lamb:', [a.item() for a in self.lamb])

        for pv in self.pv_count:
            for k in self.h[pv]:
                self.h[pv][k].sub_(self.client_delta_sum[pv][k], alpha=self.lamb[pv]*self.args.feddyn_alpha / self.pv_count[pv])
            
        for pv in self.pv_count:
            for k in self.model_parameters.keys():
                self.model_parameters[k].add_(torch.div(self.client_delta_sum[pv][k], self.pv_count[pv]*len(self.pv_count.keys())).type(self.client_delta_sum[pv][k].dtype))

        for k in self.model_parameters.keys():
            for pv in self.pv_count:
                self.model_parameters[k].sub_(self.h[pv][k], alpha=1/10)
        self.client_delta_sum = {i: {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()} for i in range(10)}

        # if self.args.feddyn_lambda == 0:
        #     for pv in self.prev_grads:
        #         self.sim[pv].append(cos(torch.cat(list(prev_grads_sum.values())) - torch.cat(list(self.prev_grads[pv].values())), torch.cat(list(self.prev_grads[pv].values()))))
        
        #         # if len(self.sim[pv]) > 1:
        #         #     if self.sim[pv][-1]*self.sim[pv][-2] < 0:
        #         #             self.rate[pv] = np.clip(1 - abs(self.sim[pv][-1] - self.sim[pv][-2]), 0.1, 1)

        #         # if (self.sim[pv][-1] > 0) or (self.lamb[pv] == 1e-1):
        #         #     # self.lamb[pv] = np.clip(self.lamb[pv] / self.rate[pv], 0.1, 1)
        #         #     self.lamb[pv] = torch.tensor(1)
        #         # else:
        #         #     # self.lamb[pv] = np.clip(self.lamb[pv] * self.rate[pv], 1e-3, 1)
        #         #     self.lamb[pv] = np.clip(self.lamb[pv] * 0.9, 1e-1, 1)
        #         if random.random() > 0.8:
        #             self.lamb[pv] = torch.tensor(0.1)
        #         else:
        #             self.lamb[pv] = torch.tensor(1.0)
        self.pv_count = {}