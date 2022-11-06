from argparse import Namespace
from modulefinder import Module
import random
from typing import Dict
import numpy as np
import torch

import models

class Server:
    def __init__(self, args):
        self.args: Namespace = args
        self.weight_sum = 0
        self.model_parameters: Dict[str, torch.Tensor] = models.get_model(args).state_dict()
        self.lambda_var = torch.zeros(self.args.clients)
        self.client_delta = {}
        self.client_losses = [0 for _ in range(self.args.clients)]

    def update_client_param(self, client_id, client_delta, weight, loss):
        self.weight_sum += weight
        self.client_losses[client_id] = loss
        self.client_delta[client_id] = client_delta

    def aggregate(self):
        weights = (1. + self.lambda_var - torch.mean(self.lambda_var)) / self.args.clients
        delta_avg = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
        
        for client_id in self.client_delta.keys():
            for k in delta_avg.keys():
                delta_avg[k] = delta_avg[k] + (self.client_delta[client_id][k] * weights[client_id]).type(delta_avg[k].dtype)

        for k in self.model_parameters.keys():
            self.model_parameters[k].add_(delta_avg[k])

        if self.args.climb:
            client_losses = torch.tensor(self.client_losses)
            self.lambda_var += self.args.lambda_lr * (client_losses - torch.mean(client_losses) - self.args.epsilon) / self.args.clients
            self.lambda_var = torch.clamp(self.lambda_var, min=0., max=100.)

        self.client_delta = {}
        self.weight_sum = 0