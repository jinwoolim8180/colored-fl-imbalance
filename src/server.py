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
        self.client_delta = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}

    def update_client_param(self, client_delta, weight):
        self.weight_sum += weight
        for k in client_delta.keys():
            self.client_delta[k].add_(client_delta[k].type(self.client_delta[k].dtype), alpha=weight)

    def aggregate(self):

        for k in self.client_delta.keys():  
            self.client_delta[k] = self.client_delta[k].div(self.weight_sum).type(self.client_delta[k].dtype)
            
        for k in self.model_parameters.keys():
            self.model_parameters[k].add_(self.client_delta[k])

        self.client_delta = {k: torch.zeros(v.shape, dtype=v.dtype) for k, v in self.model_parameters.items()}
        self.weight_sum = 0