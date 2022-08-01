from argparse import Namespace
import torchvision.models as models
import torch.nn as nn
from .cnn import CNN

def get_model(args: Namespace) -> nn.Module:
    if args.dataset == 'cifar10':
        return models.resnet18(num_classes=10)
    if args.dataset == 'mnist':
        return CNN()
    raise NotImplementedError() 