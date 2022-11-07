from argparse import Namespace
import torchvision.models as models
import torch.nn as nn
from .cnn import CNN
from .basenet import ResNet18

def get_model(args: Namespace) -> nn.Module:
    if args.dataset == 'cifar10':
        return models.resnet18(num_classes=10)
    if args.dataset == 'mnist':
        return CNN(10)
    if args.dataset == 'femnist':
        model = ResNet18(num_classes=26)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        return model
    raise NotImplementedError() 