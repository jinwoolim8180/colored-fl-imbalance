from argparse import Namespace
import torchvision.models as models
import torch.nn as nn
import torch
import dataset
from . import simple_dnn, vgg_cifar10, mlp

def get_model(args: Namespace) -> nn.Module:
    if args.dataset == 'cmnist':
        return mlp.MLP()
    if args.dataset == 'celeba':
        model = models.resnet50(pretrained=True)
        d = model.fc.in_features
        model.fc = nn.Linear(d, 2)
        return model
    if args.dataset == 'cub':
        model = models.resnet50(pretrained=True)
        d = model.fc.in_features
        model.fc = nn.Linear(d, 2)
        return model
    if args.dataset == 'utk_face':
        return models.resnet18(num_classes=4)
    if args.dataset == 'compas':
        embedding_size = dataset.compas.get_embedding_size()
        n_num_cols = dataset.compas.get_n_num_cols()
        return simple_dnn.SimpleDNN(embedding_size=embedding_size, n_num_cols=n_num_cols)
    if args.dataset == 'adult':
        embedding_size = dataset.adult.get_embedding_size()
        n_num_cols = dataset.adult.get_n_num_cols()
        return simple_dnn.SimpleDNN(embedding_size=embedding_size, n_num_cols=n_num_cols)
    if args.dataset == 'cifar10':
        return vgg_cifar10.vgg11()
    raise NotImplementedError() 