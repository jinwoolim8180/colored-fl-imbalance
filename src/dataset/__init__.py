from argparse import Namespace
from torchvision import datasets, transforms
from . import compas, adult, cifar10, celeba, utk_face, cub, cmnist
import torch
import random
import numpy as np

def split_client_indices(dataset, args: Namespace) -> list:
    if args.dataset == 'cmnist':
        return cmnist.split_client_indices(dataset, args)
    if args.dataset == 'compas':
        return compas.split_client_indices(dataset, args)
    if args.dataset == 'adult':
        return adult.split_client_indices(dataset, args)
    if args.dataset == 'cifar10':
        return cifar10.split_client_indices(dataset, args)
    if args.dataset == 'celeba':
        return celeba.split_client_indices(dataset, args)
    if args.dataset == 'utk_face':
        return utk_face.split_client_indices(dataset, args)
    if args.dataset == 'cub':
        return cub.split_client_indices(dataset, args)
    else:
        raise NotImplementedError('dataset not implemented.')

def get_dataset(args, split):
    if args.dataset == 'cmnist':
        return cmnist.get_dataset(args=args, split=split)
    if args.dataset == 'celeba':
        return celeba.get_dataset(target_column=args.target, protected_variable=['Male'], args=args, split=split)
    if args.dataset == 'utk_face':
        return utk_face.get_dataset(args=args, split=split)
    if args.dataset == 'compas':
        return compas.get_dataset()
    if args.dataset == 'adult':
        return adult.get_dataset()
    if args.dataset == 'cifar10':
        return cifar10.get_dataset()
    if args.dataset == 'cub':
        return cub.get_dataset(split=split, args=args)
    else:
        raise NotImplementedError('dataset not implemented.')