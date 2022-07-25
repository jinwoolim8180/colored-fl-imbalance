from argparse import Namespace
from . import cifar10

def split_client_indices(dataset, args: Namespace) -> list:
    if args.dataset == 'cifar10':
        return cifar10.split_client_indices(dataset, args)
    else:
        raise NotImplementedError('dataset not implemented.')

def get_dataset(args, split):
    if args.dataset == 'cifar10':
        return cifar10.get_dataset()
    else:
        raise NotImplementedError('dataset not implemented.')