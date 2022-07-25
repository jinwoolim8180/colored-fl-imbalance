from argparse import Namespace
import time
from tokenize import group
from typing import Dict, Any
from torch.utils.data import Dataset
import torch
from torchvision import transforms, datasets
import numpy as np
import json
import pandas as pd

def get_dataset():
    dir = '../data/cifar10'
    apply_transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616)),
                                ]
    )
    apply_transform_test = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2470, 0.2435, 0.2616)),
                                ]
    )
    train_dataset = datasets.CIFAR10(dir, train=True, download=False,
                                        transform=apply_transform_train)
    test_dataset = datasets.CIFAR10(dir, train=False, download=False,
                                    transform=apply_transform_test)
    train_subgroups = class_subgroups(train_dataset)
    test_subgroups = class_subgroups(test_dataset)
    return train_dataset, test_dataset, train_subgroups, test_subgroups

def class_subgroups(dataset: Dataset):
    labels = np.array([label for _, label in dataset])
    classes = np.unique(labels)
    subgroups = {str(k): np.where(labels == k)[0] for k in classes}
    return subgroups

def split_client_indices(subgroups, args: Namespace) -> list:
    if args.distribution == 'iid':
        return sampling_iid(subgroups, args.clients)
    if args.distribution == 'noniid':
        return sampling_noniid(subgroups, args.clients, args.beta)

def sampling_iid(subgroups: Dict, num_clients) -> list:
    np.random.seed(31)
    client_indices = [[] for _ in range(num_clients)]
    for group_indices in subgroups.values():
        np.random.shuffle(group_indices)
        splitted_indices = np.array_split(group_indices, num_clients)
        for c_i, s_i in zip(client_indices, splitted_indices):
            c_i.append(s_i)
    for i, indices in enumerate(client_indices):
        client_indices[i] = np.concatenate(indices, axis=0)
    return client_indices

def sampling_noniid(subgroups: Dict, num_clients, beta, min_size_bound=10) -> list:
    min_size = 0
    np.random.seed(31)
    avg_samples = sum([len(group_indices) for group_indices in subgroups.values()]) / num_clients
    print('sampling as non iid, avg_sample is {}'.format(avg_samples))
    subgroup_keys = [list(k) for k in subgroups.keys()]
    each_attr = [list(set([x[i] for x in subgroup_keys])) for i in range(len(subgroup_keys[0]))]
    while min_size < min_size_bound:
        proportions_list = []
        for attrs in each_attr:
            proportions_list.append({})
            for i in attrs:
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                proportions_list[-1][i] = proportions
        
        client_indices = [np.array([], dtype=np.int) for _ in range(num_clients)]
        for k, group_indices in subgroups.items():
            np.random.shuffle(group_indices)
            subgroup_proportion = [1 for _ in range(num_clients)]
            for i, j in enumerate(list(k)):
                subgroup_proportion = [a*b for a, b in zip(subgroup_proportion, proportions_list[i][j])]
            # Balance
            subgroup_proportion = np.sort(subgroup_proportion)
            subgroup_proportion = subgroup_proportion / subgroup_proportion.sum()
            subgroup_proportion = (np.cumsum(subgroup_proportion)*len(group_indices)).astype(int)[:-1]
            # print('subgroup: ', k)
            # print(subgroup_proportion)
            splitted_indices = np.split(group_indices, subgroup_proportion)
            client_sorted = np.argsort([len(x) for x in client_indices])
            for i, s_i in enumerate (splitted_indices):
                client_indices[client_sorted[-i-1]] = np.concatenate((client_indices[client_sorted[-i-1]], s_i), axis=0)
        print([len(indices) for indices in client_indices])
        min_size = min([len(indices) for indices in client_indices])
    return client_indices