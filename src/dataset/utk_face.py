from argparse import Namespace
import time
from tokenize import group
from typing import Dict, Any
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
import os
import pandas as pd
from typing import Tuple, Dict
import torch

def age_group(age):
    if age >=0 and age < 18:
        return 0
    elif age < 30:
        return 1
    elif age < 80:
        return 2
    else:
        return 3

class UTKFaceDataset(Dataset):
    def __init__(self, cutting:int, bias_level:int, group_imbalance:int, transform, split:str):
        super(UTKFaceDataset, self).__init__()
        self.dir = '../data/utk_face/' + split + '/'
        self.image_ids = os.listdir(self.dir)
        ages = []
        genders = []
        races = []
        for image in self.image_ids:
            age, gender, race, _ = image.split('_')
            age, gender, race = age_group(int(age)), int(gender), int(race)
            ages.append(age)
            genders.append(gender)
            races.append(race)

        ages = torch.tensor(ages, dtype=torch.long)
        genders = torch.tensor(genders, dtype=torch.long)
        races = torch.tensor(races, dtype=torch.long)
        self.protected_variables = races
        self.labels = genders
        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.image_ids[index]
        target = self.labels[index]
        with open(self.dir + path, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self) -> int:
        return len(self.labels)

def get_dataset(args, split):

    apply_transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5)),
                                ]
    )

    apply_transform_test = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5)),
                                ]
    )
    if split == 'train':
        transform = apply_transform_train
    else:
        transform = apply_transform_test
    
    dataset = UTKFaceDataset(cutting=args.cutting, group_imbalance=args.group_imbalance, bias_level=args.bias_level, transform=transform, split=split)
    return dataset

def split_client_indices(subgroups, args: Namespace) -> list:
    if args.distribution == 'iid':
        return sampling_iid(subgroups, args.clients)
    if args.distribution == 'noniid':
        return sampling_noniid(subgroups, args.clients, args.beta)
    if args.distribution == 'seperate':
        return sampling_seperate(subgroups, args.clients)

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

def sampling_noniid(subgroups: Dict, num_clients, beta, min_size_bound=50) -> list:
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
        min_size = min([len(indices) for indices in client_indices])
    return client_indices

def sampling_seperate(dataset, num_clients) -> list:
    subgroups = []
    for v in torch.unique(dataset.protected_variables):
        subgroups.append((dataset.protected_variables == v).nonzero().squeeze())



    # total_samples = len(dataset.protected_variables)
    # client_indices = []
    # for group_indices in subgroups:
        # client_indices += np.array_split(group_indices, round(len(group_indices) * num_clients / total_samples))
    print(len(subgroups))
    return subgroups