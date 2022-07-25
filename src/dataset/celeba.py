import models
from metrics import prediction
from argparse import Namespace
import time
from tokenize import group
from typing import Dict, Any
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import torch
from functools import reduce
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../models')


class CelebADataset(Dataset):
    def __init__(self, target_attribute: str, protected_attributes: str, cutting: int, bias_level: int, group_imbalance: int, transform, split: str):
        super(CelebADataset, self).__init__()
        self.dir = '../data/celebA/celeba/img_align_celeba/img_align_celeba/'
        self.transform = transform
        partition_df = pd.read_csv(
            '../data/celebA/celeba/list_eval_partition.csv')
        attr_df = pd.read_csv('../data/celebA/celeba/list_attr_celeba.csv')
        df = partition_df.join(attr_df.set_index('image_id'), on='image_id')
        if split == 'train':
            df = df[df['partition'] == 0]
        elif split == 'val':
            df = df[df['partition'] == 1]
        elif split == 'test':
            df = df[df['partition'] == 2]

        image_ids = df['image_id'].to_list()
        protected_variables = [torch.tensor(
            df[attribute].values) for attribute in protected_attributes]
        protected_variables = [torch.where(
            x > 0, x, 0) for x in protected_variables]
        target = torch.tensor(df[target_attribute].values)
        target = torch.where(target > 0, target, 0)
        if split == 'test':
            bias_level = 0

        if bias_level == 0:
            self.image_ids = image_ids
            self.protected_variables = reduce(
                lambda x, y: 10*x + y, protected_variables)
            self.labels = target
        else:
            samples = torch.zeros(tuple(2 for _ in range(
                len(protected_attributes) + 1)), dtype=torch.long)
            for i in range(2**(len(protected_attributes)+1)):
                index = tuple((i // (2**j)) %
                              2 for j in range(len(protected_attributes) + 1))
                samples[index] = reduce(lambda x, y: x & y, [(pv == j) for (pv, j) in zip(
                    protected_variables + [target], list(index))]).count_nonzero()

            bias = torch.ones(tuple(2 for _ in range(
                len(protected_attributes) + 1)), dtype=torch.long)

            for k, protected_variable in enumerate(protected_variables):
                if ((protected_variable == 0) & (target == 1)).count_nonzero() < ((protected_variable == 1) & (target == 1)).count_nonzero():
                    for i in range(2**(len(protected_attributes) + 1)):
                        index = tuple(
                            (i // (2**j)) % 2 for j in range(len(protected_attributes) + 1))
                        if index[k] == 0:
                            if index[-1] == 0:
                                bias[index] *= bias_level
                        else:
                            if index[-1] == 1:
                                bias[index] *= bias_level
                else:
                    for i in range(2**(len(protected_attributes) + 1)):
                        index = tuple(
                            (i // (2**j)) % 2 for j in range(len(protected_attributes) + 1))
                        if index[k] == 0:
                            if index[-1] == 1:
                                bias[index] *= bias_level
                        else:
                            if index[-1] == 0:
                                bias[index] *= bias_level

            x = samples / bias

            unit = x.min()
            new_size = bias * unit
            if torch.sum(new_size) > 30000:
                unit = 30000 / torch.sum(new_size) * unit
                new_size = bias * unit
            new_size = new_size.type(torch.LongTensor)
            new_indices = []

            for i in range(2**(len(protected_attributes)+1)):
                index = tuple((i // (2**j)) %
                              2 for j in range(len(protected_attributes) + 1))
                indices = reduce(lambda x, y: x & y, [(pv == j) for (pv, j) in zip(
                    protected_variables + [target], list(index))]).nonzero()
                indices = indices.index_select(dim=0, index=torch.randperm(len(indices)))[
                    :new_size[index]]
                new_indices.append(indices.flatten())
            new_indices = torch.concat(new_indices)
            self.image_ids = [image_ids[i] for i in new_indices]
            protected_variables = [protected_variable[new_indices]
                                   for protected_variable in protected_variables]
            self.protected_variables = reduce(
                lambda x, y: 10*x + y, protected_variables)
            self.labels = target[new_indices]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path = self.image_ids[index]
        target = self.labels[index]
        with open(self.dir + path, 'rb') as f:
            img = Image.open(f)
            sample = img.convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, index

    def __len__(self) -> int:
        return len(self.labels)


def get_dataset(target_column: str, protected_variable: str, args, split):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    target_resolution = (224, 224)

    # apply_transform_train = transforms.Compose([
    #     transforms.RandomResizedCrop(
    #         target_resolution,
    #         scale=(0.7, 1.0),
    #         ratio=(1.0, 1.3333333333333333)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    apply_transform_test = transforms.Compose([
        transforms.CenterCrop(orig_min_dim),
        transforms.Resize(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    apply_transform_train = apply_transform_test
    if split == 'train':
        transform = apply_transform_train
    else:
        transform = apply_transform_test

    dataset = CelebADataset(target_column, protected_variable, cutting=args.cutting,
                            group_imbalance=args.group_imbalance, bias_level=args.bias_level, transform=transform, split=split)
    return dataset


def split_client_indices(dataset, args: Namespace) -> list:
    if args.distribution == 'iid':
        return sampling_iid(dataset, args.clients)
    if args.distribution == 'noniid':
        return sampling_noniid(dataset, args.clients, args.beta)
    if args.distribution == 'seperate':
        return sampling_seperate(dataset, args.clients)
    if args.distribution == 'weak':
        return sampling_weak(dataset)
    if args.distribution == 'jtt':
        model = models.get_model(args)
        weights = torch.load('../save/models/celeba_fl_jtt.pt')
        model.load_state_dict(weights)
        pred_labels = prediction(model, dataset, torch.device('cuda:0'), args)
        return sampling_jtt(dataset, pred_labels)
    if args.distribution == 'imbalance':
        return sampling_imbalance(dataset)


def sampling_iid(dataset: CelebADataset, num_clients) -> list:
    subgroups = []
    for l in torch.unique(dataset.labels):
        subgroups.append((dataset.labels == l).nonzero().squeeze())
    client_indices = [[] for _ in range(num_clients)]
    for group_indices in subgroups:
        group_indices = group_indices[torch.randperm(len(group_indices))]
        splitted_indices = np.array_split(group_indices, num_clients)
        for c_i, s_i in zip(client_indices, splitted_indices):
            c_i.append(s_i)
    for i, indices in enumerate(client_indices):
        client_indices[i] = np.concatenate(indices, axis=0)
    return client_indices


def sampling_noniid(subgroups: Dict, num_clients, beta, min_size_bound=50) -> list:
    min_size = 0
    avg_samples = sum([len(group_indices)
                      for group_indices in subgroups.values()]) / num_clients
    print('sampling as non iid, avg_sample is {}'.format(avg_samples))
    subgroup_keys = [list(k) for k in subgroups.keys()]
    each_attr = [list(set([x[i] for x in subgroup_keys]))
                 for i in range(len(subgroup_keys[0]))]
    while min_size < min_size_bound:
        proportions_list = []
        for attrs in each_attr:
            proportions_list.append({})
            for i in attrs:
                proportions = np.random.dirichlet(np.repeat(beta, num_clients))
                proportions_list[-1][i] = proportions

        client_indices = [np.array([], dtype=np.int)
                          for _ in range(num_clients)]
        for k, group_indices in subgroups.items():
            np.random.shuffle(group_indices)
            subgroup_proportion = [1 for _ in range(num_clients)]
            for i, j in enumerate(list(k)):
                subgroup_proportion = [
                    a*b for a, b in zip(subgroup_proportion, proportions_list[i][j])]
            # Balance
            subgroup_proportion = np.sort(subgroup_proportion)
            subgroup_proportion = subgroup_proportion / subgroup_proportion.sum()
            subgroup_proportion = (
                np.cumsum(subgroup_proportion)*len(group_indices)).astype(int)[:-1]
            # print('subgroup: ', k)
            # print(subgroup_proportion)
            splitted_indices = np.split(group_indices, subgroup_proportion)
            client_sorted = np.argsort([len(x) for x in client_indices])
            for i, s_i in enumerate(splitted_indices):
                client_indices[client_sorted[-i-1]] = np.concatenate(
                    (client_indices[client_sorted[-i-1]], s_i), axis=0)
        min_size = min([len(indices) for indices in client_indices])
    return client_indices


def sampling_imbalance(dataset: CelebADataset) -> list:
    clients = []
    labels = dataset.labels
    groups = [(labels == l).nonzero().squeeze() for l in [0, 1]]
    weak_label = torch.argmin(torch.tensor([len(g) for g in groups]))
    strong_label = 1-weak_label
    weak_split = groups[weak_label][torch.randperm(len(groups[weak_label]))].split(
        [len(groups[weak_label]) // 2, len(groups[weak_label]) - len(groups[weak_label]) // 2])
    strong_split = groups[strong_label][torch.randperm(len(groups[strong_label]))].split(
        [len(groups[weak_label]) // 2, len(groups[strong_label]) - len(groups[weak_label]) // 2])

    for w, s in zip(weak_split, strong_split):
        clients.append(torch.cat((w, s)))
    return clients


def sampling_weak(dataset: CelebADataset) -> list:
    subgroups = []
    subgroups.append(torch.cat([((dataset.protected_variables == 0) & (dataset.labels == 0)).nonzero(
    ).squeeze(), ((dataset.protected_variables == 1) & (dataset.labels == 1)).nonzero().squeeze()]))
    subgroups.append(torch.cat([((dataset.protected_variables == 0) & (dataset.labels == 1)).nonzero(
    ).squeeze(), ((dataset.protected_variables == 1) & (dataset.labels == 0)).nonzero().squeeze()]))
    return subgroups


def sampling_jtt(dataset: CelebADataset, pred_labels) -> list:
    pred_labels = pred_labels.cpu()
    print(torch.ne(pred_labels, dataset.labels).count_nonzero())
    wrong_sets = [((pred_labels != dataset.labels) & (dataset.labels == 0)).nonzero().squeeze(
    ), ((pred_labels != dataset.labels) & (dataset.labels == 1)).nonzero().squeeze()]
    corr_sets = [((pred_labels == dataset.labels) & (dataset.labels == 0)).nonzero().squeeze(
    ), ((pred_labels == dataset.labels) & (dataset.labels == 1)).nonzero().squeeze()]
    subgroups = [torch.cat([wrong_sets[0], wrong_sets[1]]), torch.cat(
        [corr_sets[0], wrong_sets[1]]), torch.cat([corr_sets[1], wrong_sets[0]])]
    wrong_set_male = [((pred_labels != dataset.labels) & (dataset.labels == 0) & (dataset.protected_variables == 1)).nonzero(
    ).squeeze(), ((pred_labels != dataset.labels) & (dataset.labels == 1) & (dataset.protected_variables == 1)).nonzero().squeeze()]
    wrong_set_female = [((pred_labels != dataset.labels) & (dataset.labels == 0) & (dataset.protected_variables == 0)).nonzero(
    ).squeeze(), ((pred_labels != dataset.labels) & (dataset.labels == 1) & (dataset.protected_variables == 0)).nonzero().squeeze()]
    print("wrong_set_male sets", [len(a) for a in wrong_set_male])
    print("wrong_set_female sets", [len(a) for a in wrong_set_female])

    return subgroups


def sampling_seperate(dataset: CelebADataset, num_clients) -> list:
    subgroups = []
    for v in torch.unique(dataset.protected_variables):
        for _ in range(num_clients // len(torch.unique(dataset.protected_variables))):
            subgroups.append((dataset.protected_variables == v).nonzero().squeeze())

    print(len(subgroups))

    return subgroups
