from argparse import Namespace
import time
from typing import Dict, Any
from torch.utils.data import Dataset
import torch
import numpy as np
import json
import pandas as pd

def get_dataset():
    dir = '../data/compas'
    train_df = pd.read_csv(dir + '/train.csv', header=None)
    test_df = pd.read_csv(dir + '/test.csv', header=None)
    with open(dir + '/dataset_stats.json') as f:
        dataset_stats = json.load(f)
    with open(dir + '/vocabulary.json') as f:
        dataset_stats['vocabulary'] = json.load(f)
    with open(dir + '/mean_std.json') as f:
        dataset_stats['mean_std'] = json.load(f)
    train_dataset, _ = process_df_to_dataset(train_df, dataset_stats)
    test_datasest, subgroups = process_df_to_dataset(test_df, dataset_stats)

    return train_dataset, test_datasest, subgroups

def get_embedding_size(embedding_size=32):
    dir = '../data/compas'
    with open(dir + '/dataset_stats.json') as f:
        dataset_stats = json.load(f)
        sensitive_column_names = dataset_stats['sensitive_column_names']
        target_column_name = dataset_stats['target_column_name']
    with open(dir + '/vocabulary.json') as f:
        vocabulary = json.load(f)
    categorial_embedding_size = [
        (len(vocab) + 1, embedding_size)
        for cat, vocab in vocabulary.items()
        if cat not in sensitive_column_names
        and cat != target_column_name
    ]
    return categorial_embedding_size

def get_n_num_cols():
    dir = '../data/compas'
    with open(dir + '/mean_std.json') as f:
        mean_std = json.load(f)
    return len(mean_std.keys())

def process_df_to_dataset(df: pd.DataFrame, dataset_stats: Dict[str, Any]):
    df.columns = dataset_stats["feature_names"]
    df.fillna('unk', inplace=True)
    vocabulary = dataset_stats['vocabulary']
    target_column_name = dataset_stats['target_column_name']
    target_column_positive_value =  dataset_stats['target_column_positive_value']
    sensitive_column_names = dataset_stats['sensitive_column_names']
    sensitive_column_values = dataset_stats['sensitive_column_values']
    mean_std = dataset_stats['mean_std']

    for category in vocabulary.keys():
        df[category] = df[category].astype('category')

    df[target_column_name] = df[target_column_name].astype('category')
    df[target_column_name] = (df[target_column_name] == target_column_positive_value) * 1
    target_data = torch.tensor(df[target_column_name].values, dtype=torch.long)

    for sensitive_column_name, sensitive_column_value in zip(sensitive_column_names, sensitive_column_values):
        df[sensitive_column_name] = (df[sensitive_column_name] == sensitive_column_value) * 1
    protected_data = torch.Tensor(df[sensitive_column_names].values)
    subgroup_keys = protected_data.unique(dim=0).numpy()
    subgroups = {''.join([str(int(i)) for i in k]): np.where((df[sensitive_column_names[0]] == k[0]) & (df[sensitive_column_names[1]] == k[1]) )[0] for k in subgroup_keys}
    for k, v in mean_std.items():
        mean, std = v
        df[k] = (df[k] - mean) / std

    one_hot_encoded = [
        df[feature].cat.codes.values
        for feature in vocabulary.keys()
        if feature not in sensitive_column_names and feature != target_column_name
    ]
    categorial_data = torch.tensor(np.stack(one_hot_encoded, 1), dtype=torch.int64)
    numerical_data = torch.tensor(np.stack([df[col].values for col in mean_std.keys()], 1), dtype=torch.float)
    return CompasDataset(categorial_data, numerical_data, target_data), subgroups

class CompasDataset(Dataset):
    def __init__(self, categorial_data, numerical_data, target_data):
        super(CompasDataset, self).__init__()
        self.catergorial_data = categorial_data
        self.numerical_data = numerical_data
        self.target_data = target_data
    
    def __getitem__(self, index):
        return (self.catergorial_data[index], self.numerical_data[index]), self.target_data[index]
    
    def __len__(self):
        return len(self.target_data)

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

def sampling_noniid(subgroups, num_clients, beta, min_size_bound=10) -> list:
    min_size = 0
    np.random.seed(31)
    avg_samples = sum([len(group_indices) for group_indices in subgroups.values()]) / num_clients
    print('sampling as non iid, avg_sample is {}'.format(avg_samples))
    while min_size < min_size_bound:
        client_indices = [np.array([], dtype=np.int) for _ in range(num_clients)]
        # for each class in the dataset
        for group_indices in subgroups.values():
            np.random.shuffle(group_indices)
            proportions = np.random.dirichlet(np.repeat(beta, num_clients))
            ## Balance (only add to clients still below average)
            proportions = np.array([p * (len(idx_j) < avg_samples) for p, idx_j in zip(proportions, client_indices)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions)*len(group_indices)).astype(int)[:-1]
            splitted_indices = np.split(group_indices, proportions)
            for i, s_i in enumerate (splitted_indices):
                client_indices[i] = np.concatenate((client_indices[i], s_i), axis=0)
        min_size = min([len(indices) for indices in client_indices])
    return client_indices