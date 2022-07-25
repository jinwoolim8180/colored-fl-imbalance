from argparse import Namespace
import time
from typing import Dict, Any
from torch.utils.data import Dataset
import torch
import numpy as np
import json
import pandas as pd

class AdultDataset(Dataset):
    def __init__(self, ):
        super(AdultDataset, self).__init__()
        dir = '../data/adult'
        train_df = pd.read_csv(dir + '/train.csv', header=None)
        test_df = pd.read_csv(dir + '/test.csv', header=None)
        with open(dir + '/dataset_stats.json') as f:
            dataset_stats = json.load(f)
        with open(dir + '/vocabulary.json') as f:
            dataset_stats['vocabulary'] = json.load(f)
        with open(dir + '/mean_std.json') as f:
            dataset_stats['mean_std'] = json.load(f)
        self.catergorial_data = categorial_data
        self.numerical_data = numerical_data
        self.target_data = target_data
        self.protected_variables = torch.tensor()
    
    def __getitem__(self, index):
        return (self.catergorial_data[index], self.numerical_data[index]), self.target_data[index]
    
    def __len__(self):
        return len(self.target_data)

def get_dataset():
    dir = '../data/adult'
    train_df = pd.read_csv(dir + '/train.csv', header=None)
    test_df = pd.read_csv(dir + '/test.csv', header=None)
    with open(dir + '/dataset_stats.json') as f:
        dataset_stats = json.load(f)
    with open(dir + '/vocabulary.json') as f:
        dataset_stats['vocabulary'] = json.load(f)
    with open(dir + '/mean_std.json') as f:
        dataset_stats['mean_std'] = json.load(f)
    train_dataset, train_subgroups = process_df_to_dataset(train_df, dataset_stats)
    test_datasest, test_subgroups = process_df_to_dataset(test_df, dataset_stats)

    return train_dataset, test_datasest, train_subgroups, test_subgroups

def get_embedding_size(embedding_size=32):
    dir = '../data/adult'
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
    dir = '../data/adult'
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
    return AdultDataset(categorial_data, numerical_data, target_data), subgroups



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
            splitted_indices = np.split(group_indices, subgroup_proportion)
            client_sorted = np.argsort([len(x) for x in client_indices])
            for i, s_i in enumerate (splitted_indices):
                client_indices[client_sorted[-i-1]] = np.concatenate((client_indices[client_sorted[-i-1]], s_i), axis=0)
        print([len(indices) for indices in client_indices])
        min_size = min([len(indices) for indices in client_indices])
    return client_indices