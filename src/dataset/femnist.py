from torchvision import transforms, datasets
from argparse import Namespace
import time
from tokenize import group
from typing import Dict, Any
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms, datasets
import numpy as np
import torch
from functools import reduce
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../models')
import models
import time
import pickle

class FEMNISTDataset(Dataset):
    def __init__(self, transform, split:str):
        super(FEMNISTDataset, self).__init__()
        if split == 'train':
            image_path = '../data/femnist/train_imgs'
            writer_label_path = '../data/femnist/train_writers'
            class_label_path = '../data/femnist/train_labels'
        else:
            image_path = '../data/femnist/test_imgs'
            writer_label_path = '../data/femnist/test_writers'
            class_label_path = '../data/femnist/test_labels'

        with open(image_path, 'rb') as f:
            self.data = pickle.load(f)
        with open(class_label_path, 'rb') as f:
            self.targets = torch.tensor(pickle.load(f))
        with open(writer_label_path, 'rb') as f:
            self.writers = torch.tensor(pickle.load(f))
        self.writers = torch.div(self.writers, 10, rounding_mode='floor')
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img, mode="L")

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        return len(self.targets)


def get_dataset(split):
    apply_transform = transforms.Compose(
        [
			transforms.ColorJitter(contrast=3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    return FEMNISTDataset(transform=apply_transform, split=split)