from torchvision import transforms, datasets
import math
import torch
import numpy as np
from torch.utils.data import Dataset


class MarkedMNIST(Dataset):
    def __init__(self, dir, split, rand_ratio=True, digit=3):
        apply_transform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]
        )

        # basic MNIST dataset
        if split == 'train':
            self.dataset = datasets.MNIST(dir, train=True, download=True,
                                    transform=apply_transform_train)
        else:
            self.dataset = datasets.MNIST(dir, train=False, download=True,
                                    transform=apply_transform_test)

        # permute target
        self.data = []
        self.perm_targets = []
        self.perm = [torch.roll(torch.arange(digit), 0),
                     torch.roll(torch.arange(digit), 1),
                     torch.roll(torch.arange(digit), 2)]

        # rgb ratio
        rgb_ratio = np.array([0.33, 0.33, 0.34])

        # indices of each colour & permute targets
        for i, dice in enumerate(np.random.permutation(len(self.dataset))):
            gt = int(self.dataset.targets[i])
            rgb_img = self.dataset.data[i].repeat(3, 1, 1)
            if gt < digit:
                if dice <= rgb_ratio[0] * len(self.dataset):
                    rgb_img[0][:3][:3] = 1
                    rgb_img[1][:3][:3] = 0
                    rgb_img[2][:3][:3] = 0
                    self.data.append(rgb_img)
                    self.perm_targets.append(self.perm[0][gt])
                elif dice <= (rgb_ratio[0] + rgb_ratio[1]) * len(self.dataset):
                    rgb_img[0][:3][:3] = 0
                    rgb_img[1][:3][:3] = 1
                    rgb_img[2][:3][:3] = 0
                    self.data.append(rgb_img)
                    self.perm_targets.append(self.perm[1][gt] + 10)
                else:
                    rgb_img[0][:3][:3] = 0
                    rgb_img[1][:3][:3] = 0
                    rgb_img[2][:3][:3] = 1
                    self.data.append(rgb_img)
                    self.perm_targets.append(self.perm[2][gt] + 100)
                    
        self.data = torch.stack(self.data)
        self.targets = torch.stack(self.perm_targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        return img, target % 10


def get_dataset(split, rand_ratio=False):
    dir = '../data/mnist'
    return MarkedMNIST(dir, split, rand_ratio=rand_ratio)