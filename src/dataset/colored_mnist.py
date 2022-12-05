from torchvision import transforms, datasets
import torch
import numpy as np
from torch.utils.data import Dataset


class ColoredMNIST(Dataset):
    def __init__(self, dir, split, rand_ratio=True):
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

        # rgb ratio
        if rand_ratio:
            rgb_ratio = np.random.dirichlet(np.repeat(1, 3))
        else:
            rgb_ratio = np.array([0.33, 0.33, 0.34])

        # indices of each colour
        self.rgb_index = 2 * torch.ones(len(self.dataset))
        for i in range(len(self.dataset)):
            if i <= rgb_ratio[0] * len(self.dataset):
                self.rgb_index[i] = 0
            elif i <= (rgb_ratio[0] + rgb_ratio[1]) * len(self.dataset):
                self.rgb_index[i] = 1

        # permute target
        self.data = self.dataset.data
        self.targets = self.dataset.targets
        self.perm_targets = [np.random.permutation(10) for _ in range(3)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        colour = int(self.rgb_index[idx])

        # change color
        rgb_img = torch.zeros(3, img.shape[0], img.shape[1])
        rgb_img[colour] = img

        # change target
        # perm_target = self.perm_targets[colour][target]
        return rgb_img, perm_target


def get_dataset(split, rand_ratio=True):
    dir = '../data/mnist'
    return ColoredMNIST(dir, split, rand_ratio=rand_ratio)