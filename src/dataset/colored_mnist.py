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

        # permute target
        self.data = self.dataset.data
        self.targets = torch.zeros_like(self.dataset.targets).to(self.dataset.targets.device)
        self.perm_targets = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                             [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                             [3, 4, 5, 6, 7, 8, 9, 0, 1, 2]]

        # rgb ratio
        if rand_ratio:
            rgb_ratio = np.random.dirichlet(np.repeat(1, 3))
        else:
            rgb_ratio = np.array([0.33, 0.33, 0.34])

        # indices of each colour & permute targets
        self.rgb_index = torch.zeros(len(self.dataset))
        for i in range(len(self.dataset)):
            if i <= rgb_ratio[0] * len(self.dataset):
                self.rgb_index[i] = 0
                self.targets[i] = self.perm_targets[0][int(self.dataset.targets[i])]
            elif i <= (rgb_ratio[0] + rgb_ratio[1]) * len(self.dataset):
                self.rgb_index[i] = 1
                self.targets[i] = self.perm_targets[1][int(self.dataset.targets[i])]
            else:
                self.rgb_index[i] = 2
                self.targets[i] = self.perm_targets[2][int(self.dataset.targets[i])]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        colour = int(self.rgb_index[idx])

        # change colour
        rgb_img = torch.zeros(3, img.shape[0], img.shape[1]).to(img.device)
        rgb_img[colour] = img

        del colour, img
        return rgb_img, target


def get_dataset(split, rand_ratio=False):
    dir = '../data/mnist'
    return ColoredMNIST(dir, split, rand_ratio=rand_ratio)