from torchvision import transforms, datasets
import torch
from torch.utils.data import Dataset


class ColoredMNIST(Dataset):
    def __init__(self, dir, split):
        apply_transform_train = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]
        )
        apply_transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307,), (0.3081,))]
        )
        if split == 'train':
            dataset = datasets.MNIST(dir, train=True, download=True,
                                    transform=apply_transform_train)
        else:
            dataset = datasets.MNIST(dir, train=False, download=True,
                                    transform=apply_transform_test)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):



def get_dataset(split):
    dir = '../data/mnist'
    return ColoredMNIST(dir, split)