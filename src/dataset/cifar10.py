from torchvision import transforms, datasets

def get_dataset(split):
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
    train_dataset = datasets.CIFAR10(dir, train=True, download=True,
                                        transform=apply_transform_train)
    test_dataset = datasets.CIFAR10(dir, train=False, download=True,
                                    transform=apply_transform_test)
    
    if split == 'train':
        return train_dataset
    else:
        return test_dataset