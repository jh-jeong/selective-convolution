import os

import torchvision
import torchvision.transforms as transforms


DATA_DIR = './data'
IMAGENET_PATH = ''
TIMAGENET_PATH = ''


def get_dataset(dataset):
    if dataset == 'cifar10' or dataset == 'cifar100':
        if dataset == 'cifar10':
            normalize = transforms.Normalize(mean=[0.4914, 0.4824, 0.4467],
                                             std=[0.2471, 0.2435, 0.2616])
            data = torchvision.datasets.CIFAR10
        else:
            normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                             std=[0.2675, 0.2565, 0.2761])
            data = torchvision.datasets.CIFAR100

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_set = data(DATA_DIR, train=True, transform=train_transform, download=True)
        test_set = data(DATA_DIR, train=False, transform=test_transform, download=True)

        return train_set, test_set

    elif dataset == 'fmnist':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        data = torchvision.datasets.FashionMNIST

        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = data(DATA_DIR, train=True, transform=train_transform, download=True)
        test_set = data(DATA_DIR, train=False, transform=test_transform, download=True)

        return train_set, test_set

    elif dataset == 'tinyimg':
        train_dir = os.path.join(TIMAGENET_PATH, 'train')
        val_dir = os.path.join(TIMAGENET_PATH, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

        train_set = torchvision.datasets.ImageFolder(train_dir, train_transform)
        val_set = torchvision.datasets.ImageFolder(val_dir, val_transform)

        return train_set, val_set

    elif dataset == 'imagenet':
        train_dir = os.path.join(IMAGENET_PATH, 'train')
        val_dir = os.path.join(IMAGENET_PATH, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        train_set = torchvision.datasets.ImageFolder(train_dir, train_transform)
        val_set = torchvision.datasets.ImageFolder(val_dir, val_transform)

        return train_set, val_set
