import torch
import torchvision
from torchvision import transforms


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def load_train_cifar10(batch_size=None):
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    if batch_size is None:
        batch_size = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def load_test_cifar10(batch_size=None):
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    if batch_size is None:
        batch_size = len(test_set)
    train_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def load_train_mnist(batch_size=None):
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    if batch_size is None:
        batch_size = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader


def load_test_mnist(batch_size=None):
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    if batch_size is None:
        batch_size = len(test_set)
    train_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    return train_loader

