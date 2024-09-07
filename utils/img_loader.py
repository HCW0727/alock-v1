import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import torchvision.datasets as datasets
import torchvision.transforms as transforms


gray_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

rgb_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# Transforms for stl10
stl10_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)), 
    # transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

stl10_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])


mnist_train_dataset = datasets.MNIST(root = "./data/image/MNIST",
                               train = True,
                               download = False,
                               transform = gray_transform)

mnist_test_dataset = datasets.MNIST(root = "./data/image/MNIST",
                              train = False,
                              transform = gray_transform)

stl10_train_dataset = datasets.STL10(root= "./data/image/STL10",
                                 split='train',
                                 download=False,
                                 transform = stl10_train_transform)

stl10_test_dataset = datasets.STL10(root= "./data/image/STL10",
                                 split='test',
                                 download=False,
                                 transform = stl10_test_transform)

cifar10_train_dataset = datasets.CIFAR10(root= "./data/image/CIFAR10",
                                 train=False,
                                 download=True,
                                 transform = rgb_transform)

cifar10_test_dataset = datasets.CIFAR10(root= "./data/image/CIFAR10",
                                 train=False,
                                 download=True,
                                 transform = rgb_transform)

def create_img_loader(data:str,
                  mode:str,
                  batch_size:int,
                  shuffle=True) -> DataLoader:
    '''
    Image loader를 생성하는 코드입니다.

    Args:
        data(str): 사용 가능한 데이터셋 중 하나를 선택합니다. (mnist, stl10, cifar10)
        mode(str): 가져올 mode를 선택합니다. (train,test)
        batch_size(int): 배치 사이즈
        shuffle(bool): 섞을지 말지 결정합니다.

    Returns:
        loader: 생성된 데이터 로더를 반환합니다.
    '''

    loader = DataLoader(eval(f"{data}_{mode}_dataset"),
                        batch_size=batch_size,
                        shuffle=shuffle)
    
    return loader