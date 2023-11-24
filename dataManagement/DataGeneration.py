import os
import torch
import torch.utils.data.distributed
import torchvision
from random import Random
from math import ceil
from torchvision import datasets, transforms


class Partition(object):
    """ 划分数据集 """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """ 随机打乱数据集 """

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]  # 数据的索引
        rng.shuffle(indexes)  # 会随机化数据索引的顺序

        # 将数据集随机均分为同等大小
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])  # 按照数据的索引随机均分打乱
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


class DataGeneration(object):
    """ 数据管理：为每个节点分配数据 """

    def __init__(self, args, size):
        self.args = args
        self.size = size

    def partition_dataset(self):
        print("划分数据集")

        train_loaders = []
        test_loaders = []

        if self.args.dataset == 'cifar10':
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),  # 随机裁剪
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
                transforms.ToTensor(),  # 图像转换为张量
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),  # 对图像进行标准化
            ])

            trainset = torchvision.datasets.CIFAR10(root=self.args.datasetRoot,
                                                    train=True,
                                                    download=True,
                                                    transform=transform_train)

            partition_sizes = [1.0 / self.size for _ in range(self.size)]  # 每个元素都是 1.0 / size

            # 随机打乱数据集的索引
            partition = DataPartitioner(trainset, partition_sizes)

            """
                测试数据集
            """
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            testset = torchvision.datasets.CIFAR10(root=self.args.datasetRoot,
                                                   train=False,
                                                   download=True,
                                                   transform=transform_test)
            # 根据索引获得数据   ？？？
            for i in range(self.size):
                partitions = partition.use(i)
                train_loader = torch.utils.data.DataLoader(partitions,
                                                           batch_size=self.args.bs,
                                                           shuffle=True,
                                                           pin_memory=False)
                train_loaders.append(train_loader)

                test_loader = torch.utils.data.DataLoader(testset,
                                                          batch_size=64,
                                                          shuffle=False)
                test_loaders.append(test_loader)

        return train_loaders, test_loaders
