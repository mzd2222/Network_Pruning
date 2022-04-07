from torch.utils.data import dataloader
from torchvision import datasets, transforms


class Data_Loader_CIFAR:

    def __init__(self, train_batch_size, test_batch_size, dataSet='CIFAR10', data_path='./data/', download=False):
        if dataSet == 'CIFAR10':
            mean = [0.4940607, 0.4850613, 0.45037037]
            std = [0.20085774, 0.19870903, 0.20153421]
            self.dataset_num_class = 10
        elif dataSet == 'CIFAR100':
            mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
            std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
            self.dataset_num_class = 100

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        if dataSet == 'CIFAR100':
            train_data_set = datasets.CIFAR100(data_path, transform=train_transform, download=download, train=True)
            test_data_set = datasets.CIFAR100(data_path, transform=test_transform, download=download, train=False)

        elif dataSet == 'CIFAR10':
            train_data_set = datasets.CIFAR10(data_path, transform=train_transform, download=download, train=True)
            test_data_set = datasets.CIFAR10(data_path, transform=test_transform, download=download, train=False)

        self.train_data_loader = dataloader.DataLoader(train_data_set, batch_size=train_batch_size, shuffle=False)
        self.test_data_loader = dataloader.DataLoader(test_data_set, batch_size=test_batch_size, shuffle=True)



