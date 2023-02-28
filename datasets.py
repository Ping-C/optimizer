import numpy as np
from torch.utils.data import Dataset

from numpy.random import default_rng
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class Cutout:
    def __init__(self, size=16, p=0.5):
        self.size = size
        self.half_size = size // 2
        self.p = p

    def __call__(self, image):
        if torch.rand([1]).item() > self.p:
            return image

        left = torch.randint(-self.half_size, image.size(1) - self.half_size, [1]).item()
        top = torch.randint(-self.half_size, image.size(2) - self.half_size, [1]).item()
        right = min(image.size(1), left + self.size)
        bottom = min(image.size(2), top + self.size)

        image[:, max(0, left): right, max(0, top): bottom] = 0
        return image

class Cifar10:
    def __init__(self, batch_size, threads, aug='none', train_count=None, num_classes=2, seed=10):
        mean, std = self._get_statistics()
        torch.manual_seed(seed)
        if aug == "cutout":
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        elif aug == "none":
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        complete_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        complete_test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
        labels = torch.tensor(complete_train_set.targets)
        labels_test = torch.tensor(complete_test_set.targets)
        new_labels = -torch.ones_like(labels)
        new_labels_test = -torch.ones_like(labels_test)
        train_indices_list = []
        val_indices_list = []
        test_indices_list = []
        for i, cur_class in enumerate(torch.arange(10)[torch.randperm(10)][:num_classes]):
            indices_of_cur_class = torch.arange(50000)[labels == cur_class]
            new_labels[labels == cur_class] = i
            indices_len = len(indices_of_cur_class)
            indices_of_cur_class = indices_of_cur_class[torch.randperm(indices_len)]
            val_indices_list.append(indices_of_cur_class[:256//num_classes])
            train_indices_list.append(indices_of_cur_class[256//num_classes:256//num_classes+train_count//num_classes])

            indices_of_cur_class_test = torch.arange(10000)[labels_test == cur_class]
            new_labels_test[labels_test == cur_class] = i
            indices_len_test = len(indices_of_cur_class_test)
            indices_of_cur_class_test = indices_of_cur_class_test[torch.randperm(indices_len_test)]
            test_indices_list.append(indices_of_cur_class_test)



        complete_train_set.targets = new_labels
        complete_test_set.targets = new_labels_test
        val_indices = torch.cat(val_indices_list, dim=0)
        train_indices = torch.cat(train_indices_list, dim=0)
        test_indices = torch.cat(test_indices_list, dim=0)
        train_set = torch.utils.data.Subset(
            complete_train_set,
            train_indices
        )
        val_set = torch.utils.data.Subset(
            complete_train_set,
            val_indices
        )
        test_set = torch.utils.data.Subset(
            complete_test_set,
            test_indices
        )

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.val = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test_all_data, self.test_all_labels = zip(*[(x[None, :], y) for x, y in test_set])
        self.test_all_data = torch.cat(self.test_all_data, dim=0)
        self.test_all_labels = torch.tensor(self.test_all_labels)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class MNIST:
    def __init__(self, batch_size, threads, aug='none', train_count=None, num_classes=2, seed=10):
        mean, std = self._get_statistics()
        torch.manual_seed(seed)
        if aug == "none":
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        complete_train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        complete_test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

        labels = complete_train_set.targets
        labels_test = complete_test_set.targets
        new_labels = -torch.ones_like(labels)
        new_labels_test = -torch.ones_like(labels_test)
        train_indices_list = []
        val_indices_list = []
        test_indices_list = []
        for i, cur_class in enumerate(torch.arange(10)[torch.randperm(10)][:num_classes]):
            indices_of_cur_class = torch.arange(60000)[labels == cur_class]
            new_labels[labels == cur_class] = i
            indices_len = len(indices_of_cur_class)
            indices_of_cur_class = indices_of_cur_class[torch.randperm(indices_len)]
            val_indices_list.append(indices_of_cur_class[:256//num_classes])
            train_indices_list.append(indices_of_cur_class[256//num_classes:256//num_classes+train_count//num_classes])

            indices_of_cur_class_test = torch.arange(10000)[labels_test == cur_class]
            new_labels_test[labels_test == cur_class] = i
            indices_len_test = len(indices_of_cur_class_test)
            indices_of_cur_class_test = indices_of_cur_class_test[torch.randperm(indices_len_test)]
            test_indices_list.append(indices_of_cur_class_test)


        complete_train_set.targets = new_labels
        complete_test_set.targets = new_labels_test
        val_indices = torch.cat(val_indices_list, dim=0)
        train_indices = torch.cat(train_indices_list, dim=0)
        test_indices = torch.cat(test_indices_list, dim=0)
        train_set = torch.utils.data.Subset(
            complete_train_set,
            train_indices
        )
        val_set = torch.utils.data.Subset(
            complete_train_set,
            val_indices
        )
        test_set = torch.utils.data.Subset(
            complete_test_set,
            test_indices
        )

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.val = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test_all_data, self.test_all_labels = zip(*[(x[None, :], y) for x, y in test_set])
        self.test_all_data = torch.cat(self.test_all_data, dim=0)
        self.test_all_labels = torch.tensor(self.test_all_labels)

    def _get_statistics(self):
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])


class Cifar100:
    def __init__(self, batch_size, threads, aug):
        mean, std = self._get_statistics()

        if aug == "cutout":
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                Cutout()
            ])
        else:
            train_transform = transforms.Compose([
                torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
                torchvision.transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        self.train = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=threads)
        self.test = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=threads)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def _get_statistics(self):
        train_set = torchvision.datasets.CIFAR10(root='./cifar', train=True, download=True, transform=transforms.ToTensor())

        data = torch.cat([d[0] for d in DataLoader(train_set)])
        return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

class NArmSpiral(Dataset):
    """
    `torch.utils.data.Dataset` subclass for the NArmSpiral dataset
    `NArmSpiral` can be used to provide additional control over simply loading the .csv file
    directly. This class can be pass to a `torch.utils.data.DataLoader` object to iterate
    through the dataset. It also automatically separate the dataset into two parts: the test
    dataset and the train dataset. The train dataset takes 80% of the points for itself while
    the test dataset uses the last 20%.
    Parameters
    ----------
    filename: str
        .csv file containing the dataset.
    train: bool
        Specify of the dataset should contain training data or test data.
    Attributes
    ----------
    classes : list
        List of classes inside the dataset. Classes start at 0.
    data: ndarray
        Contains the points
    """
    def __init__(self, filename, train=True):
        self._file_data = np.loadtxt(filename, delimiter=';', dtype=np.float32)

        # Empty array to store the splices for training or test
        self.data = np.empty((0, self._file_data.shape[-1]), dtype=np.float32)

        # List of classes name and count for individual classes
        self.classes, _samples = np.unique(self._file_data[:, 2], return_counts=True)

        # We assume the classes have the same amount of samples
        self._sample_count = _samples[0]

        # Split the file data into array of each classe
        split_classes = np.split(self._file_data, len(self.classes))

        # Divide the classes into 80% training samples and 20% test samples
        part = int(self._sample_count * 0.8)

        for _class in split_classes:
            if train:
                self.data = np.concatenate((self.data, _class[:part, :]))
            else:
                self.data = np.concatenate((self.data, _class[part:, :]))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index, :2], self.data[index, 2].astype(np.int64)

class Kink(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30, noise = None):

        # Empty array to store the splices for training or test

        x0 = np.concatenate([np.linspace(-1, 0, 1000), np.linspace(0, 1, 1000)])
        y0 = np.concatenate([np.linspace(-1, 0, 1000), np.linspace(0, -1, 1000)])
        d0 = np.stack((x0,y0), axis=1)
        x1 = np.concatenate([np.linspace(-1, 0, 1000), np.linspace(0, 1, 1000)])
        y1 = np.concatenate([np.linspace(-1+margin, 0+margin, 1000), np.linspace(0+margin, -1+margin, 1000)])
        d1 = np.stack((x1, y1), axis=1)
        l0 = np.ones(2000)
        l1 = np.zeros(2000)

        self.labels = np.concatenate((l0, l1))
        self.data = np.concatenate((d0, d1))
        self.labels = default_rng(seed).permutation(self.labels)
        self.data = default_rng(seed).permutation(self.data)
        if noise:
            self.data += default_rng(seed).standard_normal(self.data.shape)
        train_sample_count = int(samples*0.7)
        test_sample_count = int(samples*0.3)
        if train:
            self.data = self.data[:train_sample_count]
            self.labels = self.labels[:train_sample_count]
        else:
            self.data = self.data[-test_sample_count:]
            self.labels = self.labels[-test_sample_count:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

class SemiCircle(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30):

        # Empty array to store the splices for training or test

        x0 = np.sin(np.linspace(-np.pi/2, np.pi/2, samples//2))
        y0 = np.cos(np.linspace(-np.pi/2, np.pi/2, samples//2))-margin/2
        d0 = np.stack((x0,y0), axis=1)
        x1 = np.sin(np.linspace(-np.pi/2, np.pi/2, samples//2))
        y1 = np.cos(np.linspace(-np.pi/2, np.pi/2, samples//2))+margin/2
        d1 = np.stack((x1, y1), axis=1)
        l0 = np.ones(samples//2)
        l1 = np.zeros(samples//2)

        self.labels = np.concatenate((l0, l1))
        self.data = np.concatenate((d0, d1))
        self.labels = default_rng(seed).permutation(self.labels)
        self.data = default_rng(seed).permutation(self.data)
        sample_count = int(samples*0.7)
        if train:
            self.data = self.data[:sample_count]
            self.labels = self.labels[:sample_count]
        else:
            self.data = self.data[sample_count:]
            self.labels = self.labels[sample_count:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

import numpy as np

def get_slab_data(num_slabs=3):
    # 4 slab dataset
    complex_margin = 0.1
    linear_margin = 0.1
    slab_thickness = (2-(num_slabs-1)*complex_margin*2)/num_slabs
    slab_end = 1
    slab_begin = 1-slab_thickness
    slab_sign = -1
    slab_data = []
    slab_labels = []
    for i in range(num_slabs):
        if slab_sign == -1:
            slab_current = np.concatenate([
                np.random.uniform(-1, 0-linear_margin, size=(400, 1)), 
                np.random.uniform(slab_begin, slab_end, size=(400, 1))], axis=1)
            slab_labels.append(np.ones(len(slab_current)))
        else:
            slab_current = np.concatenate([
                np.random.uniform(0+linear_margin, 1, size=(400, 1)), 
                np.random.uniform(slab_begin, slab_end, size=(400, 1))], axis=1)
            slab_labels.append(np.zeros(len(slab_current)))
        slab_data.append(slab_current)
        slab_begin -= slab_thickness + complex_margin * 2
        slab_end -= slab_thickness + complex_margin * 2
        slab_sign *= -1

    slab_data = np.concatenate(slab_data, axis=0)
    slab_labels = np.concatenate(slab_labels)
    return slab_data, slab_labels
    
def get_nonlinear_data(samples, num_slabs=3):
    # 4 slab dataset
    complex_margin = 0.1
    linear_margin = 0.1
    slab_thickness = (2-(num_slabs-1)*complex_margin*2)/num_slabs
    y_intersections = []
    y_intersection = -1+slab_thickness+complex_margin
    y_start = -1 + slab_thickness/2
    a_sign = (-1) ** num_slabs
    X = []
    for i in range(num_slabs-1):
        y_intersections.append(y_intersection)
        y_end = y_start + complex_margin * 2 + slab_thickness
        y_cur = np.linspace(y_start, y_end, 10)
        x_cur = (y_cur-y_intersection)*a_sign
        X_cur = np.concatenate([x_cur[:, None], y_cur[:, None]], axis=1)
        X.append(X_cur)
        y_intersection += complex_margin*2 + slab_thickness
        y_start = y_end
        a_sign *= -1
    X = np.concatenate(X, axis=0)
    shift = complex_margin * (2**0.5)
    X_nonlinear = np.concatenate([
        X + np.array([[shift, 0]]),
        X - np.array([[shift, 0]])], axis=0)
    Y_nonlinear = np.concatenate([
        np.zeros(len(X)),
        np.ones(len(X))
    ])

    return X_nonlinear, Y_nonlinear

def get_linear_data(samples):
    X2 = np.linspace(1, -1, samples)
    X1 = np.zeros(len(X2))

    shift = 0.1
    X_linear_c0 = np.concatenate([X1[:, None]+shift, X2[:, None]], axis=1)
    X_linear_c1 = np.concatenate([X1[:, None]-shift, X2[:, None]], axis=1)
    X_linear = np.concatenate([X_linear_c0, X_linear_c1], axis=0)
    Y_linear = np.concatenate([np.zeros(len(X1)), np.ones(len(X1))])
    return X_linear, Y_linear

class Slab(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30, noise = None):

        # Empty array to store the splices for training or test
        np.random.seed(seed=seed)
        self.data, self.labels = get_slab_data(num_slabs=4)
        shuffle_idx = np.random.permutation(np.arange(len(self.data)))
        self.data = self.data[shuffle_idx]
        self.labels = self.labels[shuffle_idx]
        train_sample_count = int(samples*0.7)
        test_sample_count = int(samples*0.3)
        if train:
            self.data = self.data[:train_sample_count]
            self.labels = self.labels[:train_sample_count]
        else:
            self.data = self.data[-test_sample_count:]
            self.labels = self.labels[-test_sample_count:]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

class SlabNonlinear4(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30, noise = None):
        np.random.seed(seed=seed)
        X, Y = get_nonlinear_data(samples=samples, num_slabs=4)
        self.data = X
        self.labels = Y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

class SlabLinear(Dataset):
    def __init__(self, train=True, margin=0.25, seed=100, samples=30, noise = None):
        np.random.seed(seed=seed)
        X, Y = get_linear_data(samples=samples)
        self.data = X
        self.labels = Y

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index].astype(np.float32), self.labels[index].astype(np.int64)

if __name__ == "__main__":
    # plotting slab datasets
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 1, figsize=(1*4, 1*4))
    axes = [axes]

    X, Y = get_slab_data(num_slabs=4)
    axes[0].scatter(X[Y==0, 0], X[Y==0, 1], color='g', alpha=0.2)
    axes[0].scatter(X[Y==1, 0], X[Y==1, 1], color='r', alpha=0.2)

    X, Y = get_nonlinear_data(samples=30, num_slabs=4)
    axes[0].scatter(X[Y==0, 0], X[Y==0, 1], color='g', alpha=0.2)
    axes[0].scatter(X[Y==1, 0], X[Y==1, 1], color='r', alpha=0.2)
    axes[0].set_axis_off()

    X, Y = get_linear_data(samples=30)
    axes[0].scatter(X[Y==0, 0], X[Y==0, 1], color='g', alpha=0.2)
    axes[0].scatter(X[Y==1, 0], X[Y==1, 1], color='r', alpha=0.2)
    axes[0].set_axis_off()
    
    plt.savefig('slab_dataset_nonlinear.png')