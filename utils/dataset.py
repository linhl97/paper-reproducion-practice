import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


#mean and std of cifar100 dataset
CIFAR100_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

def cifar100_dataloader(mean, std, batch_size=16, num_workers=4, shuffle=True):
    """ return cifar100 dataloader
    Args:
        mean: mean of cifar100 dataset
        std: std of cifar100 dataset
    Returns: cifar100 trainloader and testloader
    """

    transforms_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
    ])
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transforms_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transforms_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

    return train_dataloader, test_dataloader

if __name__ == '__main__':
    cifar = torchvision.datasets.CIFAR100(root='./data', train=False, download=False)