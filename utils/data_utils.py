import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(batch_size=64, num_workers=0, data_root='./data'):
    """
    获取 CIFAR-10 训练集和测试集的 DataLoader，并为训练集应用数据增强。

    Args:
        batch_size (int): 每个批次的样本数量。
        num_workers (int): 用于数据加载的子进程数量。
        data_root (str): 数据集下载和存储的根目录。

    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    # 训练集的数据增强和标准化
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 随机裁剪到32x32，并在边缘填充4像素
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 将PIL图像转换为Tensor，并归一化到[0,1]
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)) # 标准化
    ])

    # 测试集的标准化 (不进行数据增强)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))
    ])

    # 下载并加载训练数据集
    train_dataset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_transform)
    # 下载并加载测试数据集
    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_transform)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # 定义类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes