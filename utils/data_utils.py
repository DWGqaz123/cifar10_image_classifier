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

def get_cifar10_dataloaders_for_transfer_learning(batch_size=64, num_workers=0, data_root='./data'):
    """
    获取 CIFAR-10 训练集和测试集的 DataLoader，针对迁移学习进行预处理。
    图像将被调整大小和标准化，以匹配预训练模型的期望输入 (ResNet通常期望 224x224)。

    Args:
        batch_size (int): 每个批次的样本数量。
        num_workers (int): 用于数据加载的子进程数量。
        data_root (str): 数据集下载和存储的根目录。

    Returns:
        tuple: (train_loader, test_loader, classes)
    """
    # ResNet 等预训练模型通常期望 224x224 的输入尺寸
    # ImageNet 的均值和标准差 (因为预训练模型在 ImageNet 上训练)
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    # 训练集的数据增强和标准化
    train_transform = transforms.Compose([
        transforms.Resize(256), # 先缩放图像，使其较短边为 256
        transforms.RandomCrop(224), # 然后随机裁剪 224x224
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.ToTensor(), # 将PIL图像转换为Tensor，并归一化到[0,1]
        transforms.Normalize(imagenet_mean, imagenet_std) # 使用 ImageNet 的均值和标准差标准化
    ])

    # 测试集的标准化 (不进行数据增强，但同样调整尺寸和标准化)
    test_transform = transforms.Compose([
        transforms.Resize(256), # 缩放
        transforms.CenterCrop(224), # 中心裁剪 224x224
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std)
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