import torch
import torch.nn as nn
import torchvision.models as models

class ResNetTransferModel(nn.Module):
    def __init__(self, num_classes=10, freeze_features=False):
        super(ResNetTransferModel, self).__init__()
        
        # 1. 加载预训练的 ResNet18 模型
        # pretrained=True 会下载ImageNet上的预训练权重
        self.resnet = models.resnet18(pretrained=True)
        
        # 2. (可选) 冻结特征提取层
        # 如果 freeze_features 为 True，则冻结除了分类层之外的所有参数
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # 3. 替换 ResNet 的全连接层 (分类头)
        # 获取原来 fc 层的输入特征数
        num_ftrs = self.resnet.fc.in_features
        # 替换为新的线性层，输出 num_classes 个类别
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)