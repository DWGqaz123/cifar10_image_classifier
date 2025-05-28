import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # 第一个卷积块: 输入3个通道 (RGB), 输出32个特征图, 卷积核5x5, padding=2
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16

        # 第二个卷积块: 输入32个特征图, 输出64个特征图, 卷积核5x5, padding=2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8

        # 第三个卷积块 ：输入64个特征图, 输出128个特征图, 卷积核5x5, padding=2
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
        # -------------------------

        # 全连接层
        # 重新计算输入到全连接层的特征图尺寸
        # 原始图像: 32x32
        # 经过 pool1 (2x2, stride=2): 16x16
        # 经过 pool2 (2x2, stride=2): 8x8
        # 经过 pool3 (2x2, stride=2): 4x4
        # 第三个卷积层输出128个特征图，所以展平后是 128 * 4 * 4 = 2048 个特征
        self.fc1 = nn.Linear(in_features=128 * 4 * 4, out_features=128) # 全连接层输入维度调整
        self.dropout = nn.Dropout(p=0.15) # 保持你之前的最佳 Dropout 概率
        self.fc2 = nn.Linear(in_features=128, out_features=10) # 10个类别输出

    def forward(self, x):
        # 卷积块1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        # 卷积块2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # --- 新增的第三个卷积块的前向传播 ---
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        # -----------------------------------

        # 展平特征图，准备输入全连接层
        x = x.view(x.size(0), -1) 

        # 全连接层及 Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = self.fc2(x)
        return out