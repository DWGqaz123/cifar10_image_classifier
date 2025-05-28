# CIFAR-10 Image Classifier (PyTorch)

This project implements an image classification model using PyTorch to recognize the 10 categories in the CIFAR-10 dataset. It applies key techniques including data augmentation, dropout, learning rate scheduling, weight decay, convolutional block expansion, and transfer learning.

---

## 📦 Dataset

- **CIFAR-10:** 60,000 32×32 color images (50,000 train + 10,000 test), 10 classes.

---

## 🧪 Key Techniques

- Convolutional Neural Network (CNN)  
- Data Augmentation (random crop, flip)  
- Dropout  
- Learning Rate Scheduler  
- Weight Decay  
- Deeper CNN Architecture  
- Transfer Learning (ResNet18)

---

## ⚙️ Environment

- **Device:** MacBook Pro (M2) with MPS acceleration  
- **Framework:** PyTorch 2.7.0  
- **Reproducibility:** Random seed `SEED = 66`

---

## 📊 Best Results Summary

| Method                     | Train Acc | Test Acc |
|---------------------------|-----------|----------|
| Base Model (10 Epochs)    | 73.44%    | 75.66%   |
| + Data Augmentation       | 80.57%    | 80.18%   |
| + Dropout + Scheduler     | 77.16%    | 79.87%   |
| + Weight Decay (1e-5)     | 81.16%    | 81.77%   |
| + 3 Conv Blocks           | 91.37%    | 85.44%   |
| + ResNet18 Fine-Tuned     | **99.71%**| **95.54%**|

---

## 📘 中文版

# CIFAR-10 图像分类器（PyTorch）

本项目基于 PyTorch 实现图像分类模型，用于识别 CIFAR-10 数据集中 10 个类别。通过引入数据增强、Dropout、学习率调度、权重衰减、卷积结构扩展及迁移学习等技术，逐步优化模型性能。

---

## 📦 数据集

- **CIFAR-10：** 共 60,000 张 32×32 彩色图像（50,000 训练 + 10,000 测试），共 10 类。

---

## 🔧 核心技术

- 卷积神经网络（CNN）  
- 数据增强（随机裁剪与翻转）  
- Dropout 正则化  
- 学习率调度器  
- 权重衰减  
- 卷积块拓展  
- 迁移学习（ResNet18）

---

## ⚙️ 实验环境

- **设备：** MacBook Pro (M2)，支持 MPS 加速  
- **框架：** PyTorch 2.7.0  
- **复现性：** 固定随机种子 `SEED = 66`

---

## 📊 最佳结果摘要

| 方法                     | 训练准确率 | 测试准确率 |
|--------------------------|------------|------------|
| 初始模型（10轮）         | 73.44%     | 75.66%     |
| + 数据增强               | 80.57%     | 80.18%     |
| + Dropout + 调度器       | 77.16%     | 79.87%     |
| + 权重衰减 (1e-5)        | 81.16%     | 81.77%     |
| + 三层卷积结构           | 91.37%     | 85.44%     |
| + ResNet18 微调          | **99.71%** | **95.54%** |
