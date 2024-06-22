---
title: 论文思路
date: 2024-04-25 12:16:48
tags: 论文思路
---

## 数据预处理
1. 傅里叶变换 （时间域变换为频域）
2. 归一化（零归一化，批归一化，层归一化）
3. 数据增强操作。
guding_channl/AD -> 训练集大小: 65
guding_channl/AD -> 测试集大小: 7
guding_channl/CN -> 训练集大小: 95
guding_channl/CN -> 测试集大小: 10
guding_channl/MCI -> 训练集大小: 91
guding_channl/MCI -> 测试集大小: 10

## 激活函数
1. startRule
2. ReLU

## 模型架构
1. 编码器——解码器
2. 残差连接（基于ResNet）
3. MLP
4. 双输入卷积神经网络。


## 损失函数
1. 交叉熵损失函数


## 优化函数
1. SGD
2. RMSprop

## 评估指标
怎么进行评估，
![二分类示意图](pic/predict.jpg)
True Positives (TP)：正类别样本中被正确预测为正类别的数量。True Negatives (TN)：负类别样本中被正确预测为负类别的数量。False Positives (FP)：负类别样本中被错误预测为正类别的数量。False Negatives (FN)：正类别样本中被错误预测为负类别的数量。
1. ACC （准确率） ( TP+TN )  / (TP+TN+FP+FN)
2. pression (精确率) (TP/TP+FP) 
3. Recall (Sensitivity，灵敏度) (TP / TP+FN )
4. F1-score (F1 值) （2 x (precision x Recall)/(precision + Recall )）
5. Specificity (特异性) （TN / (TN + FP )）
6. 混淆矩阵


## 数据输入
1. 双输入卷积神经网络 （傅里叶信号+归一化信号）


## 实验补充
1. 横向实验：按照8：2的数据划分来进行验证。
2. 做消融实验，三个（原模型，删去） 
3. 验证不同的时间长度的正确性。


## 创新点
1. 双数据输入，对数据采用不同的数据预处理，比如FFT(傅里叶变换)，小波变换，零归一化.
2. 在公开数据集上验证处理不同的数据预处理方法对实验结果的好坏。
3. 对比私有数据集，验证自己模型的稳健性（鲁棒性）。
4. 使用图像处理的模型结构。
5. 公开数据集的二分类结果，对比论文。
6. 验证选取合适的数据预处理方法是合适的。(注：数据预处理+深度学习模型架构。)

