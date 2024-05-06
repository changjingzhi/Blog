---
title: Pytorch基础
date: 2024-04-27 21:43:19
tags:
---
参考书目：《Python深度学习基于PyTorch》
## Pytorch 基础
Pytorch采用python语言接口实现编程，非常容易上手。它就像带GPU的Numpy，与Python一样都属于动态框架。PyTorch继承了Torch灵活、动态的编程环境和用户友好的界面，支持以快速灵活的方式构建动态神经网络，还允许在训练过程中快速更改代码而不妨碍其性能，支持动态图形等尖端AI模型的能力，是快速实验的理想选择。

## 背景介绍
Pytorch是建立在Torch库上的python包，目的在于加速深度学习的应用。它包含了多维张量的数据结构以及基于其上的多种数学操作。动态计算图
PyTorch 主要由4个包组成：
torch：类似于Numpy的通用数组库，可将张量类型转换为torch.cuda.Tensor.Float，并在GPU上进行计算。
torch.autograd: 用于构建计算图形并自动获取梯度的包
torch.nn: 具有共享层和损失函数的神经网络库。
torch.ptim： 具有通用优化算法（如SGD，Adam等）的优化包

## 安装配置
pytorch的GPU版本 的安装配置有一点繁琐，这里阐述一下需要的装备
1. 电脑（装有显卡，本台电脑是GTX-3080），安装GPU的驱动（如英伟达的NVDIA）以及CUDA，cuDNN计算框架。安装GPU驱动的时候就会安装CUDA，cuDNN的安装要去官网查找对应版本。
2. 软件miniconda ，python的环境包管理
3. VS_code

## Numpy和Tensor
前面说到深度学习的最主要的东西是矩阵，深度学习就是一个大的函数。Tensor是numpy的Pytorch中的实现(这么说不知道行不行，但我是这么认为的)，pytorch中的Tensor可以是零维（又称为一个标量或一个数）、一维、二维以及多维数组。Tensor可以把产生的Tensor放置在GPU中进行加速计算。
对Tensor的操作很多，从接口的角度可以划分为两类，
torch.function 如 torch.sum、torch.add
tensor.function ，如tensor.view、tensor.add等
这些操作对于大部分Tensor都是等价的，比如torch.add(x)与x.add(y)等价（注：前提是x的dtype是tensor）

```
import torch
x = torch.tensor([1 , 2])
y =torch.tensor([3,4])
z = x.add(y)
print(z)
print(x)
x.add_(y)
print(x)
```
Tensor创建的方式
```
# Tensor(*size) 直接从参数构建一个张量
```
## 挖坑

### 仍然没有搞明白使用梯度下降算法来怎么对矩阵中的参数进行更新的方法？
目前网络上的梯度下降多使用线性函数来进行距离，没有推导过程，而我又不会推导，死循环了。所以我想知道怎么使用矩阵来实现梯度下降，进而实现SGD，Adam，Rsomp等梯度下降优化函数。

