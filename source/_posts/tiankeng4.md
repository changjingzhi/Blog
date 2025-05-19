---
title: 填坑——软编码
date: 2024-04-26 12:56:01
tags: 填坑
---

## 背景介绍
今天想重新复写一下模型结构，让模型能够自适应的去适应在不同的数据集，实现软编码。
首先介绍一下硬编码
[参考博客](https://blog.csdn.net/weixin_44943389/article/details/134928228)
硬编码是指将具体的数值、路径、参数等直接写入程序代码中，而不通过变量或配置文件来表示。这样的做法使得程序中的这些数值和参数变得固定，不容易修改，且缺乏灵活性。硬编码的值通常被称为"魔法数"（Magic Numbers）或"魔法字符串"，因为它们没有直观的含义，只能通过查看代码来了解。
例如，以下是一个硬编码的示例，其中数值 10 直接出现在代码中：
```
for i in range(10):
    print("Iteration", i)
```
软编码（Softcoding）：

软编码是指通过变量、配置文件、参数等方式将具体数值或参数抽象出来，而不是直接写入代码。通过软编码，程序变得更加灵活，可以更容易地进行修改和维护，且适应性更强。

使用软编码的例子：
```
iterations = 10
for i in range(iterations):
    print("Iteration", i)
```
硬编码：将具体数值、参数等直接写入程序代码中，缺乏灵活性，不易修改和维护。

软编码：通过变量、配置文件等方式将数值或参数抽象出来，使得程序更具灵活性，易于修改和维护。

## 我的解决办法

1. 在foward中进行重新赋值（有问题），问题就是没有前向训练过程中都重新创建了一个linear，这个linear层的参数没有训练，相当于随机（注：只是我猜的，没有验证，挖个坑在这）
```
import torch
import torch.nn as nn

class IntegratedNet(nn.Module):
    def __init__(self):
        super(IntegratedNet, self).__init__()
        self.linear = None # 在初始类中先第一一个self.linear=None ，而后在forward中重新定义

    def forward(self, x):
        if self.linear is None:
            self.linear = nn.Linear(in_features=x.size(1), out_features=1)
        
        x = self.linear(x)
        return x

# 创建模型实例
model = IntegratedNet()

# 创建输入数据
x = torch.randn(2, 512, 64)

# 前向传播
output = model(x)

# 打印输出的形状
print(output.size())

```

2. 第二种解决方法在__init__中留下一个接口，在调用这个模型时直接重新赋值。
```
class IntegratedNet(nn.Module):
    def __init__(self, input_size=3, mlp_dim=512, mlp_ratio=4,
                 dims=[64, 128, 320, 512],in_feature=64):
        super(IntegratedNet, self).__init__()

model = IntegratedNet(input_size=2,in_feature=209)  # 全部重新赋值，实现
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 打印模型结构摘要
summary(model, (2, 33, 3333))

```


3. 重新构建网络结构，加入自适应池化层。
