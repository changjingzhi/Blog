---
title: 论文思路——论文结构
date: 2024-05-11 09:29:04
tags: 论文思路
---
鉴于论文大修，所以我打算重新构建一下思路。
1. 标题
2. 摘要 （介绍基础背景，解决什么问题，做了什么工作。）
3. 关键词

1. 介绍
2. 数据和方法
3. 结果
4. 讨论
5. 参考文献


## 论文选题方向
1. 抑郁症分类 （已经尝试过了）
2. 情绪检测，使用 DEAP数据集和SEED （代码和论文全套）
3. 新开分类——基于公开数据集来做实验
4. 扩大脑电数据集-训练模型

## 现在有的论文思路
1. 抑郁症分类
2. AD-CN——FDT公开数据集优化问题(使用MateFormer来验证扩大数据集的思路)
3. 研究公开数据集的论文，进行更改调优。

## 基础网站

[医学数据获取网站](https://openneuro.org/)
[代码寻找网站](https://paperswithcode.com/)
[论文查找网站](https://ac.scmor.com/)

## 论文待办

1. 证明时间剪切长度的优良性
2. 编写自动化代码，在十则交叉验证中.第一，训练代码要保存loss和acc，使用早停。(深度学习自动化训练)，加入日志功能，封装训练代码

自动化代码
```

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import EEGDataset, transform1,transform3
from net import IntegratedNet
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_identityformer_model(model, model_name, num_epochs=100, num_classes=3, batch_size=8, learning_rate=0.0005, w_wight=2560, chennal=32,load =False):
    # Checking CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    m = nn.Softmax(dim=1)  # 只对样本的维度做softmax
    # Creating datasets and data loaders
    train_dataset = EEGDataset(csv_file='train_data.csv', transform=transform1)
    test_dataset = EEGDataset(csv_file='test_data.csv', transform=transform1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    save_dir = os.path.join('model', model_name)
    os.makedirs(save_dir, exist_ok=True)

    if load == True:
        model.load_state_dict(torch.load('三分类预训练模型-0.99.pth'))
        # num_ftrs = model.final_linear.in_features
        # model.final_linear = nn.Linear(num_ftrs,2)
        # model.to(device)
    best_val_acc = 0
    best_model_path = os.path.join(save_dir, "{}_best_model.pth".format(model_name))

    # 训练
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []


    early_stop = False
    patience = 5  # 容忍次数，超过这个次数准确率不再提升就停止
    counter = 0

    for epoch in range(num_epochs):
        train_loss_total = 0  # 所有batch的loss累加值
        train_acc_total = 0   # 所有batch的acc累加值`
        val_loss_total = 0
        val_acc_total = 0

        model.train()    # 标志模型的模式是什么，因为dropout只在训练时启用
        for i, (train_x, train_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            train_x = train_x.to(device)
            # print('train_x shape',train_x.shape)
            train_y = train_y.to(device)
            train_x = train_x.unsqueeze(1)
            train_x = train_x.view(batch_size, 1, chennal, w_wight)
            # 前向传播
            train_y_pred = model(train_x)
            train_loss = loss_fn(train_y_pred, train_y)

            # 通过模型每个样本得到4个实数值（train_y_pred）,通过softmax将实数值转换成概率值，通过max取概率最大的下标，最后用下标和标签做比较
            train_acc = (m(train_y_pred).max(dim=1)[1] == train_y).sum()/train_y.shape[0]
            train_loss_total += train_loss.data.item()
            train_acc_total += train_acc.data.item()
            # 反向传播
            train_loss.backward()
            # 梯度下降
            optimizer.step()
            optimizer.zero_grad()

            # print("epoch:{} train_loss:{} train_acc:{}".format(epoch, train_loss.data.item(), train_acc.data.item()))

        train_loss_arr.append(train_loss_total / len(train_loader))  # 平均值
        train_acc_arr.append(train_acc_total / len(train_loader))
        print("epoch:{} train_loss:{} train_acc:{}".format(epoch, train_loss_arr[-1], train_acc_arr[-1]))
        

       
        # 测试集
        model.eval()
        for j, (val_x, val_y) in enumerate(test_loader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_x = val_x.unsqueeze(1)
            val_x = val_x.view(batch_size, 1, chennal, w_wight)
            # 前向传播
            val_y_pred = model(val_x)
            val_loss = loss_fn(val_y_pred, val_y)
            val_acc = (m(val_y_pred).max(dim=1)[1] == val_y).sum()/val_y.shape[0]
            val_loss_total += val_loss.data.item()
            val_acc_total += val_acc.data.item()

        val_loss_arr.append(val_loss_total / len(test_loader)) # 平均值
        val_acc_arr.append(val_acc_total / len(test_loader))
        print("epoch:{} val_loss:{} val_acc:{}".format(epoch, val_loss_arr[-1], val_acc_arr[-1]))

        logging.info(f"Epoch {epoch}: Train Loss: {train_loss_arr[-1]}, Train Acc: {train_acc_arr[-1]}, Val Loss: {val_loss_arr[-1]}, Val Acc: {val_acc_arr[-1]}")
        # 保存最佳模型
        if val_acc_arr[-1] > best_val_acc:
            best_val_acc = val_acc_arr[-1]
            torch.save(model.state_dict(), best_model_path)
            print("保存模型成功!")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                early_stop = True
                break

        # 保存训练和验证过程中的loss和acc
        np.save(os.path.join(save_dir, f"{model_name}_train_loss.npy"), np.array(train_loss_arr))
        np.save(os.path.join(save_dir, f"{model_name}_train_acc.npy"), np.array(train_acc_arr))
        np.save(os.path.join(save_dir, f"{model_name}_val_loss.npy"), np.array(val_loss_arr))
        np.save(os.path.join(save_dir, f"{model_name}_val_acc.npy"), np.array(val_acc_arr))


        if early_stop:
            break
   
    print('Training completed!')


# 创建模型
model = IntegratedNet(input_size=1, in_feature=320, num_classes=3)

# 训练参数
num_epochs = 100
num_classes = 3
batch_sizes = [8,16,16,16]
learning_rates = [0.0005, 0.0005, 0.0005, 0.0005]
chennal = 11
w_wight = 5120

# 在训练前加入日志记录
logging.info("Start training")

# 多次训练
for i, (learning_rate, batch_size) in enumerate(zip(learning_rates, batch_sizes)):
    # 生成当前训练的 model_name
    model_name = f"混合数据集-第{i + 1}次-AD-CN-MCI-10秒-learning-{learning_rate}bitch-{batch_size}"
    logging.info(f"Training parameters - Model: {model_name}, Num Epochs: {num_epochs}, Num Classes: {num_classes}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Chennal: {chennal}, W Weight: {w_wight}")
    # 创建并训练模型
    train_identityformer_model(model, model_name=model_name, num_epochs=num_epochs, num_classes=num_classes, batch_size=batch_size, learning_rate=learning_rate, chennal=chennal, w_wight=w_wight, load=False)

# 训练结束后记录日志
logging.info("Training completed")
```

## 防止模型过拟合的方法
过拟合: 

根据数据集的简单和复杂的程度来选取对应的模型容量

模型容量： ·拟合各种函数的能力 ·低容量的模型难以拟合训练数据·高容量的模型可以记住所有的训练数据
评估模型容量的两个指标： 参数的个数，参数值的选择范围。
![数据复杂时应该选择复杂的模型](pic/paper-idear5-1.png)
![模型容量和误差之间的关系](pic/paper-idear5-2.png)
深度学习的核心任务将泛化误差往下降低。

了解数据的复杂度： 1. 样本个数。 2.  每个样本的元素个数。 3. 样本的时间、空间结构。 4. 样本的多样性。
VC维：对于一个分类模型，VC等于一个最大的数据集的大小，不管如何给定标号，都存在一个模型来对它进行完美分类
·模型容量需要匹配数据复杂度，否则可能导致欠拟合和过拟合
·统计机器学习提供数学工具来衡量模型复杂度·实际中一般靠观察训练误差和验证误差

神经网络是一种语言，一种使用各个层来解释我的数据的语言，
### 解决拟合的思路。
1. 数据集划分
2. 正则化 (L1,l2正则化。)
3. 增加训练数据
4. 特征选择
5. 交叉验证
6. 提前停止 (Early stopping)
7. Dropout
8. 网络结构 Architecture （模型容量） 特征提取器更换，分类器（SVM）
9. 限制权值/权重衰减 weight-decay
10. 增加噪声 Noise
11. 数据增强 （空间特征 + 时间特征）
