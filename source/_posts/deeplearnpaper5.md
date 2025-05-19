---
title: 经典网络结构-shuffleNet
date: 2024-04-30 18:58:54
tags: 深度学习论文
---
[ShuffleNet论文参考](https://zhuanlan.zhihu.com/p/32304419)
[ShuffleNet的论文原文](https://arxiv.org/pdf/1707.01083)

## 实践使用shuffleNet来实现垃圾的40分类

### 划分固定数据集
在这里划分固定数据集，生成两个csv表，一个是训练集，一个是验证集
```
import os
import csv
import numpy as np
train_path = "train_data.csv"
val_path = "val_data.csv"

train_percent = 0.9

def create_data_txt(path):
    f_train = open(train_path,"w",newline="")
    f_val = open(val_path,"w",newline="")
    train_writer = csv.writer(f_train)
    val_writer = csv.writer(f_val)

    for cls,dirname in enumerate(os.listdir(path)):
        flist = os.listdir(os.path.join(path,dirname))
        np.random.shuffle(flist)
        fnum = len(flist)
        for i,filename in enumerate(flist):
            if i < fnum*train_percent:
                train_writer.writerow([os.path.join(path,dirname,filename),str(cls)])
            else:
                val_writer.writerow([os.path.join(path, dirname, filename), str(cls)])

    f_train.close()
    f_val.close()


if __name__ == "__main__":
    create_data_txt("data_garbage")
```
### dataset 设置。
在这里设置数据预处理的操作
```
import torch
from PIL import Image
from torchvision import transforms,utils
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

train_tf = transforms.Compose([
    # transforms.RandomResizedCrop(size=(224,224), scale=(0.9,1.1)),
    transforms.Resize(224),
    transforms.CenterCrop((224,224)),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=(0.9,1.1),contrast=(0.9,1.1)),
    # transforms.Resize((50,50)),
    transforms.ToTensor(),
])

val_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop((224, 224)),
    # transforms.Grayscale(1),
    transforms.ToTensor(),
])

#自定义数据集
class Animals_dataset(Dataset):
    def __init__(self,istrain=True):
        if istrain:
            f = open("train_data.csv", "r")
        else:
            f = open("val_data.csv", "r")
        self.dataset = f.readlines()
        f.close()
        self.istrain = istrain

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        img_path = data.split(",")[0]
        cls = int(data.split(",")[1])

        img_data = Image.open(img_path).convert("RGB")
        if self.istrain:
            dst = train_tf(img_data)
        else:
            dst =val_tf(img_data)

        return dst,torch.tensor(cls)

def visulization():
    train_dataset = Animals_dataset(True)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    examples = enumerate(train_dataloader)
    batch_index,(data, lable) = next(examples)
    print(data.shape)

    grid = utils.make_grid(data)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.show()

if __name__ == "__main__":
    visulization()
```
### 训练代码
```
import torch
from torch import optim,nn
from torch.utils.data import DataLoader
from dataset import *
from torchvision import models
from matplotlib import pyplot as plt

m = nn.Softmax(dim=1)
def train(method="normal",ckpt_path=""):
    # 数据集和数据加载器
    train_dataset = Animals_dataset(True)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataset = Animals_dataset(False)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    #模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#系统自己决定有啥训练
    if method=="normal":
       # 创建ShuffleNet模型
        model = models.shufflenet_v2_x1_0(pretrained=True)  # 使用预训练的ShuffleNetV2模型

        # 修改最后的全连接层以适应您的数据集
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,40)  # 将全连接层输出维度修改为您数据集的类别数
        model.to(device)
    print("train on ",device)
    #损失函数（二分类交叉熵）
    loss_fn = nn.CrossEntropyLoss()

    #优化器
    optimizer = optim.RMSprop(model.parameters(),lr=0.0001)

    #断点恢复
    start_epoch = 0
    if ckpt_path != "":
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1

    #训练
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    for epoch in range(30):
        train_loss_total = 0
        train_acc_total = 0
        val_loss_total = 0
        val_acc_total = 0
        model.train()
        for i,(train_x,train_y) in enumerate(train_dataloader):
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            train_y_pred = model(train_x)
            train_loss = loss_fn(train_y_pred.squeeze(),train_y)
            train_acc = (m(train_y_pred).max(dim=1)[1] == train_y).sum()/train_y.shape[0]
            train_loss_total += train_loss.data.item()
            train_acc_total += train_acc.data.item()

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print("epoch:{} train_loss: {} train_acc: {}".format(epoch,train_loss.data.item(),train_acc.data.item()))
        
        train_loss_arr.append(train_loss_total / len(train_dataloader))
        train_acc_arr.append(train_acc_total / len(train_dataloader))

        model.eval()

        for j, (val_x,val_y) in enumerate(val_dataloader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)

            val_y_pred = model(val_x)
            val_loss = loss_fn(val_y_pred.squeeze(),val_y)
            val_acc = (m(val_y_pred).max(dim=1)[1]==val_y).sum()/val_y.shape[0]
            val_loss_total += val_loss.data.item()
            val_acc_total += val_acc.data.item()

        val_loss_arr.append(val_loss_total / len(val_dataloader))  # 平均值
        val_acc_arr.append(val_acc_total / len(val_dataloader))
        print("epoch:{} val_loss:{} val_acc:{}".format(epoch, val_loss_arr[-1], val_acc_arr[-1]))

    plt.subplot(1, 2, 1)   # 画布一分为二,1行2列，用第一个
    plt.title("loss")
    plt.plot(train_loss_arr, "r", label="train")
    plt.plot(val_loss_arr, "b", label="val")
    plt.legend()

    plt.subplot(1, 2, 2)  # 画布一分为二,1行2列，用第一个
    plt.title("acc")
    plt.plot(train_acc_arr, "r", label="train")
    plt.plot(val_acc_arr, "b", label="val")
    plt.legend()
    plt.savefig("loss_acc-1.png")

    plt.show()

    # 保存模型
    # 1.torch.save()
    # 2.文件的后缀名：.pt、.pth、.pkl
    torch.save(model.state_dict(), r"shuffeNet.pth")
    print("保存模型成功!")



if __name__ == "__main__":
    train()


    train()


```


## 挖坑
### 什么是断点训练
