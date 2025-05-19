---
title: 经典网络结构——VGG
date: 2024-04-22 18:55:02
tags: 深度学习论文
---

## 参考文献
2014年
[中英文对照翻译](https://blog.csdn.net/C_chuxin/article/details/82833070)
[VGG论文解读](https://zhuanlan.zhihu.com/p/460777014)
[原文](https://arxiv.org/pdf/1409.1556)
[VGG论文解读](https://zhuanlan.zhihu.com/p/107884876)
## 背景介绍
VGG是牛津大学的Visual Geometry Group的团队在ILSVRC 2014上的相关工作。在这项工作中，主要研究卷积网络深度对大规模图像识别准确率的影响。其主要的贡献是对使用非常小的卷积滤波器（3 X 3）的体系架构来增加网络深度进行彻底的评估。实验结果表明将网络的深度提升至16-19个权重层可以实现对现有技术的显著改进。其在2014年的 imageNet 大规模视觉挑战赛（ILSVRC - 2014）中取得亚军。（冠军是 GoogleNet，预告下一篇是GoogleNet）

## VGG原理
VGG原理
相比于 LeNet 网络，VGG 网络的一个改进点是将 大尺寸的卷积核 用 多个小尺寸的卷积核 代替。

比如：VGG使用 2个3X3的卷积核 来代替 5X5的卷积核，3个3X3的卷积核 代替7X7的卷积核。

这样做的好处是：

1. 在保证相同感受野的情况下，多个小卷积层堆积可以提升网络深度，增加特征提取能力（非线性层增加）。
2. 参数更少。比如 1个大小为5的感受野 等价于 2个步长为1，3X3大小的卷积核堆叠。（即1个5X5的卷积核等于2个3X3的卷积核）。而1个5X5卷积核的参数量为 5*5*C^2。而2个3X3卷积核的参数量为 2*3*3*C^2。很显然，18C^2 < 25C^2。
3. 3X3卷积核更有利于保持图像性质。

VGG缺点：

VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。
注：这里参数量的计算，忽略了偏置。并且假设 输入和输出通道数都为C。


## 模型结构
VGGNet以下6种不同结构，我们以通常所说的VGG-16(即下图D列)为例，展示其结构示意图
![VGG_6种模型结构](pic/VGG.png)
![VGG_16模型结构图](pic/VGG_16.png)
![VGG参数图](pic/VGG_16_chanshu.png)
![VGG16参数图](pic/VGG_chanshu.png)

## 摘要 Abstract

In this work we investigate the effect of the convolutional network depth on its accuracy in the large-scale image recognition setting. Ourmain contribution is a thorough evaluation of networks of increasing depth usingan architecture withvery small (3×3) convolution filters, which shows that a significant improvementon the prior-art configurations can be achieved by pushing the depth to 16–19 weight layers. These findings were the basis of our ImageNet Challenge 2014 submission, where our team secured the first and the second places in the localisa-tion and classification tracks respectively. We also show that our representations generalise well to other datasets, where they achieve state-of-the-art results. Wehave made our two best-performing ConvNet models publicly available to facili-tate further research on the use of deep visual representations in computer vision.


在这项工作中，我们研究了卷积网络深度对其在大规模图像识别设置中的准确性的影响。我们的主要贡献是使用一个非常小的(3×3)卷积filter的架构对增加深度的网络进行了彻底的评估，这表明通过将深度提升到16 - 19个weight层，可以显著改善先前的配置。这些发现是我们提交ImageNet挑战赛2014的基础，我们的团队分别获得了本地化和分类的第一名和第二名。我们还展示了我们的成果可以很好地推广到其他数据集，在这些数据集上他们可以得到最优结果。我们已经公开了两个性能最好的卷积神经网络模型，以促进在计算机视觉中使用深度视觉表示的进一步研究。

## 训练

训练和之前的AlexNet整体类似，使用小批量梯度下降，参数方面：batch设为256，动量设为0.9，除最后一层外的全连接层也都使用了丢弃率0.5的dropout。learning rate最初设为0.01,权重衰减系数为5×10^-4。对于权重层采用了随机初始化，初始化为均值0，方差0.01的正态分布。 训练的图像数据方面，为了增加数据集，和AlexNet一样，这里也采用了随机水平翻转和随机RGB色差进行数据扩增。对经过重新缩放的图片随机排序并进行随机剪裁得到固定尺寸大小为224×224的训练图像。

## 训练结果
![对比结果](pic/VGG_train_result.webp)
通过表格间各个网络的对比发现如下结论：

总体来说卷积网络越深，损失越小，效果越好。
C优于B，表明多增加的非线性relu有效
D优于C，表明了卷积层filter对于捕捉空间特征有帮助。
E深度达到19层后达到了损失的最低点，但是对于其他更大型的数据集来说，可能更深的模型效果更好。
B和同类型filter size为5×5的网络进行了对比，发现其top-1错误率比B高7%，表明小尺寸filter效果更好。
在训练中，采用浮动尺度效果更好，因为这有助于学习分类目标在不同尺寸下的特征。
## 挖坑

### 使用VGG来实现垃圾的40分类

1. 第一步准备训练集，固定数据集（当然也可以不固定数据集，但是在对比实验中一定要固定数据集划分）
utils.py文件,这个文件的作业是产生2个csv文件，固定训练集和测试集
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

dataset.py 文件根据utisl.py文件来对数据进行数据预处理操作。
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

train.py 训练模型的代码

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
        model = models.vgg16(num_classes=40,dropout=0.45).to(device)
    elif method=="step1":
        model=models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        for i in model.parameters():
            i.requires_grad=False
        model.classifier=nn.Sequential(
            nn.Linear(512*7*7,2048),
            nn.ReLU(True),
            nn.Dropout(p=0.35),
            nn.Linear(2048,1024),
            nn.ReLU(True),
            nn.Dropout(p=0.35),
            nn.Linear(1024,40)
        )
        model.to(device)
    elif method=="step2":
        model=models.vgg16()
        model.classifier=nn.Sequential(
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.35),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.35),
            nn.Linear(1024, 40)
        )
        model.load_state_dict(torch.load("model/vgg16_step1_trush.pth"))
        model.to(device)
    print("train on ",device)
    #损失函数（二分类交叉熵）
    loss_fn = nn.CrossEntropyLoss()

    #优化器
    optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.9)

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

    for epoch in range(10):
        train_loss_total = 0 #所有batch的loss累加值
        train_acc_total = 0 #所有batch的acc累加值
        val_loss_total = 0
        val_acc_total = 0

        model.train()#标志此时为训练状态，启用dropout随机失活，否则不启用
        for i,(train_x,train_y) in enumerate(train_dataloader):
            train_x = train_x.to(device)
            train_y = train_y.to(device)

            #前向传播
            train_y_pred = model(train_x)
            train_loss = loss_fn(train_y_pred,train_y)
            train_acc = (m(train_y_pred).max(dim=1)[1]==train_y).sum()/train_y.shape[0]
            train_loss_total += train_loss.data.item()
            train_acc_total += train_acc.data.item()
            #反向传播
            train_loss.backward()
             #梯度下降
            optimizer.step()
            optimizer.zero_grad()

            print("epoch:{} train_loss:{} train_acc:{}".format(epoch, train_loss.data.item(), train_acc.data.item()))

        train_loss_arr.append(train_loss_total / len(train_dataloader)) #平均值
        train_acc_arr.append(train_acc_total / len(train_dataloader))

        #测试集
        for j, (val_x, val_y) in enumerate(val_dataloader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            #前向传播
            val_y_pred,_,_ = model(val_x)
            val_loss = loss_fn(val_y_pred,val_y)
            val_acc = (m(val_y_pred).max(dim=1)[1]==val_y).sum()/val_y.shape[0]
            val_loss_total += val_loss.data.item()
            val_acc_total += val_acc.data.item()

        val_loss_arr.append(val_loss_total / len(val_dataloader))  # 平均值
        val_acc_arr.append(val_acc_total / len(val_dataloader))
        print("epoch:{} val_loss:{} val_acc:{}".format(epoch, val_loss_arr[-1], val_acc_arr[-1]))
        #保存模型（断点连续）
        checkpoint={
            "net":model.state_dict(),
            "optimizer":optimizer.state_dict(),
            "epoch":epoch
        }
        torch.save(checkpoint,"checkpoint/ckpt.pth")


    plt.subplot(1,2,1) #画布一分为二,1行2列，用第一个
    plt.title("loss")
    plt.plot(train_loss_arr,"r",label = "train")
    plt.plot(val_loss_arr,"b",label = "val")
    plt.legend()

    plt.subplot(1, 2, 2)  # 画布一分为二,1行2列，用第一个
    plt.title("acc")
    plt.plot(train_acc_arr, "r", label="train")
    plt.plot(val_acc_arr, "b", label="val")
    plt.legend()
    plt.savefig("loss/loss_acc_vgg.png")

    plt.show()

    #保存模型
    #1.torch.save()
    #2.文件的后缀名：.pt、.pth、.pkl
    torch.save(model.state_dict(),"model/vgg_trush.pth")
    print("保存模型成功!")


if __name__ == "__main__":
    train()


```

test.py测试模型的代码
```
import torch.cuda

from torchvision import models
import os
from torch import nn
from dataset import *
from PIL import Image
from torch.utils.data import DataLoader
from dataset import *
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix
from matplotlib import rcParams
rcParams['font.family'] = 'SimHei'

m = nn.Softmax(dim=1)
labels = os.listdir("data_garbage")

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.googlenet(num_classes=40).to(device)
    model.load_state_dict(torch.load("model/vgg_trush.pth"))
    model.eval()

    img = Image.open("tests/5.jpg")
    dst = val_tf(img).to(device)
    dst = torch.unsqueeze(dst, dim=0)   # (1, 3, 224, 224)
    y_hat = model(dst)

    values = m(y_hat).sort(dim=1, descending=True)[0][0]
    index = m(y_hat).sort(dim=1, descending=True)[1][0]
    for i in range(5):
        print("{:} - {:.5f}".format(labels[index[i]], values[i]))

    plt.imshow(img)
    plt.show()

def val():
    # 数据集和数据加载器
    val_dataset = Animals_dataset(False)
    val_data_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.googlenet(num_classes=40).to(device)
    model.load_state_dict(torch.load("model/vgg_trush.pth"))
    model.eval()

    val_y_total = []
    val_y_pred_total = []
    for val_x, val_y in val_data_loader:
        val_x = val_x.to(device)
        val_y_pred = model(val_x).cpu()

        val_y_total.extend(val_y.cpu().numpy())    # 将列表中的数据取出来追加
        val_y_pred_total.extend(m(val_y_pred).max(dim=1)[1].cpu().numpy())

    p = precision_score(val_y_total, val_y_pred_total, average="weighted")
    recall = recall_score(val_y_total, val_y_pred_total, average="weighted")
    f1 = f1_score(val_y_total, val_y_pred_total, average="weighted")

    print("precision: {:.5f}, recall={:.5f}, f1={:.5f}".format(p, recall, f1))

    cm = confusion_matrix(val_y_total, val_y_pred_total)

    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.xticks(range(40), labels=labels)
    plt.yticks(range(40), labels=labels)

    plt.colorbar()
    plt.xlabel("预测值")
    plt.ylabel("真实值")
    thresh = cm.mean()
    for i in range(40):
        for j in range(40):
            info = cm[j, i]
            plt.text(i, j, info, color="white" if info>thresh else "black")
    plt.savefig("confusion_matrix.jpg")
    plt.show()


if __name__ == "__main__":
    evaluate()


```

### 什么是感受野？