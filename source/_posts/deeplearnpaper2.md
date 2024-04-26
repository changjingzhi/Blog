---
title: 经典网络结构——VGG
date: 2024-04-22 18:55:02
tags: 深度学习论文
---

## 参考文献

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

In this work we investigate the effect of the convolutional network depth on itsaccuracy in the large-scale image recognition setting. Ourmain contribution isa thorough evaluation of networks of increasing depth usingan architecture withvery small (3×3) convolution filters, which shows that a significant improvementon the prior-art configurations can be achieved by pushing the depth to 16–19weight layers. These findings were the basis of our ImageNet Challenge 2014submission, where our team secured the first and the second places in the localisa-tion and classification tracks respectively. We also show that our representationsgeneralise well to other datasets, where they achieve state-of-the-art results. Wehave made our two best-performing ConvNet models publicly available to facili-tate further research on the use of deep visual representations in computer vision.


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

### 什么是感受野？