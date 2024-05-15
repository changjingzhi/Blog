---
title: 目标检测——RCNN
date: 2024-05-15 10:17:07
tags: 深度学习论文
---

目标检测，最为经典的项目实例就是人脸检测，在paper with code[链接](https://paperswithcode.com/sota) 网站中在Object Decection中包含大量案例，但是最为经典的还是RCNN，开山之作。

## 背景介绍
论文链接[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)
[参考链接](https://zhuanlan.zhihu.com/p/23006190)
[代码实现](https://github.com/yangxue0827/RCNN)
## 算法过程
CNN算法分为4个步骤

候选区域生成： 一张图像生成1K~2K个候选区域 （采用Selective Search 方法）
特征提取： 对每个候选区域，使用深度卷积网络提取特征 （CNN）
类别判断： 特征送入每一类的SVM 分类器，判别是否属于该类
位置精修： 使用回归器精细修正候选框位置

