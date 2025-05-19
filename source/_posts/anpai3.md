---
title: 深度学习-复习计划
date: 2024-06-28 15:30:46
tags: 计划
---

选择、填空、解答、应用。

## 基础
1. 以下哪种图像每个像素点有RBG三个值（  C  ）
A. 二值图像
B. 灰度图像
C. 彩色图像
D. 以上都没有

2. OpenCV库中哪个接口可以实现图像翻转（ D ）
A. cv2.imshow()
B. cv2.imread()
C. cv2.resize()
D. cv2.flip()


3. 使用OpenCV库驱动摄像头捕获视频流时，若电脑只有一个摄像头，请问如下哪个调用是正常的（    B  ）
A. cv2.VideoCapture(-1)
B. cv2.VideoCapture(0)
C. cv2.VideoCapture(1)
D. cv2.VideoCapture()

4. 如下哪个是线性回归损失函数MSE的公式？（   C   ）
![](pic/sdxx-fxjh-1.png)


5. 二分类问题中，Precision精确率的计算公式是（  C    ）
A. TP/(TP+TN)
B. TP/(TP+FN)
C. TP/(TP+FP)
D. 以上都不对

![二分类示意图](pic/predict.jpg)


6. 以下哪个loss曲线图对应的是模型过拟合状态？（ B ）
![](pic/sdxx-fxjh-2.png)


7. 如下哪个是二分类交叉熵损失函数（C）
![](pic/sdxx-fxjh-3.png)

| |
| :------ | 
|![](pic/sdxx-fxjh-4.png)|
|![](pic/sdxx-fxjh-5.png)|
| :------ | 
|![](pic/sdxx-fxjh-6.png)|
|![](pic/sdxx-fxjh-7.png)|
| :------ | 
|![](pic/sdxx-fxjh-8.png)|
|![](pic/sdxx-fxjh-9.png)|
|![](pic/sdxx-fxjh-10.png)|
| :------ | 
|![](pic/sdxx-fxjh-11.png)|
| 迁移学习| 
|![](pic/sdxx-fxjh-12.png)|
| 复习| 
|![](pic/sdxx-fxjh-13.png)|
|![](pic/sdxx-fxjh-14.png)|
|![](pic/sdxx-fxjh-15.png)|
| 目标检测| 
|![](pic/sdxx-fxjh-16.png)|
|![](pic/sdxx-fxjh-17.png)|
|![](pic/sdxx-fxjh-18.png)|


## 详细知识点复习
1. MAP计算方法

2. 使用的工具 python，opencv， pytorch， visual code， conda， 

3. 池化的总类：池化（Pooling）是一种常用的操作，通常与卷积神经网络（CNN）结合使用。池化操作通过对输入数据的局部区域进行聚合或采样来减小数据的空间尺寸，从而减少参数数量、降低计算量，并提取出输入数据的重要特征。

池化的作用有以下几个方面

降采样：池化操作可以减小输入数据的空间尺寸，从而降低后续层的计算复杂度。通过降低数据的维度，池化可以在保留重要特征的同时减少冗余信息，提高计算效率。
平移不变性：池化操作具有一定的平移不变性。在图像处理中，通过对局部区域进行池化操作，可以使得输入图像在平移、旋转和缩放等变换下具有一定的不变性。这对于图像识别和目标检测等任务是有益的。
特征提取：池化操作可以提取输入数据的重要特征。通过对局部区域进行池化，池化操作会选择区域中的最大值（最大池化）或平均值（平均池化）作为输出值，从而提取出输入数据的显著特征。这有助于减少数据的维度，并保留重要的特征信息。
减少过拟合：池化操作可以在一定程度上减少过拟合。通过减小数据的空间尺寸，池化操作可以降低模型的参数数量，从而减少过拟合的风险。此外，池化操作还可以通过丢弃一些冗余信息来提高模型的泛化能力。
池化的种类

最大池化（Max Pooling）：最大池化是一种常见的池化操作。在最大池化中，输入数据的局部区域被分割成不重叠的块，然后在每个块中选择最大值作为输出。最大池化可以提取出输入数据的显著特征，同时减小数据的空间尺寸。
平均池化（Average Pooling）：平均池化是另一种常见的池化操作。在平均池化中，输入数据的局部区域被分割成不重叠的块，然后计算每个块中元素的平均值作为输出。平均池化可以平滑输入数据并减小数据的空间尺寸。
自适应池化（Adaptive Pooling）：自适应池化是一种具有灵活性的池化操作。与最大池化和平均池化不同，自适应池化不需要指定池化窗口的大小，而是根据输入数据的尺寸自动调整池化窗口的大小。这使得自适应池化可以适应不同尺寸的输入数据。
全局池化（Global Pooling）：全局池化是一种特殊的池化操作，它将整个输入数据的空间尺寸缩减为一个单一的值或向量。全局池化可以通过对输入数据的所有位置进行池化操作，从而提取出输入数据的全局特征。常见的全局池化有全局平均池化（Global Average Pooling）和全局最大池化（Global Max Pooling）[链接](https://chenlidbk.xyz/2024/04/23/tiankeng2/#%E9%97%AE%E9%A2%982%EF%BC%8C%E6%B1%A0%E5%8C%96%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F%E4%BD%9C%E7%94%A8%E6%98%AF%E4%BB%80%E4%B9%88%EF%BC%9F)

3. 通道数和卷积核之间的关系是什么？这个通道数是怎么减少和扩大的?

4. 评估指标中的  average = ( 'binary'、''weighted'、 'samples' 、 'macro' ) 的区别是什么？
| |
| :------ | 
|![](pic/sdxx-fxjh-19.png)|


5. RCNN，全称为区域卷积神经网络（Region-based Convolutional Neural Networks），是一种用于图像中物体检测的深度学习模型。它由Ross Girshick、Jeff Donahue、Trevor Darrell和Jitendra Malik在他们的2014年论文“Rich feature hierarchies for accurate object detection and semantic segmentation”中提出。RCNN通过几个关键步骤来实现物体检测：
| |
| :------ | 
|![](pic/sdxx-fxjh-20.png)|

6. AP计算过程
[推荐视频](https://www.bilibili.com/video/BV1ez4y1X7g2/?spm_id_from=trigger_reload&vd_source=9814cf6702c46a0b906cb31de22baa58)
| |
| :------ | 
|![](pic/sdxx-fxjh-21.png)|
|![](pic/sdxx-fxjh-22.png)|