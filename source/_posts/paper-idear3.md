---
title: 论文思路——论文阅读文献
date: 2024-04-30 09:29:04
tags: 填坑
---

## 论文阅读记录
1. [DICE-Net](https://ieeexplore.ieee.org/abstract/document/10179900) 发表在IEEE Access 2022-2023年实时影响因子为3.9，中科院分区3区到4区。论文为OA论文，开源。
启发是对数据预处理的方法（比如RBP，SCC），一种未验证的双输入数据模型结构（注：我看来就是数据的通道加一）。这篇论文在公开 [数据集](https://www.mdpi.com/2306-5729/8/6/95)  做的分类为CN/AD,FTD/CN,这么划分，是因为他要做对比实验，横向对比。实验结果是CN/AD 的ACC为83.28%，SENS 79.81% ，SPEC 为87.94%, PREC为88.94%，F1为84.12% 。在FTD/CN上ACC为74.96%，SENS 60.62% ，SPEC 为78.63%, PREC为64.01%，F1为62.27% 
![论文结果](pic/lwsl2.png)
2. [静息状态下脑电信号多特征融合学习预测阿尔茨海默病](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.1272834/full) 2023年
这篇论文在AD/FDT/CN 三分类上达到了80.23%的准确率。论文中提到了（DICE-net），还提到了一篇论文使用Transformer-based methodlogy[链接](https://www.sciencedirect.com/science/article/abs/pii/S0378437122004642)     在公开数据集SEED上进行情绪检测在三类问题上实现了83.03的准确率。
在2020年，一篇论文[](https://ieeexplore.ieee.org/document/9162148)中使用了 Fast Fourier Transform （FFT）来进行数据预处理，然后经过处理的数据给入CNN网络，实现了79%的结果。
![论文结果](pic/lwsl3.png)
看着看着感觉这篇论文有点细节问题没有处理好。（注： 这篇论文有点小毛病,明明做的FDT的分类，为什么有MCI的标记呀）


3. [A Dataset of Scalp EEG Recordings of Alzheimer’s Disease, Frontotemporal Dementia and Healthy Subjects from Routine EEG](https://www.mdpi.com/2306-5729/8/6/95)提供了19通道的 (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2) 的脑电记录数据，有36个AD患者，23个FDT（前额叶痴呆），29个CN对照。数据包含了未经过伪迹处理的和已经经过伪迹处理的信号,同时论文中也使用了一定的方法来对数据进行处理。
![论文结果](pic/lwsl.png)





## 挖坑

### 公开数据集介绍
This article provides a detailed description of a resting-state EEG dataset of individuals with Alzheimer’s disease and frontotemporal dementia, and healthy controls. The dataset was collected using a clinical EEG system with 19 scalp electrodes while participants were in a resting state with their eyes closed. The data collection process included rigorous quality control measures to ensure data accuracy and consistency. The dataset contains recordings of 36 Alzheimer’s patients, 23 frontotemporal dementia patients, and 29 healthy age-matched subjects.
提供了19通道的 (Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, and O2) 的脑电记录数据，有36个AD患者，23个FDT（前额叶痴呆），29个CN对照。数据包含了未经过伪迹处理的和已经经过伪迹处理的信号（伪迹处理的信号过程请参考论文）

### 论文结构
鉴于论文大修，所以我打算重新构建一下思路。
1. 标题
2. 摘要
3. 关键词

1. 介绍
2. 数据和方法
3. 结果
4. 讨论
5. 参考文献