---
title: 论文阅读记录
date: 2024-07-13 09:50:42
tags: 论文思路
---

## 第一篇

《EEG Context Fusion for AI-Based Object Detection and Drone Navigation in Situationally Aware
Brain-Computer Interfaces》

We are interested in the utility that artificially intelligent mobile systems such as drones offer to personnel in fast-paced situations such as hostage rescue and disaster relief for real-time situational awareness. Ideally, these assistive systems place no additional cognitive or physical burden on their user; rather, they should respond to the user’s intent with minimal physical and cognitive burden. To this end, brain-computer interface (BCI) and in particular electroencephalography (EEG) offers a novel way to capture intent. EEG-based drone control has been explored, but typically relies on the domain of motor imagery (MI), which still requires somewhat-manual drone piloting for micro directional movement, adding cognitive burden. We propose leveraging the robust computer-vision based AI that exists on modern drones to use objects as waypoints and fly mostly autonomously, with EEG in an object recognition paradigm for passively selecting the waypoint. While existing techniques for EEG object recognition-including steady-state visually evoked potentials (SSVEP), rapid serial visual presentation (RSVP), and eye tracking - rely on “marked” input or a controlled environment, we need one that will work on passive objects. In this work, we propose merging available scene imagery in a vision network with deep-learning based EEG processing to achieve passive intent recognition. The proposed context-scene fusion for a situational awareness paradigm presents accuracy in the same ranges as existing object recognition without requiring SSVEP, RSVP, eye tracking, or MI techniques and serves as proof-of-concept for this approach to real-world BCI applications.

我们对人工智能移动系统（如无人机）在快速变化的环境中提供实时态势感知的效用感兴趣，例如在人质救援和灾难救援中。理想情况下，这些辅助系统不应给用户增加额外的认知或身体负担；相反，它们应该在最小的身体和认知负担下响应用户的意图。为此，脑-机接口（BCI），特别是脑电图（EEG），提供了一种捕捉意图的新方法。基于EEG的无人机控制已经被探索过，但通常依赖于运动想象（MI）领域，这仍然需要一定程度的手动无人机操控以进行微方向移动，从而增加了认知负担。

我们建议利用现代无人机上存在的强大计算机视觉AI技术，将物体用作航点，主要通过自动驾驶无人机，并在对象识别范式中使用EEG来被动选择航点。现有的EEG对象识别技术，包括稳态视觉诱发电位（SSVEP）、快速串行视觉呈现（RSVP）和眼动追踪，依赖于“标记”输入或受控环境，而我们需要一种在被动对象上工作的技术。在这项工作中，我们建议将可用的场景图像与视觉网络中的深度学习EEG处理相结合，以实现被动意图识别。所提出的情境-场景融合态势感知范式在不需要SSVEP、RSVP、眼动追踪或MI技术的情况下，呈现出与现有对象识别相同范围的准确性，并作为这一现实世界BCI应用方法的概念验证。

简要来说，这项研究通过将无人机的计算机视觉能力与EEG的深度学习处理结合起来，减少了传统方法中的认知负担，提高了无人机在快速变化环境中的实用性。


## 第二篇

Multi-feature fusion learning for Alzheimer’s disease prediction using EEG signals in resting state

### Introduction

Diagnosing Alzheimer’s disease (AD) through visual examination of Electroencephalography (EEG) signals presents significant challenges due to the complexity and noise inherent in EEG data. This difficulty has led to the exploration of deep learning techniques, particularly Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs), for AD prediction. However, CNN-based methods often fall short in classification performance, primarily because they struggle to extract meaningful lesion signals from intricate EEG patterns.

通过视觉检查脑电图（EEG）信号来诊断阿尔茨海默病（AD）面临显著挑战，因为EEG数据本身复杂且噪声较多。这一困难促使了对深度学习技术的探索，尤其是卷积神经网络（CNN）和视觉变换器（ViT）在AD预测中的应用。然而，基于CNN的方法在分类性能上往往表现不佳，主要是因为它们难以从复杂的EEG模式中提取有意义的病变信号。

### Methods

In contrast, ViTs have shown considerable proficiency in capturing global signal patterns. To address the limitations of both approaches, we propose a novel hybrid architecture that merges the strengths of CNNs and ViTs. Our proposed Dual-Branch Feature Fusion Network (DBN) effectively integrates texture features from CNNs with global semantic information from ViTs, which are crucial for identifying dynamic electrical signal changes in the cerebral cortex. Additionally, we incorporate Spatial Attention (SA) and Channel Attention (CA) blocks within the network architecture, enhancing the model’s ability to discern abnormal EEG signal patterns from the fused features. To ensure robust predictions, we employ a two-factor decision-making mechanism that includes correlation analysis of predicted EEG signals from the same subject, thereby establishing consistency.

相比之下，ViT在捕捉全局信号模式方面表现出色。为了解决这两种方法的局限性，我们提出了一种新颖的混合架构，结合了CNN和ViT的优势。我们提出的双分支特征融合网络（DBN）有效整合了来自CNN的纹理特征和来自ViT的全局语义信息，这对于识别大脑皮层中动态电信号变化至关重要。此外，我们在网络架构中引入了空间注意力（SA）和通道注意力（CA）模块，增强了模型识别合成特征中的异常EEG信号模式的能力。为了确保稳健的预测，我们采用了两因素决策机制，包括对来自同一受试者的预测EEG信号进行相关性分析，从而建立一致性。

### Results

Our approach is further complemented by results from the Clinical Neuropsychological Scale (MMSE) assessment, enabling a comprehensive evaluation of the subject’s susceptibility to AD. Experimental validation on the publicly available OpenNeuro database demonstrates the efficacy of our method, achieving an impressive 80.23% classification accuracy in distinguishing between AD, Frontotemporal Dementia (FTD), and Normal Control (NC) subjects.

我们的这一方法还结合了临床神经心理评估量表（MMSE）的结果，使得对受试者AD易感性的综合评估得以实现。在公开可用的OpenNeuro数据库上的实证验证表明，我们的方法具有良好的效果，成功在AD、额颞痴呆（FTD）和正常对照（NC）受试者之间实现了80.23%的分类准确率。

### Discussion

These results surpass current state-of-the-art methodologies in EEG-based AD prediction. Moreover, our methodology facilitates the visualization of salient regions within pathological images, offering valuable insights for the interpretation and analysis of AD predictions.

这一结果超越了目前EEG基础AD预测的前沿方法。此外，我们的方法还实现了病理图像中显著区域的可视化，为AD预测的解释和分析提供了宝贵的见解。



## 第三篇
DICE-Net: A Novel Convolution-Transformer Architecture for Alzheimer Detection in EEG Signals
### 目标

阿尔茨海默病（AD）是一种进行性神经退行性疾病，影响大量老年人。脑电图（EEG）作为一种有前景的工具，能够及时诊断和分类AD或其他类型的痴呆。本论文提出了一种新颖的AD EEG分类方法，采用双输入卷积编码网络（DICE-net）。

### 方法

本研究使用了36名AD患者、23名额颞痴呆（FTD）患者和29名年龄匹配的健康个体（CN）的EEG记录。经过去噪处理后，提取了带宽功率和相干性特征，并将其输入到DICE-net中，后者由卷积层、变换编码器和前馈层组成。

### 主要结果

我们的结果表明，DICE-net在AD-CN分类任务中使用留一法验证取得了83.28%的准确率，优于多个基线模型，并展现了良好的泛化性能。

### 重要性

我们的研究结果表明，卷积变换网络能够有效捕捉EEG信号的复杂特征，从而用于AD患者与对照组的分类，并可扩展到其他类型的痴呆，如FTD。这一方法有望提高早期诊断的准确性，并推动更有效的AD干预措施的发展。

## 第四篇

基于睡眠脑电信号和深度学习的抑郁症识别研究*

目的　基于睡眠脑电信号，探索深度学习Vision Transformer（ViT）结合Transformer网络对抑郁症患者识别
的有效性。方法　首先对28例抑郁症患者和37例正常对照的睡眠脑电信号进行预处理，并将信号转为图像格式，保留其
频域及空间域特征信息，之后将图像输送到ViT-Transformer编码网络，分别学习抑郁症患者和正常对照的快速眼动（rapid
eye movement, REM）睡眠期和非快速眼动（non-rapid eye movement, NREM）睡眠期的脑电信号特征，并对抑郁症进行识
别。结果　基于ViT-Transformer网络，从不同脑电频率角度，发现delta、theta和beta波的组合对抑郁症识别具有比较好的
结果。其中，REM期delta-theta-beta波组合的脑电信号特征对抑郁症识别的准确率达92.8%，精准率为93.8%，抑郁症患者
的召回率为84.7%，F0.5值为0.917±0.074；NREM期delta-theta-beta波组合的脑电信号特征对抑郁症的识别准确率为91.7%精准率为90.8 %，召回率为85.2%，F0.5值为0.914±0.062。此外，通过对整夜睡眠脑电的睡眠分期进行可视化，发现分类错误
通常发生在睡眠期转期时。结论　应用深度学习ViT-Transformer网络，本研究发现基于delta-theta-beta波组合的REM期睡
眠脑电信号特征对抑郁症识别更有效。

## 第五篇

Early Alzheimer’s disease diagnosis based on EEG spectral images using deep learning （2019年）

Early diagnosis of Alzheimer’s disease (AD) is a proceeding hot issue along with a sharp upward trend in the incidence rate. Recently, early diagnosis of AD employing Electroencephalogram (EEG) as a specific hallmark has been an increasingly significant hot topic area. In consideration of the limited size of available EEG spectral images, how to extract more abstract features for better generalization still remains tremendously troubling. In this paper, we demonstrate that it can be settled well with multi-task learning strategy based on discriminative convolutional high-order Boltzmann Machine with hybrid feature maps. First, differently from our original model — Contractive Slab and Spike Convolutional Deep Boltzmann Machine (CssCDBM), we directly conduct EEG spectral image classification via inducing label layer, resulting in a discriminative version of CssCDBM, referred to as DCssCDBM. This demonstrates DCssCDBM can be extended well into the classification model instead of feature extractor alone previously. Then, the most important approach innovation is that we train our DCssCDBM with multi-task learning framework via EEG spectral images based Identification and verification tasks for overfitting reduction for the first time, which could increase the inter-subject variations and reduce the intra-subject variations respectively, both of which are essential to early diagnosis of AD. The proposed method shows the better ability of high-level representations extraction and demonstrates the advanced results over several state-of-the-art methods.

阿尔茨海默病(AD)的早期诊断是一个不断发展的热点问题，其发病率有急剧上升的趋势。近年来，利用脑电图(EEG)作为AD早期诊断的特异性标志已成为一个日益重要的热点领域。考虑到可用的脑电信号频谱图像的大小有限，如何提取更多的抽象特征以更好地泛化仍然是一个非常棘手的问题。本文证明了基于混合特征映射的判别卷积高阶玻尔兹曼机的多任务学习策略可以很好地解决这一问题。首先，与我们的原始模型——收缩Slab和Spike卷积深度玻尔兹曼机(CssCDBM)不同，我们直接通过诱导标签层对EEG频谱图像进行分类，得到了判别版的CssCDBM，简称DCssCDBM。这表明DCssCDBM可以很好地扩展到分类模型中，而不是之前单独的特征提取器。然后，最重要的方法创新是，我们首次通过基于脑电图频谱图像的识别和验证任务，用多任务学习框架训练我们的DCssCDBM，以减少过拟合，这可以分别增加受试者间的变化和减少受试者内的变化，这两者对于AD的早期诊断至关重要。该方法具有较好的高级表示提取能力，其结果优于现有的几种方法。


## 第六篇

LCADNet: a novel light CNN architecture for EEG‑based Alzheimer disease detection [代码和pdf文件](https://github.com/SandeepSangle12/LCADNet/blob/main/s13246-024-01425-w.pdf)

[Maxpool1D的解释](https://blog.csdn.net/yingluo54/article/details/122168364)


## 第七篇

Classification assessment methods

## 第八篇
Novelty detection-based approach for Alzheimer’s disease and mild cognitive impairment diagnosis from EEG  [链接-包含一个公开数据集](https://link.springer.com/article/10.1007/s11517-021-02427-6) [数据集下载链接](https://figshare.com/articles/dataset/dataset_zip/5450293)