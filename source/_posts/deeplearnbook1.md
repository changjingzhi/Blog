---
title: PyTorch介绍
date: 2024-04-19 17:19:34
tags: 深度学习
---
 
在深度学习，要永远抱着学徒的心。
本人参考书目为《Python深度学习基于PyTorch》 [下载链接](http://www.feiguyunai.com/) [使用下载链接——github](https://github.com/Wumg3000/feiguyunai)

# 目前深度学习的框架有什么？
1. TensorFlow ：由Google开发的开源深度学习框架，提供了灵活性和高性能计算能力。TensorFlow 2.x版本引入了更加易用的Keras API作为主要接口。[TensorFlow的github链接](https://github.com/tensorflow/tensorflow)
2. PyTorch ：由Facebook开发的开源深度学习框架，以动态计算图的方式进行建模，易于调试和学习。PyTorch在研究领域广泛应用。[PyTorch的github链接](https://github.com/pytorch/pytorch)
3. Keras：最初作为独立的深度学习框架，现在已经成为TensorFlow的高级API。Keras提供了简单易用的接口，适合快速搭建深度学习模型。 [Keras的github链接](https://github.com/keras-team/keras)
4. MXNet：由Apache软件基金会支持的深度学习框架，具有高度可扩展性和灵活性。MXNet支持动态和静态计算图。[MXNet的github链接]（https://github.com/apache/mxnet）
5. CNTK (Microsoft Cognitive Toolkit)：由微软开发的深度学习框架，提供了高效的性能和多GPU支持。 [CNTK的github链接](https://github.com/microsoft/CNTK)
6. PaddlePaddle（百度飞桨）。这是一个由百度开发的开源深度学习平台，它为深度学习研究人员和开发者提供了丰富的API，支持多种模型结构，可以用来创建各种深度学习模型。[百度飞浆的链接]（https://www.paddlepaddle.org.cn/）

# 为什么要学习PyTorch？
1. pytorch是动态计算图，用法更接近python，并且pytoch与python共同使用了numpy的命令，降低了学习的门槛，比TensorFlow更容易上手
2. pytorch需要定义网络层、参数更新等关键步骤，有助于学习深度学习的核心（根据梯度更新参数。）
3. pytorch的流行仅次于TensorFlow。在github上的stareed为77.7K （此数据截止到2024/4/19日）
4. pytorch的动态图机制在调试方面非常简单，如果计算图运行出错，马上可以跟踪到问题。pytorch的调试和python一样，可以通过断点检查来解决问题。

# 解释一下这本书的结构
1. 第一部分：介绍深度学习的基石Numpy，介绍PyTorch基础于pytorch构建神经网络的工具箱和数据处理工具。
2. 第二部分：这本书的核心内容，包括机器学习的流程，常用算法和技巧等内容。实现了基于卷积神经网络的多个视觉处理实例，实现了多个自然语言处理、时间序列方面的实例。介绍了编码器——解码器模型、带注意力的编码器——解码器模型、对抗式生成器以及多种衍生生成器。（注：这里阐述一下关于深度学习、机器学习、人工智能之间的关系。人工智能包含机器学习，机器学习包含深度学习）
3. 第三部分：实战部分，这部分在介绍相关原理、架构的基础上，使用了pytoch实现了多个深度学习典型实例，比如人脸识别、迁移学习、数据增强、中英文互译、生成式网络实例、模型迁移、强化学习、深度强化学习等实例。






## 挖坑
### 什么是python的断点检查？
### 什么是动态计算图？