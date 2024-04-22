---
title: PyTorch基础
date: 2024-04-21 17:52:49
tags: 深度学习
---
第一部分

# Numpy基础

## 什么是Numpy？
NumPy（Numerical Python的简称）是一个开源的Python库，用于进行科学计算。它提供了一个强大的N维数组对象，以及大量的函数用于处理这些数组。NumPy的主要功能包括：
1. 多维数组对象：NumPy的核心功能是其多维数组对象（ndarray）。这是一个快速、灵活的容器，可以容纳大量同类型数据，使你能够对这些数据进行数学运算。
2. 广播功能：NumPy提供了广播功能，这是一种强大的机制，允许NumPy在执行算术运算时处理不同形状的数组。
3. 数学函数：NumPy提供了大量的数学函数，可以对数组中的元素进行各种数学运算，如加、减、乘、除、平方根等。
4. 线性代数：NumPy包含了线性代数函数库，可以进行矩阵乘法、求逆、解线性方程，傅里叶变换等操作。
5. 随机数生成：NumPy提供了生成各种随机数的功能，如均匀分布、正态分布等。
6. 更方便的读取\写入磁盘上的阵列数据和操作存储映像文件的工具。
NumPy是许多科学计算库（如Pandas、Matplotlib、SciPy等）的基础库，也是机器学习和数据科学中常用的库。在深度学习中图像、声音、文本等输入数据最终都要转换为数组或矩阵，NumPy的多维数组可以用来表示向量、矩阵和张量，这些都是深度学习算法的基本构成元素。

## 为什么要使用Numpy
实际上python包含多个数据类型，数值类型（int，float，complex）、布尔类型（bool）、字符串（str）、列表（list）、元组（tuple）、字典（dict，{'name':'chenli','age': 114514}）、集合（set{}）。
python包含这么多的数据类型，其中的列表（list）和数组（array）为什么不能用？原因是对于大数据来说，这些结构有很多不足。比如由于列表的元素可以是任何对象，因此列表中所保存的是对象的指针。例如为了存储[1,2,3],就需要3个指针和3个整数对象，这样对于数值运算来讲，严重浪费了计算机中的内存和CPU或GPU的算力。对于array，它可以直接保存数值，但是它不支持多维，并且对应操作的函数也不多，因此也不适合。

## 怎么使用Numpy？
第一步使用 `pip install numpy `来下载numpy库。
1. 生成Numpy数组。
```
import numpy as np
lst1= [[3.14, 2.17,0,1,2],[1,2,3,4,5]] # 一个二维列表
nd1 = np.array(lst1)
print(nd1) # 显示的结果为一个二维列表
print(type(nd1)) # 显示结果为<class 'numpy.ndarray'>    
```

2. numpy中的random模块生成数组
```
import numpy as np

nd1 = np.random.random([3,3]) # 产生一个[3,3]ndarray，范围为0-1之间的随机数。
np.random.seed(123) # 为了每次生成同一份数据，可以指定一个随机种子，生成的随机数据是固定的。
nd2 = np.random.random(2,3) # 产生一个[2,3]的ndarray
np.random.shuffle(nd2) # 随机打乱nd2中的数据。
nd3 = np.random.uniform(2,3) # 生成均匀分布的随机数
nd4 = np.random.randn(2,3) # 生成标准正态分布的随机数 
nd5 = np.random.randint(2,3) # 生成随机的整数
```

3. numpy中创建特定形状的多维数组
```
import numpy as np

nd1 = np.zeros([3,3]) # 生成全是0的3x3的矩阵，np.zeros()生成全是0的ndarray
nd2 = np.zeros_like(nd1) # 以ndrr相同维度创建元素为0的数组
nd3 = np.ones_like(nd1) # 以nd1维度创建元素全是1的数组
nd4 = np.empty_like(nd1) # 以nd1维度创建一个空数组
nd5 = np.eye(5) # 该函数用于创建一个5x5的矩阵，对角线为1，其余为0
nd6 = np.full((3,5),666) # 创建3x5的元素全为666的数组，666为指定值
```

4. 保存使用numpy生成的数据
```
import numpy as np
nd = np.random.random([5,5])
np.savetxt(X=nd,fname='test.txt') # 保存nd，文件名称为test.txt
nd2 = np.loadtxt('test.txt') # 从test.txt中加载数据
```

5. 利用arange、linspace函数来生成数组
arange是numpy模块中的函数，格式为：`arange([start],stop[,step],dtype=None)` 
其中start与stop用来限定范围，step是步长默认为1，step可以为小数。
```
import numpy as np
nd = np.arange(10) # np中的内容为[0 1 2 3 4 5 6 7 8 9]
nd1 = np.arange(1，4，0.5) # np中的内容为[1.0 1.5 2.0 2.5 3.0 3.5 4.0]
nd2 = np.aramge(9,-1,-1) # nd2中的内容为[9 8 7 6 5 4 3 2 1 0]
```
linspace也是numpy中常用的函数，格式为：`np.linspace(start,stop,num=50,endpoint=True,retstep=False,dtype=None)`
linspace可以根据输入数据的指定范围以及等份数量，自动生成线性分量。endpoint（包含终点）默认为True。等分量num默认为50，如果将retstep设置为True，则会返回一个带步长的ndarray
```
import numpy as np
print(np.linspace(0,1,10)) # 产生10个数，间隔为0.111111 
```

6. 获取数据。
数据生成后，如何读取数据，常用数据的的方法。
```
import numpy as np
np.random.seed(2024)
nd = np.random.random([10]) # 产生一维的10个数据点
nd[3]  # 获取指定位置的数据，获取第4个元素
nd[3:6] # 截取一段数据
nd[1:6:2] # 获取固定间隔的数据
nd[::-2] # 倒序取数
nd2 = np.arange(25).reshape([5,5]) # 产生一个25个数据点的一维数据，而后通过reshape进行形状重整为[5,5]
nd2[:,1:3] # 截取多维数组中，指定的列，读取第2，3列。
```

7. Numpy的算术运算
在机器学习和深度学习中，涉及大量的数组或矩阵运算，这里介绍两种常用的运算。一种是对应元素相乘，又称逐元乘法（Element-Wisr Product）运算符为np.multiply()或*。一种是点积或内积元素，运算符为np.dot()


## 挖坑
### 向量和数组之间的关系是什么？向量的定义是什么？
### 矩阵是什么，作用是什么？如何实现矩阵的加减乘除
### 傅里叶变换是什么？原理是这样的？
### 什么是对象？ 封装，继承，多态是什么？

### python中的不同代码高亮表示什么？
在Python的IDLE编程环境中，不同颜色的文本表示不同的含义。以下是IDLE中常见的颜色及其含义：
黑色：普通的代码文本。
蓝色：关键字，例如if、else、for、while等。
绿色：字符串文本。
红色：语法错误或代码中的错误。
紫色：函数和方法的名称。
棕色：数字。
橙色：内置函数和模块的名称。
灰色：注释。