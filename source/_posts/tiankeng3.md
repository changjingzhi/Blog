---
title: python自定义包的层级引用
date: 2024-04-24 13:09:20
tags: 填坑
---
## 挖坑
今天debug的时候自定义了一个函数，使用了start主函数来引用processing函数，processing函数引用了同级文件夹中的python文件中的dataset函数，在运行processing的时候，test是通过的，但是在使用start函数来调用processing函数，processing函数函数调用dataset函数时就出现了报错，提示找不到这个包。（注：这里需要指明的是start函数放置在根文件夹中，processing函数放置在processing文件夹中）问题就在于python文件的文件运行路径的出错。

```
│  start.py
│
├─static
│  │  __init__.py
│  │
│  ├─model
│  │      MLPForMer.pth
│  │
│  ├─processing
│  │  │  dataset.py
│  │  │  net.py
│  │  │  processing.py
│  │  │  __init__.py
│  │  │
│  │  └─__pycache__
│  │          dataset.cpython-38.pyc
│  │          net.cpython-38.pyc
│  │          processing.cpython-38.pyc
│  │          __init__.cpython-38.pyc
│  │
│  ├─result
│  │      average_probabilities.csv
│  │      average_probabilities.png
│  │
│  ├─tmp
│  │      1.edf
│  │
│  └─__pycache__
│          __init__.cpython-38.pyc
│
└─templates
        upload.html
```

## 填坑
对待这种问题目前我知道的有两种方法
1. 第一种方法在processing文件中明确的所以绝对引用的方法,因为问题是出现在processing中的。

```
from torch.utils.data import DataLoader
from static.processing.net import * # 这里引用是相对于start函数的位置

```

2. 第二种方法，在__init__文件中给出直接引用
1.相对引用package需要采用from 相对位置 import package_name的方式。因为相对位置只能写在from和import中间。
2.from . import * 只会检索当前目录下的module，而不会导入package。
### 挖坑
### windown怎么打印树状图？
使用`tree`来打印文件夹
使用`tree /f`来打印文件目录，如上面的文件目录结构。

### __init__文件的作用是什么？

作为包的标识：
1. 当一个目录包含__init__.py文件时，Python会将该目录视为一个包，而不仅仅是一个普通的目录。这使得包内的模块可以被正确导入和使用。
2. __init__.py文件可以是一个空文件，也可以包含初始化包的代码，比如设置包的属性、导入子模块等。

初始化包：
1. 在包被导入时，__init__.py文件会在包内的其他模块之前被执行。这使得可以在__init__.py中执行一些初始化操作，比如设置包级别的变量、执行必要的初始化代码等。
2. 这也可以用于在导入包时自动执行一些操作，比如注册插件、加载配置等。·


## 填坑
### 居中显示
可以使用center标签，或者使用div标签，或者使用p标签，或者h标签都是可以的

```
<center> <>数据结构和算法是居中展示，使用center标签</center>
<div align=center>数据结构和算法是居中展示，使用div标签</div>
<p align="center">数据结构和算法是居中展示，使用p标签</p>
<h5 style="text-align:center">数据结构和算法是居中展示，使用h标签</h5>
```

### 给改文字大小
使用font标签，字体使用face，颜色使用color，尺寸使用size。
颜色可以使用字母比如red，black，blue，yellow等，也可以是十六进制表示比如#0000ff或者#F025AB等等
size 是从1到7，数字越小字体越小，浏览器默认是3
这几个属性可以都设置，也可以只设置其中的1到2个

```
<font face="黑体">我是黑体字体</font>
<font face="微软雅黑">我是微软雅黑字体</font>
<font face="STCAIYUN">我是华文彩字体云</font>
<font color=red size=3 face="黑体">我是红色，黑色字体，大小是3</font>
<font color=#F025AB size=5>我的颜色是#F025AB，大小是5</font>

```