---
title: 图像处理 -复习计划
date: 2024-06-29 20:29:34
tags: 计划
---
选择题20分，计算30分[卷积， 提取位平面，滤波]，编程30分[手写代码 ]，问答 20分

## 图像处理基本知识
图像研究背景
图像是人类获取信息、表达信息和传递信息的重要手段。在人类接受的信息中，图像等视觉信息所占的比重达到75%。数字图像处理技术已经成为信息科学、计算机科学、工程科学、地球科学等诸多方面的学者研究图像的有效工具。

图像是物体反射或透射光的分布，像是人的视觉系统所接受的图在人脑中所形成的印象或认识。图像包括不可见的图像和可见的图像。不可见的图像包括数学函数，可见的图像包含光图像，照片、图、画。

数字图像的概念。数字图像描述的是如何用一个数值方式来表示一个图像。数字图像是图像的数字表示，像素是其最小单位。数字图像通常可以使用矩阵来表示。图像的数字化过程，将模拟图像经过离散化后，得到使用数字表示的图像。采样空间上连续的图像转换成离散的采样点（即像素）集的操作。采样指标——分辨率，映射到图像平面上的单个像素的景物元素的尺寸。

幅值离散化—— 量化，将各个像素所含的明暗信息离散化后，用数字来表示。一般的量化值为整数。0 ~ 255 的整数来描述"从黑到白"

黑白图像是指图像的每个像素只能是黑或者白，又称为二值图像，二值图像的像素为0或者1。
灰度图像是指每个像素的信息由一个量化的灰度级来描述的图像，没有彩色信息。通常使用2^8位来表示一个像素点，所以每个点的取值为[0,255] 
彩色图像通常会采用RGB三原色来表示，其中RGB是由不同的灰度级来描述，因此彩色图像通常由一个三维矩阵来表示（或三个二维矩阵）。

RGB模型是指红（Red）、绿（Green）、蓝（Blue）三种光原色。RGB模型是色光的彩色模型，该模型也叫加色合成法（Additive Color Synthesis）,所有的显示器、投影设备，以及电视等许多设备都是依赖于这种加色模型的。

CMYK模型CMYK(Cyan - Magenta  Ye11ow  B1ack)C代表青色，M代表洋红色，￥代表黄色。K代表黑色。它是通过颜色相减来产生其他颜色的，所以我们称这种方式为减色合成法(Subtractive Co1or Synthesis) CMYK模型主要用于印刷。

HSI模型。人眼的色彩知觉主要包括三个要素，即区分颜色的三种基本特性量，称做色调(Hue)、饱和度(Saturation)和亮度(Intensity) o利用色调﹑饱和度和亮度来描逑彩色空间，即HSI模型·

HSV模型 HSV分别代表，色相Hue，饱和度Saturation，明度Value

数字图像处理系统基于图像采集、图像传输、图像处理与分析、图像存储、图像输出。

图像处理的主要研究方向，图像的数字化、图像增强与清晰化、图像变换、图像融合、图像检索/识别/分类
图像隐藏、图像编码与压缩、图像三维重建、图像复原。



## 开发环境

linux（版本 Ubuntu 18.04）
conda 在cinda中创建opencv环境，然后在该环境中安装：python（版本3.7）， opencv（版本3），skimage，Vs code（编辑器）


第一段代码
```
from skimage import data
from matplotlib import pyplot s plt 

if __name__ == '__main__':
    image1 = data.coffee() ## 加载data中的coffee图像
    plt.imshow(image1) ## 将image1图像
    plt.show()
    print('hello')

```


第二段代码

```
import CV2

if __name__ == '__main__'

    img1 = cv2.imread('1.jpeg') # 使用cv2.imread函数来对位置为1.jpeg的图片进行调用
    img2 = cv2.imread('2.jpeg')
    img3 = cv3.imread('3.jpeg')
    

    print(img1.shape) # 打印图片的shape
    print(img2.shape)
    print(img3.shape)

    cv2.imshow('image1',img1) # 使用cv2.imshow来进行显示
    cv2.imshow('iimage2',img2)
    cv2.imshow('image3',img3)

    key = cv2.waitkey()
    cv2.destroyAllWindows()
```


## 数字图像处理基础（完成）
1. 图像的读取和显示
使用 cv2 中的imread（）函数格式 img =  cv2.imread(filename[,int flag=1] ) flag取值 flag = -1 ， 8位深度，原通道，flag = 0 ， 8位深度，1通道。 flag = 1， 8 位深度，3通道。flag=2 ，原深度，1通道。 flag = 3， 原深度，3通道，flag = 4， 8位深度，3通道。


```

import cv2

if __name =='__main__':
    img00 =cv2.imread('dog.bmp',-1)
    cv2.imshow('-1',img00)
    img0 = cv2.imread('dog.bmp',0)
    cv2.imshow('0',img0)
    img1 = cv2.imread('dog.bmp',1)
    cv2.imshow('1',img1)
    img2 = cv2.imread('dog.bmp',2)
    cv2.imread('2',img2)
    img3 = cv2.imread('dog.bmp',3)
    cv2.imshow('3',img3)
    img4 = cv2.imread('dog.bmp',4)
    cv2.imshow('4',img4)

    print('img00 size : ', img00.shape)
    print('img0 size : ',img0.shape)
    print('img1 size : ',img1.shape)
    print('img2 size : ',img2.shape)
    print('img3 size :', img3.shape)
    print('img4 size : ',img4.shape)

    cv2.waitKey() # 等待按键
    cv2.destroyAllWindows() # 关闭所有窗口

```

功能实现，刚运行程序时显示图像1，按a时显示图片1，按b键显示图片2.按除了a，b，Esc之外的按键显示图片3，按Esc键退出程序。

```
import cv2

if __name__ == '__main__':
    img1 = cv2.imread('1.jpeg')
    img2 = cv2.imread('2.jpeg')
    img3 = cv2.imread('3.jpeg')

    key = ord('a')
    while(key!=27):
        if key == ord('a'):
            cv2.imshow('a',img1)
            key = cv2.waitKey()
        elif key == ord('b'):
            cv2.imshow('b',img2)
            key = cv2.waitKey()
        else:
            cv2.imshow('c',img3)
            key = cv2.waitKey()
        cv.destroyAllWindows()

```

功能实现2，如果成功读取，则显示图像，如果读取失败，则显示“The file is not exist”

```
import cv2

if ___name__ == '__main__':
    img = cv2.imread('lena_gray.jpg',-1)

    if img is not None:
        print(img.shape)
        cv2.imshow('lena',img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print('The file is not exist')


```

2. 图像的存储

存储图像使用函数： imwrite() 格式 val = cv2.imwrite(filename,img[, params]) val:返回值（成功- True， 失败 - False） filename :文件路径和文件名， img： 图像数据变量， params 保存参数可选

```
import cv2 

if __name__ == '__main__':
    img1 = cv2.imread('lena_gray.jpg')
    result = cv2.imwrite('lean_gray.png',img1)
    print(result)


```


3. 图像的基本表示方法
图像的基本表示方法，二值图像，0表示黑，1表示白， 灰度图像通常情况下，用8为二进制来表示一副灰度图像，像素取值为[0, 255],0为黑，255为白。彩色图像，通常采用RGB三通道来表示和存储一副彩色图像，每一通道每一像素取值为[0, 255] 。在opencv中，二值图像/灰度图像以二维数组形式进行存放，彩色图像以三维数组形式进行存放（X * Y * 3 ，通道存放顺序为BGR，X，Y为图像x，y轴向上的像素点个数）


功能实现，利用numpy库生成一个3x3的全零矩阵，利用imshow显示图像，将该图像[1, 1] 位置的像素值置为255，显示改变后的图像

```
import numpy 
import cv2

if __name__ == '__main__':
    img = np.zeros((3,3),dtype = np.uint8)
    cv2.imshow('img1',img)
    img[1,1] = 255
    cv2.imshow('img2',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

```

功能实现，以单通道模式读取lena_gray.jpg，将lean_gray.jpg中100行，100列到200行，120列置为白色，显示图像

```
import cv2

if __name__ == '__main__':
    img = cv2.imread('lena_gray.jpg',-1)
    if img is not None：
        cv2.imshow('lena',img)
        img[100:120,100:120] = 255
        cv2.imshow('lena1',img)
        cv2.waitKey()
        cv2.desroyAllWindows()
    else : 
        print('Failed to read image file')


```

4. 像素处理

```
import cv2
import numpy as np

img = np.zeros((300,300,3),dtype=np.uint8)
img[:,:100,0] = 255
img[:,101:200,1] = 255
img[:,201:300,2] = 255
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()

```


5. 像素的访问

使用numpy.array访问像素。 numpy.array提供了item()和itemset()函数来访问和修改像素，item()读取像素点值 语法： val = image.item(行，列) itemset() 修改像素值 语法： image.itemset(索引值，新值)

```
import cv2
import numpy as np

img = np.zeros((5,5),dtype= np.uint8)
img.itemset((3,3),255)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(img.item(3,3))


```

```
import cv2
import numpy as np

img = np.zeros((5,5,3),dtype=np.uint8)
img.itemset((3,3,0),255)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()
print(img.item(3,3,0))


```


```
import cv2 # opencv中按BGR排布，蓝绿红
import numpy as np

img = np.zeros((5,5,3),dtype=np.uint8)

img.itemset((0,0,0),255) # 蓝色区块，位于（0 x轴，0 y轴，0通道位置） 注： 如下图，对于
img.itemset((1,1,1),255) # 绿色区块，位于（1 x轴， 1 y轴，1通道位置 ） 
img.itemset((2,2,0),255) # 
img.itemset((2,2,1),255) # 中间的白色区块。 
img.itemset((2,2,2),255) # 
img.itemset((3,3,2),255)
img.itemset((4,4,1),255)
img.itemset((0,4,0),255)
img.itemset((1,3,2),255)
img.itemset((3,1,1),255)
img.itemset((4,0,0),255)

cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',500,500)
cv2.imshow('img',img)
cv2.waitKey()
cv2.destroyAllWindows()
```
6. 感兴趣区域（ROT）
提取阿宝和小伙伴们

```
import cv2

if __name__ == '__main__':
img = cv2.imread('kongfu_panda.jpg)
if img is not None:
    ROI1 = img[79,510,345:670，：]
    ROI2 = img[70:340,35:250,:]
    ROI3 = img[227:499,213:356,:]
    ROI4 = img[250,510,605:751,:]
    ROI5 = img[53:421,675:969,:]
    cv2.imshow('ROI1',ROI1)
    cv2.imshow('ROI2',ROI2)
    cv2.imshow('ROI3',ROI3)
    cv2.imshow('ROI4',ROI4)
    cv2.imshow('ROI5',ROI5)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print('failed to load image')

```

数据脱敏

```
import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('police_story.png')
    if img is not None:
        face = np.random.randint(0，255，(456,445,3))
        img[158:614,364:809,:] = face
        cv2.namedWindow('Data Masking',0) # 创建一个窗口容器，第二个参数的取值：0和1
        cv2.resizeWindow('Data Masking',400,400) # 调整窗口尺寸
        cv2.imshow('Data Masking',img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        print('falied to load image')

```
7. 通道操作

RGB图像可以拆分为R通道，G通道，B通道，通道拆分方法：索引拆分，函数拆分。
通过索引拆分，b = img[:, :,0], g = img[:,:, 1], r = img[:, :,2]

功能实现，将lena_color.jpg利用索引方法进行通道拆分

```

import cv2 
lena = cv2.imread('lena_color.jpg')

if lena is not None : 
    cv2.imshow('lena',lena)
    b = lena[:,:,0]
    g = lena[:,:,1]
    r = lena[:,:,2]
    cv2.imshow('R',r)
    cv2.imshow('G',g)
    cv2.imshow('B',b)
else:
    print('failed to load image')

cv2.waitKey()
cv2.destroyAllwindows()

```


功能实现，将lena_color.jpg利用函数方法进行通道拆分

```

import cv2 
lena = cv2.imread('lena_color.jpg')

if lena is not None : 
    cv2.imshow('lena',lena)

    [x,y,z] lena.shape
    b1,g1,r1 = cv2.split(lena)
    b = np.zeros((x,y,z),dtype=np.uint8)
    g = np.zeros((x,y,z),dtype=np.uint8)
    r = np.zeros((x,y,z),dtype=np.uint8)

    b[:,:,0] = b1
    g[:,:,1] = g1
    r[:,:,2] = r1

    cv2.imshow('R',r)
    cv2.imshow('G',g)
    cv2.imshow('B',b)
else:
    print('failed to load image')

cv2.waitKey()
cv2.destroyAllwindows()

```
8. 获取图像属性
尺寸，.shape 行数，列数，通道数。 总像素数目 ，.size 行数*列数*通道数，像素数据类型.dtype

功能实现，观察lena_color.jpg图像的属性

```
import cv2 
img = cv2.imread('lena_color.jpg')
if img is not None:
    print('img.shape',img.shape)
    print('img.size',img.size)
    print('img.dtype',img.dtype)



```
9. 改变图像大小
方法： cv2.resize(原始图像,(x,y)) ,x: 水平方向分辨率， y： 垂直方向分辨率

功能实现： 读取lena_color.jpg 图像，显示图像及其哦大小，将该图像的尺寸修改为600*600,显示图像及其大小

```
import cv2

img  = cv2.imread('lena_color.jpg')
print(img.shape)
img1= cv2.resize(img,(600,600))
print(img.shape)
cv2.imshow('img',img)
cv2.imshow('img1',img1)
cv2.waitKey()
cv2.destroyAllwindows()

```


## 图像运算

1. 图像加法运算
方法： 通过运算符+ 进行加法运算，通过cv2.add()函数进行加法运算。

```
import numpy as np

img1 = np.array([[178,83,29],[202,200,158],[27,177,162]])
img2 = np.array([[26,48,57],[52,153,8],[10,232,7]])
print('img1\n',img1)
print('img\n',img2)
print('img1+ img2\n',img1+img2)

```
2. 图像加权和
方法 img = cv2.addWeighted(img1,a,img2,b,c) a为加权系数，b为加权系数，c为亮度调节。 注意的是img1和img2的大小需要一致。

3. 按位逻辑运算
位函数运算， cv2.bitwise_add() 按位与， cv2.bitwise_or按位或 , cv2.bitwise_xor()按位异或, cv2.bitwise_not()按位取反。
4. 掩码


5. 位平面分解
24位RGB彩色图像由RGB三个通道组成，每个通道中使用8个二进制，对每个像素进行表示，其取值范围为[0 ~ 255] ,通过提取灰度图像素点二进制像素值的每一位比特位的组合，可以得到多个位平面图像。图像中的全部像素值的每一位比特位的组合，可以得到多个位平面。8位灰度图可以分解为8个位平面。
6. 数字水印
初始化，载体图像预处理，建立低位拍卖你置零矩阵，载体图像最低有效位置0，读取水印图像，修改水印图像大小，水印图像二值化，嵌入水印，显示原始图像，含水印图像，建立水印位平面提取矩阵，提取低位平面，提取图像显示优化，显示水印。


```

import cv2
import numpy ass np

img = cv2.imread('lena_gray.jpg',0)
mask1 = np.ones(img.shape,dtype= np.uint8)*254

imgH7 = cv2.bitwise_and(img,mash1)
img1 = cv2.imread('treasure1.jpg',0)

x,y = img1.shape
img2 = cv2.resize(img1,[y,x])
img2[img2<100] = 0
img[img2>100] = 1

img_watermark = cv2.bitwise_or(imgH7,img2)
cv2.imshow('original image ',img)
cv2.imshow('waterMark Image ', img_watermark)

extract_mask = np.ones(img.shape,dtype=np.uint8)
watermark = cv2.bitwise_and(img_watermark,extract_mask)
watermark[watermark>0] = 255

cv2.imshow('extrated_watermark',watermark)

cv2.waitKey()
cv2.destroyAllWindows()

```








7. 图像加/解密
方案一 非运算 img = cv2.bitwise_not(img1) 特点：只需要有一个参数，方法：直接按位取反。

```
import cv2
import numpy as np

img = cv2.imread('lena_color.jpg')
mask = np.random.randint(0,256,img.shape,dtype=np.uint8)
encrypt_img = cv2.bitwise(img,mask)
decrype_img = cv2.bitwise_xor(encrypt_img,mask)
cv2.imshow('encrypt',encrypt_img)
cv2.imshow('decrpty',decrype_img)

cv2.waitKey()
cv2.destroyAllWindows()


```

方案二：异或 img = cv2.bitwise_xor(img1.img2). 特点：两个参数，方法：生成一个与图像一样大的随机密钥矩阵，将密钥矩阵与图像按位求异或。

```

import cv2
import numpy as np

img = cv2.imread('lena_color.jpg')
mask = np.random.randint(0,255,img.shape,dtype=np.uint8)
encrypt_img = cv2.bitwise_xor(img,mask)
decrype_img = cv2.bitwise_xor(encrypt_img,mask)
cv2.imshow('encrypt',encrypt_img)
cv2.imshow('decrpty',decrype_img)

cv2.waitKey()
cv2.destroyAllWindows()


```

## 几何变换

1. 缩放
方法 dst = cv2.resize(src,dsize,fx=n,fy=m,interpolation) dst： 目标图像， src:源目标， dsize:目标图像大小（分辨率） ,fx = n 将水平方向放大到n倍（该参数可选），fx = m 将垂直方向放大到m倍

2. 翻转
方法 dst  = cv2.flip(src,filpCode) dst : 目标图像， src： 源图像，filpCode旋转类型
3. 仿射，平移，旋转
通过几何变换实现平移、旋转。注： 该变换会具有平直性和平行性。 平直性： 变换后，直线还是直线， 平行性： 变换后，平行线还是平行线
方法： dst = cv2.warpAffine(src,M,dsize) dst: 目标图像 src： 源图像， M： 变换矩阵， dszie： 输出图像大小

4. 透视


## 阈值处理（完成）


二值化处理：对于每个像素，选择一个阈值。如果像素值大于阈值，将其设置为一个值（通常是白色），如果像素值小于或等于阈值，将其设置为另一个值（通常是黑色）。这样就得到了一个二值图像。

反二值化处理：这是二值化处理的反向操作。如果像素值大于阈值，将其设置为一个值（通常是黑色），如果像素值小于或等于阈值，将其设置为另一个值（通常是白色）。

截断阈值处理：对于每个像素，如果其值大于阈值，将其设置为阈值。如果像素值小于或等于阈值，保持其原值不变。

超阈值零处理：对于每个像素，如果其值大于阈值，保持其原值不变。如果像素值小于或等于阈值，将其设置为零。

低阈值零处理：这是超阈值零处理的反向操作。如果像素值大于阈值，将其设置为零。如果像素值小于或等于阈值，保持其原值不变。

自适应阈值处理：这是一种更复杂的方法，它不使用固定的阈值。相反，它根据像素周围的小区域计算阈值。因此，对于同一张图片上的不同区域，可以有不同的阈值。这对于当图像的光照条件变化很大时，例如，一半是明亮的，一半是暗淡的图像，非常有用。

```
import cv2



# 读取图像

image = cv2.imread('e1.jpg', cv2.IMREAD_GRAYSCALE)



# 二值化处理

_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)



# 反二值化处理

_, binary_inv_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)



# 截断阈值处理

_, trunc_image = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)



# 超阈值零处理

_, tozero_inv_image = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)



# 低阈值零处理

_, tozero_image = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)



# 自适应阈值处理

adaptive_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)



# 保存处理后的图像

cv2.imwrite('binary_image.jpg', binary_image)

cv2.imwrite('binary_inv_image.jpg', binary_inv_image)

cv2.imwrite('trunc_image.jpg', trunc_image)

cv2.imwrite('tozero_inv_image.jpg', tozero_inv_image)

cv2.imwrite('tozero_image.jpg', tozero_image)

cv2.imwrite('adaptive_image.jpg', adaptive_image)

```

## 图像增强
1. 直方图及直方图均衡化
计算直方图的方法： hist = cv2.calcHist(images,channels,mask,histSize,ranges,accumulate)
images : 原始图像，使用[]括起来， channels： 图像的通道，灰度-[0], 彩色图像[0],[1],[2],mask： 掩码图像， histSize： 图像灰度分组级别， range：像素值取值范围。

```
import cv2
import matplotlib.pyplot as plt

bean1 = cv2.imread('beans1.jpg')
bean2 = cv2.imread('beans2.jpg')

h1 = cv2.calcHist([bean1],[0],None,[256],[0,256])
his1 = cv2.normalize(h2,None,1,0,cv2.NoRM_L1)

h2 = cv2.calcHist([bean2],[0],None,[256],[0,256])
his2 = cv2.normalize(h2,None,1,0,cv2.NORM_L1)

fig1 = plt.figure()
fig1.add_subplot(1,2,1)
plt.imshow(bean1)
plt.title('beab1')
fig1.add_subplot(1,2,2)
plt.plot(his1)
plt.title('bean1_hist')

fig2 = plt.figure()
fig2.add_subplot(1,2,1)
plt.imshow(bean2)
plt.title('beab2')
fig2.add_subplot(1,2,2)
plt.plot(his2)
plt.title('bean2_hist')

plt.show()
```



2. 图像空间域平滑
在尽量保留原有信息的情况下，过滤掉图像内部的噪声，这一过程称为对图像的平滑处理。

均值滤波，使用当前像素点周围N*N个像素的均值代替当前像素值。均值滤波函数 dst = cv2.blur(src,ksize,anchor,borderType) dst: 返回图像， src： 原始图像， ksize： 滤波核大小。

方框滤波， 高斯滤波，中值滤波，双边滤波

方框滤波是一种简单的线性滤波器，其核心操作是将图像中每个像素点替换为其邻域内所有像素点的平均值。它对图像的每个像素点应用一个方形的卷积核，计算核覆盖区域内所有像素的平均值。

高斯滤波
高斯滤波是一种使用高斯函数作为权重分布的线性滤波器。与方框滤波不同，高斯滤波器对距离中心点越远的像素赋予越小的权重，从而更加平滑地处理图像噪声，同时保留更多的图像细节。

中值滤波
中值滤波是一种非线性滤波方法，其核心思想是将图像中的每个像素点替换为其邻域内像素值的中值。中值滤波在去除图像中的椒盐噪声方面特别有效，因为它能够很好地保留边缘信息。

双边滤波
双边滤波是一种既能平滑图像又能保留边缘细节的非线性滤波器。它结合了空间高斯加权和灰度差异高斯加权，使得相近的像素值保留而不同的像素值被平滑。


```

import cv2
import numpy as np

# 读取图像
image = cv2.imread('image.jpg')

# 方框滤波
box_filtered = cv2.boxFilter(image, -1, (5, 5))

# 高斯滤波
gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

# 中值滤波
median_filtered = cv2.medianBlur(image, 5)

# 双边滤波
bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)

# 显示结果
cv2.imshow('Original', image)
cv2.imshow('Box Filtered', box_filtered)
cv2.imshow('Gaussian Filtered', gaussian_filtered)
cv2.imshow('Median Filtered', median_filtered)
cv2.imshow('Bilateral Filtered', bilateral_filtered)
cv2.waitKey(0)
cv2.destroyAllWindows()


```
## 视频处理（完成）

```
import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:

    # 从摄像头读取一帧
    ret, frame = cap.read()
    # 检查帧是否成功读取
    if not ret:
        print("无法从摄像头读取帧")
        break

    # 获取帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 帧上显示信息
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Width: {width}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Height: {height}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # 显示视频
    cv2.imshow('Video', frame)
    # 按下q键退出循环

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

# 释放摄像头并关闭窗口

cap.release()

cv2.destroyAllWindows()

```
## 形态学操作

1. 腐蚀
```
import cv2
import numpy as np

# 读取图像
image = cv2.imread('deeplearingh_work/img/insulator.jpg',0)  # 0 表示以灰度模式读取

# 定义腐蚀操作的结构元素
# 这里我们使用一个 5x5 的矩形结构元素
kernel = np.ones((5, 5), np.uint8)

# 进行腐蚀操作

eroded_image = cv2.erode(image, kernel, iterations=8)  # iterations 是腐蚀操作的次数
# 显示原始图像和腐蚀后的图像
cv2.imshow('Original Image', image)

cv2.imshow('Eroded Image', eroded_image)

# 等待按键按下，然后关闭窗口
cv2.waitKey(0)

cv2.destroyAllWindows()
```
2. 膨胀
膨胀是形态学操作的一种，用于增加图像中前景对象的边界。膨胀操作的效果是使对象变得更大，主要用于填补图像中的小黑洞或连接断开的对象。
```
# 进行膨胀操作
dilated_image = cv2.dilate(image, kernel, iterations=8)  # iterations 是膨胀操作的次数

# 显示原始图像和膨胀后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Dilated Image', dilated_image)

# 等待按键按下，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

```



3. 开运算
开运算是先进行腐蚀再进行膨胀的操作，主要用于去除图像中的小噪声点。
```
# 进行开运算操作
opening_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

# 显示原始图像和开运算后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Opening Image', opening_image)

# 等待按键按下，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


```

4. 闭运算
闭运算是先进行膨胀再进行腐蚀的操作，主要用于填补图像中的小黑洞或连接断开的对象。


```
# 进行闭运算操作
closing_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# 显示原始图像和闭运算后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Closing Image', closing_image)

# 等待按键按下，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


```
5. 形态学运算
形态学梯度是膨胀图像与腐蚀图像之间的差异。它用来突出物体的边缘。

```
# 进行形态学梯度操作
gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)

# 显示原始图像和形态学梯度图像
cv2.imshow('Original Image', image)
cv2.imshow('Gradient Image', gradient_image)

# 等待按键按下，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()

```

6. 礼帽运算
礼帽运算是原始图像与其开运算结果之间的差异，用于提取比邻域结构元素更亮的区域。
```
import cv2

import numpy as np



# 读取图像

image = cv2.imread('deeplearingh_work\img\lena.bmp', cv2.IMREAD_GRAYSCALE)

# 定义礼帽运算的核
kernel = np.ones((5, 5), np.uint8)

# 进行礼帽运算
tophat = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
# 显示原始图像和礼帽运算后的图像
cv2.imshow('Original Image', image)
cv2.imshow('Top Hat Image', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()

```
7. 黑帽运算
黑帽运算是原始图像与其闭运算结果之间的差异，用于提取比邻域结构元素更暗的区域。黑帽运算可以用于检测图像中的凹陷或局部阴影。
```
import cv2
import numpy as np

# 读取图像
image = cv2.imread('deeplearingh_work\img/blackhat.bmp', cv2.IMREAD_GRAYSCALE)
# 定义黑帽运算的核
kernel = np.ones((5, 5), np.uint8)
# 进行黑帽运算
blackhat = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
# 显示原始图像和黑帽运算后的图像

cv2.imshow('Original Image', image)
cv2.imshow('Black Hat Image', blackhat)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
8. 通用形态函数

在图像处理和计算机视觉中，形态学操作是非常重要的技术。为了简化和统一不同形态学操作，可以定义一个通用形态学函数。这个函数可以处理各种形态学操作，如腐蚀、膨胀、开运算、闭运算、礼帽运算和黑帽运算。


9. 核函数
```
import cv2

import numpy as np
# 读取图像

image = cv2.imread('deeplearingh_work\img/tophat.bmp', cv2.IMREAD_GRAYSCALE)

# 定义核函数

kernel1 = np.ones((5, 5), np.uint8)

kernel2 = np.array([[0, 1, 0],

                     [1, 1, 1],

                     [0, 1, 0]], np.uint8)

kernel3 = np.array([[1, 0, 1],

                     [0, 1, 0],

                     [1, 0, 1]], np.uint8)

# 对图像进行腐蚀和膨胀操作，并记录结果
erosion_results = []
dilation_results = []

for kernel in [kernel1, kernel2, kernel3]:

    erosion = cv2.erode(image, kernel, iterations=1)

    dilation = cv2.dilate(image, kernel, iterations=1)

    erosion_results.append(erosion)

    dilation_results.append(dilation)

# 调整图像尺寸
resized_image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

# 显示结果

cv2.imshow('Original Image', resized_image)
for i in range(3):

    cv2.imshow(f'Kernel {i+1} Erosion', cv2.resize(erosion_results[i], (image.shape[1]//2, image.shape[0]//2)))

    cv2.imshow(f'Kernel {i+1} Dilation', cv2.resize(dilation_results[i], (image.shape[1]//2, image.shape[0]//2)))

cv2.waitKey(0)

cv2.destroyAllWindows()
```

## 图像梯度

1. 图像梯度
图像梯度是图像在某个方向上的变化率。图像梯度在图像处理中的应用包括边缘检测、特征提取和图像分割。梯度在每个像素点上都有一个方向和幅值，幅值表示变化的强度，方向表示变化的方向。


2. Sobel 算子
Sobel算子是一种用于边缘检测的离散微分算子。它结合了高斯平滑和微分操作，以减少噪声影响。Sobel算子通过在水平方向和垂直方向上的微分，来计算图像的梯度。一种结合高斯平滑和微分操作的边缘检测方法。

```

import cv2
import numpy as np

# 读取图像
image = cv2.imread('deeplearingh_work/img/insulator.jpg', 0)

# 使用Sobel算子计算x方向和y方向的梯度
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# 计算梯度的幅值和方向
magnitude = cv2.magnitude(sobelx, sobely)
angle = cv2.phase(sobelx, sobely, angleInDegrees=True)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Gradient Magnitude', magnitude)
cv2.waitKey(0)
cv2.destroyAllWindows()

```

3. Scharr算子

Scharr算子是Sobel算子的改进版，在计算图像梯度时对高频噪声更为敏感。它在某些应用中比Sobel算子具有更好的性能，特别是对于噪声较大的图像。Sobel算子的改进版，对高频噪声更为敏感。

```
# 使用Scharr算子计算x方向和y方向的梯度
scharrx = cv2.Scharr(image, cv2.CV_64F, 1, 0)
scharry = cv2.Scharr(image, cv2.CV_64F, 0, 1)

# 计算梯度的幅值和方向
magnitude_scharr = cv2.magnitude(scharrx, scharry)

# 显示结果
cv2.imshow('Scharr X', scharrx)
cv2.imshow('Scharr Y', scharry)
cv2.imshow('Gradient Magnitude Scharr', magnitude_scharr)
cv2.waitKey(0)
cv2.destroyAllWindows()




```

4. Laplacin算子

Laplacian算子是一种二阶微分算子，用于计算图像的二阶导数。它用于检测图像中的区域变化，并突出显示图像中的边缘。Laplacian算子通过卷积运算实现，其卷积核可以检测图像中的急剧变化。一种二阶微分算子，用于检测图像中的边缘。

```
# 使用Laplacian算子计算图像的二阶导数
laplacian = cv2.Laplacian(image, cv2.CV_64F)

# 显示结果
cv2.imshow('Laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()


```

## 边缘处理设备
边缘计算，是指在靠近物或数据源头的一侧.采用网络﹑计算﹑存储﹑应用核心能力为一体的开放平台，就近提供最近端服务。其应用程序在边缘侧发起，产生更快的网络服务响应，满足行业在实时业务﹑应用智能﹑安全与隐私保护等方面的基本需求·边缘计算处于物理实体和工业连接之间，或处于物理实体的顶端О而云端计算，仍然可以访问边缘计算的历史数据。

智能化：AI应用程序比传统的应用程序更加强大和灵活。实时洞察力：实时响应用户的需求，降低成本、网络成本。增加隐私：减少数据传输过程中的泄露风险。
