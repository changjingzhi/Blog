---
title: 图像处理-数据预处理
date: 2024-04-23 16:07:19
tags: 深度学习
---
## 基本知识
在深度学习中，图像数据通常以多维数组（在Python中通常使用Numpy数组）的形式表示，这个数组的形状（shape）取决于图像的维度和颜色通道数。
灰度图像：对于灰度图像（也就是黑白图像），shape通常是两维的，表示图像的高度和宽度。例如，一个256x256像素的灰度图像的shape将是(256, 256)。灰度图像的像素值通常在0到255之间，其中0表示黑色，255表示白色，中间的值表示不同的灰度级别。这是因为每个像素通常由8位（一个字节）表示，所以可以有256（即$2^8$）个不同的可能值。然而，这并不是唯一的表示方式。有时，为了方便计算，我们可能会将像素值归一化到0到1之间。在这种情况下，0仍然表示黑色，1表示白色，中间的值表示不同的灰度级别。
彩色图像：对于彩色图像，通常使用RGB（红，绿，蓝）三个颜色通道，所以shape是三维的。例如，一个256x256像素的RGB彩色图像的shape将是(256, 256, 3)。这里的3代表三个颜色通道。彩色图像通常由三个颜色通道组成：红色（R），绿色（G）和蓝色（B）。每个通道的像素值通常在0到255之间，其中0表示该颜色的完全缺失，255表示该颜色的最大强度。所以，一个RGB颜色图像的像素值范围在理论上是0到255的三维空间，即(0,0,0)到(255,255,255)。同样，有时我们也会将每个颜色通道的像素值归一化到0到1之间。在这种情况下，(0,0,0)表示黑色，(1,1,1)表示白色，其他值表示不同的颜色。需要注意的是，虽然RGB是最常用的颜色空间，但也有其他的颜色空间，如HSV（色相，饱和度，亮度）或者CMYK（青色，品红，黄色，黑色），它们的取值范围可能会有所不同。
图像批量：在深度学习中，我们通常会一次处理多个图像，这就是所谓的批量（batch）。在这种情况下，图像数据的shape将是四维的：(批量大小, 高度, 宽度, 颜色通道数)。例如，如果我们有32个256x256像素的RGB图像，那么这个批量的shape将是(32, 256, 256, 3)。
## 显示彩色图像
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
![代码结果](pic/sdxxtx1.png)
## 对图像进行截取操作
剪裁图片
![功夫熊猫](pic/kongfu_panda.jpg)

```
import cv2
import matplotlib.pyplot as plt
def show_plt(path):
    # 显示原始图片
    image_path = path
    image = plt.imread(image_path)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def mian(path):
    img = cv2.imread(path)
    if img is not None:
        ROI1 = img[79:510,345:670,:] # 可以看出 先是y轴，而后是x轴，最后是通道
        ROI2 = img[70:340,35:250,:] 
        ROI3 = img[227:499,213:356,:]
        ROI4 = img[250:510,605:751,:]
        ROI5 = img[53:421,675:969,:]

        cv2.imshow('ROI1',ROI1)
        cv2.imshow('ROI2',ROI2)
        cv2.imshow('ROI3',ROI3)
        cv2.imshow('ROI4',ROI4)
        cv2.imshow('ROI5',ROI5)
        key = cv2.waitKey(0) # 等待按键，
        if key == ord('q'):
            cv2.destroyAllWindows() # 关闭窗口
    else:
        print('failed to load image')

if __name__ == '__main__':
    show_plt('1/kongfu_panda.jpg')
    mian('1/kongfu_panda.jpg')

```
结果图片
![剪切结果](pic/tpcl1.png)

![处理图片](pic/tpcl2.png)
```
import cv2
import numpy as np

def main(path):
    img = cv2.imread(path)
    print(img.shape)
    if img is not None:
        face = np.random.randint(0,255,(600,445,3)) # 随机产生掩盖矩阵
        img[50:650,364:809,:] = face # 将图片的重新赋值
        cv2.namedWindow('Data Masking',0)
        cv2.resizeWindow('Data Masking',500,500) # 对现实的图片进行缩放，缩放到（500，500）
        cv2.imshow('Data Masking',img)
        cv2.waitKey() # 这样写的原因是保持图片的一直显示,否则一闪而逝
        cv2.destroyAllWindows()
    else:
        print('falied to load image')

if __name__ == '__main__':
    main("1/police_story.png")
```
![处理图片](pic/tpcl3.png)

显示RGB通道的图片内容
```
import cv2

lena = cv2.imread('1/lena_color.jpg')

if lena is not None:
    cv2.imshow('lean',lena)
    print('img.shape',img.shape)
    print('img.size',img.size)
    print('omg.dtype',img.dtype)
    b = lena[:,:,0]
    g = lena[:,:,1]
    r = lena[:,:,2]
    cv2.imshow('r',r)
    cv2.imshow('g',g)
    cv2.imshow('b',b)
    img1 = cv2.resize(img,(600,600)) # 调整图像本身的大小
     img1 = cv2.merge([g,r,b]) # cv2.merge可以调整BGR图像通道为自定义的[g,r,b]
else:
    print('failed to load image')

cv2.waitKey()
cv2.destroyAllWindows()
# 可以看到基本的信息都包含，只不过是灰度的图像
```
![通道显示](pic/tpcl5.png)

```
import numpy as np
import cv2

img1=np.array([[178,83,29],[202,200,158],[27,177,162]],dtype=np.uint8)
img2=np.array([[26,48,57],[52,153,8],[10,232,7]],dtype=np.uint8)

print("img1\n", img1)
print("img1\n", img2)
print("img1+img2\n",img1+img2)


cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.imshow("img",img1+img2)
cv2.resizeWindow('img',500,500)
key = cv2.waitKey()
if key == ord('q'):
    cv2.destroyAllWindows()
```
![3x3矩阵显示](pic/tpcl6.png)



## 处理方法
1. 二值化处理：这是最基本的阈值处理方法。对于每个像素，我们选择一个阈值。如果像素值大于阈值，我们将其设置为一个值（通常是白色），如果像素值小于或等于阈值，我们将其设置为另一个值（通常是黑色）。这样我们就得到了一个二值图像。
2. 反二值化处理：这是二值化处理的反向操作。如果像素值大于阈值，我们将其设置为一个值（通常是黑色），如果像素值小于或等于阈值，我们将其设置为另一个值（通常是白色）。
3. 截断阈值处理：对于每个像素，如果其值大于阈值，我们将其设置为阈值。如果像素值小于或等于阈值，我们保持其原值不变。
4. 超阈值零处理：对于每个像素，如果其值大于阈值，我们保持其原值不变。如果像素值小于或等于阈值，我们将其设置为零。
5. 低阈值零处理：这是超阈值零处理的反向操作。如果像素值大于阈值，我们将其设置为零。如果像素值小于或等于阈值，我们保持其原值不变。
6. 自适应阈值处理：这是一种更复杂的方法，它不使用固定的阈值。相反，它根据像素周围的小区域计算阈值。因此，对于同一张图片上的不同区域，我们可以有不同的阈值。这对于当图像的光照条件变化很大时，例如，一半是明亮的，一半是暗淡的图像，非常有用。



## 实现代码
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
![原图](pic/e1.jpg)
![二值化处理图像](pic/binary_image.jpg)
![反二值化处理图像](pic/binary_inv_image.jpg)
![截断阈值处理图像](pic/trunc_image.jpg)
![超阈值处理图像](pic/tozero_inv_image.jpg)
![低阈值零处理图像](pic/tozero_image.jpg)
![自适应阈值处理图像](pic/adaptive_image.jpg)

### 挖更大的坑，opencv库。

### 彩色图像怎么转换为二维图像的？
首先灰度图像中的一个像素点的范围为0-255，彩色图像可以理解为3个灰度图重合。

### 需要深度解析代码中的含义，比如一个参数有什么用处。