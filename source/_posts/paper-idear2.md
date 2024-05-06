---
title: 论文思路——数据预处理
date: 2024-04-28 21:49:52
tags: 填坑
---

## 目前的数据预处理有
This may be time-domain features(时域特征)[45——2021年](dx.doi.org/10.3390/diagnostics11081437)
1. absolute band power(绝对功率谱)
2. Discrete Wavelet Transform (离散小波变换)[37-2022-IEEE](https://ieeexplore.ieee.org/document/9857825)
3. permutation entropy (排列熵) or spectral entropy (熵谱)[21-2021-Complexity of EEG dynamics for early diagnosis of Alzheimer’s disease using permutation entropy neuromarker,’’ C]
4. coherence anaylysis features (相干性分析) such as spectrall coherece(光谱相干性)
5. RBP (注：按照下面的频率划分，the shape of data is [T , B ,C]=[T, 5, C]，Delta: 0.5 – 4 Hz Theta: 4 – 8 Hz Alpha: 8 – 13 Hz Beta: 13-25 Hz Gamma: 25-45 Hz,或许这个数据预处理的方法会生效 )
6. spectral coherence connectivity (光谱相干性，光谱相干性建立在PSD，PSD是功率谱密度)
7. FFT (Fast Fourier Transform) 快速傅里叶变换

## 数据预处理组合思路
1. Morlet Wavelet Transform ——> RBP
2. Welch PSD ——> SSC


## 需要注意的前提知识。
1. AD patients may exhaibit changes in the EEG signal, such as reduced(减少) alpha power and increased (增加) theta power.[39-2021-Clinical Neurophysiology-3区 ](https://www.sciencedirect.com/science/article/abs/pii/S1388245721005976) 
It can be visually observed that AD group has lower delta connectivity than CN group in multiple brain locations.This finding is supported by the literature  [53-2016]https://www.sciencedirect.com/science/article/pii/S1388245715009839()
2. Train, validation and test sets are created.
3. The time frequency transforms and the feature extraction steps were implemented in Python 3.10 using the MNE library.
4. GFlops (计算量)
5. hyperparameters (超参数)
## 挖坑

### 怎么计算模型的计算量。


## 看论文的心态
Because it is an English paper, there is a kind of resistance. Take your time.
由于是英文论文，有种抗拒的心态。慢慢看吧。


## 代码实现
1. FFT快速傅里叶变换[参考博客](https://zhuanlan.zhihu.com/p/347091298)
```
import os
import numpy as np
from tqdm import tqdm

# 定义输入和输出文件夹路径
input_folder = 'data_cut_npy/AD'  # 输入文件夹路径，包含要进行 FFT 变换的 .npy 文件
output_folder = 'data_FFT_npy/AD'  # 输出文件夹路径，用于保存变换后的数据

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 获取输入文件夹中的所有 .npy 文件
file_list = os.listdir(input_folder)
npy_files = [file for file in file_list if file.endswith('.npy')]

# 遍历每个 .npy 文件进行 FFT 变换并保存
for file_name in tqdm(npy_files, desc='Processing', unit='file'):
    # 读取 .npy 文件
    file_path = os.path.join(input_folder, file_name)
    data = np.load(file_path)

    # 对数据中的每一行进行 FFT 变换
    fft_data = np.apply_along_axis(np.fft.fft, axis=0, arr=data)

    # 获取 FFT 结果的幅值
    fft_magnitude = np.abs(fft_data)

    # 构造输出文件路径
    output_file_name = file_name.replace('.npy', '_fft.npy')
    output_file_path = os.path.join(output_folder, output_file_name)

    # 保存 FFT 结果
    np.save(output_file_path, fft_magnitude)

```

2. RBP 按频率划分[T, C, B]

## 什么是时域和频域
[参考资料](https://zhuanlan.zhihu.com/p/401681076)
原始的EEG数据是由很多个样本点数所构成的一个有限的离散的时间序列数据。至于样本点数的多少，则由采样率所决定，比如采样率为1000Hz，那么每秒就有1000个数据样本点。其中，每个样本点数据代表的是脑电波幅的大小，物理学上称为电压值，单位为伏特（V），由于脑电信号通常较弱，所以更常使用的单位为微伏（uV）。

```
import os
import mne
import numpy as np
from tqdm import tqdm

def process_and_save_set_files(input_folder, output_folder):
    def process_data(raw):
        # 获取信号数据
        data = raw.get_data() # shape: (n_channels, n_samples)

        # 定义频率范围
        freq_ranges = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 25),
            'Gamma': (25, 45)
        }

        # 初始化频带划分后的数据
        data_bands = []

        # 对每个频带进行处理
        for _, (fmin, fmax) in freq_ranges.items():
            # 使用 mne.filter 函数进行滤波
            filtered_data = mne.filter.filter_data(data, raw.info['sfreq'], fmin, fmax)

            # 将滤波后的数据存储到列表中
            data_bands.append(filtered_data)

        # 将列表转换为numpy数组
        data_bands = np.array(data_bands)

        return data_bands

    def find_set_files(root_folder):
        set_files = []
        for root, dirs, files in os.walk(root_folder):
            for file in files:
                if file.endswith('.set'):
                    set_files.append(os.path.join(root, file))
        return set_files

    # 找到所有的 .set 文件
    set_files = find_set_files(input_folder)

    for file_path in tqdm(set_files, desc='Processing files'):
        # 读取 .set 文件
        raw = mne.io.read_raw_eeglab(file_path, preload=True)

        # 处理数据
        processed_data = process_data(raw)

        # 构造新的文件路径
        npy_file_name = os.path.basename(file_path).replace('.set', '.npy')
        npy_file_path = os.path.join(output_folder, npy_file_name)

        # 保存为 .npy 文件
        np.save(npy_file_path, processed_data)

if __name__ == '__main__':
    # 指定您的 AD 文件夹路径和保存 .npy 文件的新文件夹路径
    ad_folder = r'C:/Users/Administrator/Desktop/RBP/data/FDT'
    output_folder = r'C:/Users/Administrator/Desktop/RBP/data_npy/FDT'

    # 处理并保存数据
    process_and_save_set_files(ad_folder, output_folder)

```
## 赫兹(HZ)的定义是什么？

Hz 是频率的单位。频率是指电脉冲，交流电波形，电磁波，声波和机械的振动周期循环时，1秒钟重复的次数。1Hz代表每秒钟周期震动1次，60Hz代表每秒周期震动60次。
对于声音，人类的听觉范围为20Hz～20000Hz，低于这个范围叫做次声波，高于这个范围的叫做超声波。