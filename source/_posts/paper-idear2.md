---
title: 论文思路——数据预处理
date: 2024-04-28 21:49:52
tags: 论文思路
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
1. AD patients may exhaibit changes in the EEG signal, such as reduced(减少) alpha power (Alpha: 8 – 13 Hz ) and increased (增加) theta power (: 4 – 8 Hz).[39-2021-Clinical Neurophysiology-3区 ](https://www.sciencedirect.com/science/article/abs/pii/S1388245721005976) 
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
0. 提取set文件
```
import os
import shutil

def find_set_files(root_folder):
    set_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.set'):
                set_files.append(os.path.join(root, file))
    return set_files

# 输入文件夹列表
input_folders = ['AD','CN','FDT']

# 目标文件夹列表
target_folders = ['data_processing/AD','data_processing/CN','data_processing/FDT']

# 确保输入和目标文件夹数量匹配
assert len(input_folders) == len(target_folders), "输入文件夹和目标文件夹数量不匹配"

# 遍历输入文件夹列表
for input_folder, target_folder in zip(input_folders, target_folders):
    # 找到所有的 .set 文件
    set_files = find_set_files(input_folder)
    
    # 确保目标文件夹存在
    os.makedirs(target_folder, exist_ok=True)
    
    # 打印所有找到的 .set 文件路径
    for file_path in set_files:
        print(file_path)
        
        # 将找到的 .set 文件复制到目标文件夹中
        shutil.copy(file_path, target_folder)
        print(f"复制 {file_path} 到 {target_folder}")


```

AD交集
```
# 原始通道列表
original_channels = ['PO3', 'PO7', 'P3', 'P7', 'CP1', 'CP5', 'Cz', 'C3', 'T7', 'FC5', 'FC1', 'F7', 'F3', 'Fz', 'AF3', 'FP1', 'FP2', 'AF4', 'F4', 'F8', 'FC2', 'FC6', 'C4', 'T8', 'CP6', 'CP2', 'P8', 'P4', 'Pz', 'PO8', 'PO4', 'OZ', 'Status']

# 所需通道列表
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

channels2 = ['Fp1', 'Fp2', 'F11', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F12', 'FT11', 'FC3', 'FCz', 'FC4', 'FT12', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'Status']
# 找出交集
intersection = list(set(original_channels) & set(channels) & set(channels2))

# 打印交集通道
print(f"交集通道: {intersection}")
print(f"交集通道数量: {len(intersection)}")

```

检查通道
```
import os
import mne

def check_electrode_order(folder_path, expected_orders):
    # 获取文件夹中的所有EDF文件
    edf_files = [file for file in os.listdir(folder_path) if file.endswith('.edf')]
    
    # 初始化匹配计数
    match_counts = {tuple(order): 0 for order in expected_orders}
    
    # 循环检查每个EDF文件的电极顺序
    for edf_file in edf_files:
        edf_file_path = os.path.join(folder_path, edf_file)

        # 读取EDF文件
        raw = mne.io.read_raw_edf(edf_file_path, preload=True)

        # 获取当前EDF文件的电极顺序
        current_order = raw.ch_names

        # 检查电极顺序是否符合任何一个预期顺序
        matched = False
        for order in expected_orders:
            if current_order == order:
                match_counts[tuple(order)] += 1
                matched = True
                break

        if not matched:
            print(f"{edf_file}: Electrode order is incorrect. Current order: {current_order}")
    
    for order, count in match_counts.items():
        print(f"Order {list(order)} matches {count} files.channal is {len(list(order))}")

# 替换为实际的文件夹路径和期望的电极顺序
folder_path = "data/NC"
expected_orders = [
    ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2', 'Status'],
    ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'AF3', 'AF4', 'PO3', 'PO4', 'PO7', 'PO8', 'Oz', 'Status'],
    ['Fp1', 'Fp2', 'F11', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F12', 'FT11', 'FC3', 'FCz', 'FC4', 'FT12', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'Status'],
    ['Fp1', 'Fp2', 'F11', 'F7', 'F3', 'Fz', 'F4', 'F8', 'F12', 'FT11', 'FC3', 'FCz', 'FC4', 'FT12', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4', 'M1', 'M2', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2', 'Trigger', 'Status'],
    ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz', 'Oz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'ECG', 'AF3', 'AF4', 'PO3', 'PO4', 'PO7', 'PO8', 'Status'],
    ['Fp1', 'Fp2', 'Fz', 'F3', 'F4', 'F7', 'F8', 'FCz', 'FC3', 'FC4', 'FT7', 'FT8', 'Cz', 'C3', 'C4', 'T3', 'T4', 'CPz', 'CP3', 'CP4', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'T5', 'T6', 'Oz', 'O1', 'O2', 'HEOL', 'HEOR', 'Status'],
    ['PO3', 'PO7', 'P3', 'P7', 'CP1', 'CP5', 'Cz', 'C3', 'T7', 'FC5', 'FC1', 'F7', 'F3', 'Fz', 'AF3', 'FP1', 'FP2', 'AF4', 'F4', 'F8', 'FC2', 'FC6', 'C4', 'T8', 'CP6', 'CP2', 'P8', 'P4', 'Pz', 'PO8', 'PO4', 'OZ', 'Status'],
    ['Fp1', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO7', 'PO3', 'PO4', 'PO8', 'OZ', 'AFZ', 'Status']
]

# 执行检查
check_electrode_order(folder_path, expected_orders)

```

数据预处理- 留一法验证

1. 随机划分数据集。
```
import os
import shutil
import random

# 指定包含数据文件的文件夹列表
data_folders = [
    "guding_channl/AD",
    "guding_channl/CN",
    "guding_channl/MCI"
]

# 对应的保存训练集和测试集的文件夹路径
train_folders = [
    "data/train/AD",
    "data/train/CN",
    "data/train/MCI"
]

test_folders = [
    "data/test/AD",
    "data/test/CN",
    "data/test/MCI"
]

# 确保输入和目标文件夹数量匹配
assert len(data_folders) == len(train_folders) == len(test_folders), "输入文件夹和目标文件夹数量不匹配"

# 遍历每个数据文件夹
for data_folder, train_folder, test_folder in zip(data_folders, train_folders, test_folders):
    # 创建保存训练集和测试集的文件夹
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # 获取数据文件夹中的所有文件
    data_files = os.listdir(data_folder)

    # 遍历数据文件夹中的文件
    for file_name in data_files:
        source_file = os.path.join(data_folder, file_name)
        try:
            # 以8:2的比例将文件分配到训练集或测试集
            if random.random() < 0.9:
                shutil.copy(source_file, train_folder)
            else:
                shutil.copy(source_file, test_folder)
        except Exception as e:
            print(f"无法复制文件 {file_name}: {e}")

    # 打印训练集和测试集的文件数量
    print(f"{data_folder} -> 训练集大小:", len(os.listdir(train_folder)))
    print(f"{data_folder} -> 测试集大小:", len(os.listdir(test_folder)))


```
3. 选取固定通道数
```
import os
import mne
import numpy as np

# 定义所需通道的列表
channels = ['C3', 'Fz', 'F8', 'F4', 'C4', 'F3', 'Pz', 'P4', 'Cz', 'P3', 'F7']
# 输入和输出文件夹路径列表
input_folders = ['data/AD', 'data/MCI', 'data/NC', '公开数据集/AD', '公开数据集/CN']
output_folders = ['guding_channl/AD', 'guding_channl/MCI', 'guding_channl/CN', 'guding_channl/AD', 'guding_channl/CN']

# 定义函数来处理EDF文件
def process_edf_file(file_path, output_folder):
    # 读取EDF文件
    raw = mne.io.read_raw_edf(file_path, preload=True)
    
    # 检查通道数量
    if not all(ch in raw.ch_names for ch in channels):
        print(f"文件 {file_path} 不包含所有所需通道，跳过处理。")
        return
    
    # 删除不需要的通道，只保留指定的通道
    raw.pick_channels(channels)
    
    # 按照指定顺序重新排列通道
    raw.reorder_channels(channels)
    
    # 获取处理后的EEG数据
    eeg_data = raw.get_data()
    
    # 获取文件名（不包含扩展名）
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 构建保存路径
    save_path = os.path.join(output_folder, f'{file_name}_processed.npy')
    
    # 保存处理后的EEG数据为npy文件
    np.save(save_path, eeg_data)
    print(f"处理并保存文件: {save_path}")

# 定义函数来处理SET文件
def process_set_file(file_path, output_folder):
    # 读取SET文件
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    
    # 检查通道数量
    if not all(ch in raw.ch_names for ch in channels):
        print(f"文件 {file_path} 不包含所有所需通道，跳过处理。")
        return
    
    # 删除不需要的通道，只保留指定的通道
    raw.pick_channels(channels)
    
    # 按照指定顺序重新排列通道
    raw.reorder_channels(channels)
    
    # 获取处理后的EEG数据
    eeg_data = raw.get_data()
    
    # 获取文件名（不包含扩展名）
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 构建保存路径
    save_path = os.path.join(output_folder, f'{file_name}_processed.npy')
    
    # 保存处理后的EEG数据为npy文件
    np.save(save_path, eeg_data)
    print(f"处理并保存文件: {save_path}")

# 定义函数来处理文件，根据文件后缀选择处理方法
def process_file(file_path, output_folder):
    file_extension = os.path.splitext(file_path)[1]
    if file_extension == '.edf':
        process_edf_file(file_path, output_folder)
    elif file_extension == '.set':
        process_set_file(file_path, output_folder)
    else:
        print(f"文件 {file_path} 格式不支持，跳过处理。")


# 确保文件夹数量相同
assert len(input_folders) == len(output_folders), "输入和输出文件夹数量不匹配"

# 创建新文件夹
for output_folder in output_folders:
    os.makedirs(output_folder, exist_ok=True)

# 遍历每个输入文件夹
for input_folder, output_folder in zip(input_folders, output_folders):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)
        # 根据文件后缀选择处理方法
        process_file(file_path, output_folder)

```

2. 剪切数据集代码
```
import os
import numpy as np
from tqdm import tqdm

# 指定包含原始npy文件的文件夹路径列表
input_folders = ['data/test/AD', 'data/test/CN','data/test/MCI','data/train/AD', 'data/train/CN','data/train/MCI']  # 输入文件夹列表
output_folders = ['data_npy_cut/test/AD', 'data_npy_cut/test/CN','data_npy_cut/test/MCI','data_npy_cut/train/AD', 'data_npy_cut/train/CN','data_npy_cut/train/MCI']  # 对应的输出文件夹列表

# 确定剪切后的数据长度
cut_length = 2048

# 确保输出文件夹存在
for output_folder in output_folders:
    os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹列表
for input_folder, output_folder in zip(input_folders, output_folders):
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npy'):
            # 加载原始npy文件
            data = np.load(os.path.join(input_folder, file_name))
            
            # 确定剪切的段数
            num_cuts = data.shape[1] // cut_length

            # 使用tqdm显示进度条
            for i in tqdm(range(num_cuts), desc=f'Processing {file_name} in {input_folder}', unit='cut'):
                cut_data = data[:, i * cut_length : (i + 1) * cut_length]
                np.save(os.path.join(output_folder, f'{file_name[:-4]}_cut_{i}.npy'), cut_data)

```
5. 固定训练数据集
```
import os
import csv
import numpy as np

train_path = "train_data.csv"
val_path = "test_data.csv"

def create_data_text(path):
    """建立数据data列表,划分数据集"""
    f_train = open(train_path, "w", newline='') 
    f_val = open(val_path, "w", newline='')
    
    train_writer = csv.writer(f_train)
    val_writer = csv.writer(f_val)
    
    # 遍历 'train' 文件夹
    train_dir = os.path.join(path, 'train')
    for cls, dirname in enumerate(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, dirname)
        if os.path.isdir(class_path):
            flist = os.listdir(class_path)
            np.random.shuffle(flist)
            fnum = len(flist)
            for i, filename in enumerate(flist):
                train_writer.writerow([os.path.join(class_path, filename), str(cls)])
    
    # 遍历 'test' 文件夹
    test_dir = os.path.join(path, 'test')
    for cls, dirname in enumerate(os.listdir(test_dir)):
        class_path = os.path.join(test_dir, dirname)
        if os.path.isdir(class_path):
            flist = os.listdir(class_path)
            np.random.shuffle(flist)
            fnum = len(flist)
            for i, filename in enumerate(flist):
                val_writer.writerow([os.path.join(class_path, filename), str(cls)])

    f_train.close()
    f_val.close()

if __name__ == "__main__":
    create_data_text("data_npy_cut")


```



6. 十折交叉验证

```
import os
import math
import shutil

def split_data_into_folds(data_folders, output_folders):
    for data_folder, output_folder in zip(data_folders, output_folders):
        os.makedirs(output_folder, exist_ok=True)

        # 获取数据集文件夹中的所有文件名
        all_files = os.listdir(data_folder)

        # 每个折中的文件数
        files_per_fold = len(all_files) // 10

        # 按照每个折的文件数划分文件列表
        fold_files = [all_files[i:i+files_per_fold] for i in range(0, len(all_files), files_per_fold)]

        # 保存到对应的折文件夹中
        for i, files_in_fold in enumerate(fold_files):
            fold_path = os.path.join(output_folder, f'fold_{i+1}')
            os.makedirs(fold_path, exist_ok=True)
            for file_name in files_in_fold:
                shutil.copy(os.path.join(data_folder, file_name), fold_path)

        # 检测是否存在fold_11文件夹
        fold11_path = os.path.join(output_folder, 'fold_11')
        if os.path.exists(fold11_path) and os.path.isdir(fold11_path):
            # 将fold_11中的内容移动到fold_10中
            fold10_path = os.path.join(output_folder, 'fold_10')
            for file_name in os.listdir(fold11_path):
                shutil.move(os.path.join(fold11_path, file_name), fold10_path)
            
            # 删除fold_11文件夹
            os.rmdir(fold11_path)

# 数据集文件夹路径列表
data_folders = ['guding_channl/AD', 'guding_channl/MCI', 'guding_channl/CN']

# 输出文件夹路径列表
output_folders = ['output_folder/AD', 'output_folder/MCI', 'output_folder/CN']

split_data_into_folds(data_folders, output_folders)

def create_train_test_sets(input_folder, output_folder, test_folders):
    for test_folder in test_folders:
        os.makedirs(os.path.join(output_folder, test_folder, 'test', os.path.basename(input_folder)), exist_ok=True)
        os.makedirs(os.path.join(output_folder, test_folder, 'train', os.path.basename(input_folder)), exist_ok=True)

        train_folders = [folder for folder in os.listdir(input_folder) if folder != test_folder]

        for file_name in os.listdir(os.path.join(input_folder, test_folder)):
            shutil.copy(os.path.join(input_folder, test_folder, file_name), os.path.join(output_folder, test_folder, 'test', os.path.basename(input_folder)))

        for train_folder in train_folders:
            for file_name in os.listdir(os.path.join(input_folder, train_folder)):
                shutil.copy(os.path.join(input_folder, train_folder, file_name), os.path.join(output_folder, test_folder, 'train', os.path.basename(input_folder)))

# 输入文件夹路径列表，包含三个文件夹 AD、MCI、CN
input_folders = ['output_folder/AD', 'output_folder/MCI', 'output_folder/CN']

# 输出文件夹路径
output_folder = 'output_folder_fixed'

# 测试文件夹名称列表，代表每个input_folders中文件的参照测试文件夹
test_folders = ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'fold_6', 'fold_7', 'fold_8', 'fold_9', 'fold_10']

# 对每个输入文件夹调用函数
for input_folder in input_folders:
    create_train_test_sets(input_folder, output_folder, test_folders)

```
## 数据增强
mne库的文档链接[链接](https://mne.tools/dev/api/python_reference.html)
pywt库的
3. 连续小波变换 + RBP + 绘制图像的代码
```
import numpy as np
from scipy.signal import welch
import os

def calculate_psd(signal, fs, nperseg):
    """
    使用Welch方法计算功率谱密度（PSD）。

    参数:
    - signal: 输入信号（二维数组：通道数 x 样本数）。
    - fs: 采样频率（Hz）。
    - nperseg: 每段的长度（样本数）。

    返回:
    - f: 频率数组。
    - psd: 功率谱密度（PSD），形状为（通道数 x 频率数）。
    """
    psd_list = []
    for ch_signal in signal:
        f, psd_ch = welch(ch_signal, fs, nperseg=nperseg)
        psd_list.append(psd_ch)
    return f, np.array(psd_list)

def calculate_rbp(f, psd, freq_bands):
    """
    计算相对波动指数（RBP）。

    参数:
    - f: 频率数组。
    - psd: 功率谱密度（PSD），形状为（通道数 x 频率数）。
    - freq_bands: 频段列表，每个频段是一个元组（开始频率，结束频率）。

    返回:
    - rbp: 相对波动指数（RBP），形状为（通道数 x 频段数）。
    """
    total_psd = np.sum(psd, axis=1, keepdims=True)
    rbp = []
    for (f_start, f_end) in freq_bands:
        band_power = np.sum(psd[:, (f >= f_start) & (f <= f_end)], axis=1, keepdims=True)
        rbp.append(band_power / total_psd)
    return np.hstack(rbp)

# 示例频段（可以根据实际需求调整）
freq_bands = [(0.5, 4), (4, 8), (8, 12), (12, 30), (30, 50)]

# 处理一个文件夹中的所有 .npy 文件
input_folder = 'data_npy_cut/test/AD/'  # 输入文件夹路径
output_folder = 'PSD-RBP/test/AD'  # 输出文件夹路径

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理文件夹中的所有 .npy 文件
for filename in os.listdir(input_folder):
    if filename.endswith('.npy'):
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename.split('.')[0] + '_rbp.npy')
        
        # 加载输入数据
        signal = np.load(input_file_path)

        # 检查输入信号的大小是否为 [19, 2500]
        assert signal.shape == (19, 2500), f"Expected input signal shape to be (19, 2500), but got {signal.shape}"

        # 设置采样频率和每段的长度
        fs = 500  # 采样频率（Hz）
        nperseg = 128  # 每段的长度（样本数）

        # 计算PSD
        f, psd = calculate_psd(signal, fs, nperseg)

        # 计算RBP
        rbp = calculate_rbp(f, psd, freq_bands)

        # 将RBP保存到新的 .npy 文件
        np.save(output_file_path, rbp)
        print(f'Saved {output_file_path}')

```
```
import numpy as np
import matplotlib.pyplot as plt

# 加载 .npy 文件
npy_file_path = 'output.npy'  # 替换为实际的 .npy 文件路径
rbp_data = np.load(npy_file_path)
print(rbp_data.shape)
# 检查数据形状
assert rbp_data.shape == (19, 5), f"Expected RBP data shape to be (19, 5), but got {rbp_data.shape}"

# 绘制 RBP 数据
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']  # 替换为实际的通道名称
freq_bands = ['0.5-4 Hz', '4-8 Hz', '8-12 Hz', '12-30 Hz', '30-50 Hz']  # 替换为实际的频段名称

fig, ax = plt.subplots(figsize=(12, 8))
cax = ax.matshow(rbp_data, cmap='viridis')

# 设置颜色条
fig.colorbar(cax)

# 设置通道名称
ax.set_xticks(np.arange(len(freq_bands)))
ax.set_xticklabels(freq_bands, rotation=45, ha='left')
ax.set_yticks(np.arange(len(channels)))
ax.set_yticklabels(channels)

# 设置标签
plt.xlabel('Frequency Bands')
plt.ylabel('Channels')
plt.title('Relative Band Power (RBP) Heatmap')

# 显示图像
plt.tight_layout()  # 调整布局以防止标签重叠
plt.show()

```
4. CWT + RBP
```
import os
import numpy as np
import pywt
from tqdm import tqdm

# 采样频率
fs = 500

# 输入和输出文件夹路径列表
input_folders = ['data_npy_cut/train/AD/', 'data_npy_cut/train/CN/','data_npy_cut/train/FDT/']
output_folders = ['CWT-RBP/train/AD', 'CWT-RBP/train/CN','CWT-RBP/train/FDT']


# 确保文件夹数量相同
assert len(input_folders) == len(output_folders), "输入和输出文件夹数量不匹配"

# 创建新文件夹
for output_folder in output_folders:
    os.makedirs(output_folder, exist_ok=True)

# 小波函数
wavelet = 'morl'

# 定义 CWT 的频率范围
freq_ranges = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]
scales = np.arange(1, 128)

def calculate_rbp(cwt_coeffs, freq_band, scales, fs):
    min_scale = np.min(np.where((fs / (scales * 2)) <= freq_band[1]))
    max_scale = np.max(np.where((fs / (scales * 2)) >= freq_band[0]))
    band_power = np.sum(np.abs(cwt_coeffs[min_scale:max_scale + 1, :])**2, axis=0)
    total_power = np.sum(np.abs(cwt_coeffs)**2, axis=0)
    return band_power / total_power

# 遍历输入文件夹
for input_folder, output_folder in zip(input_folders, output_folders):
    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith('.npy'):
            # 读取 .npy 文件
            data = np.load(os.path.join(input_folder, file_name))
            
            # 提取不同频段的数据
            data_freq_bands = []
            for ch_data in data:
                ch_data_freq_band = []
                # 计算 CWT 系数
                cwt_coeffs, _ = pywt.cwt(ch_data, scales, wavelet, sampling_period=1/fs)
                for fmin, fmax in freq_ranges:
                    # 计算频段的相对功率谱密度 (RBP)
                    rbp = calculate_rbp(cwt_coeffs, (fmin, fmax), scales, fs)
                    ch_data_freq_band.append(rbp)
                data_freq_bands.append(ch_data_freq_band)
            
            # 重塑数据形状为 [5, 19, 2500]
            data_freq_bands = np.array(data_freq_bands)
            data_freq_bands = np.transpose(data_freq_bands, (0, 1, 2))  # [5, 19, 2500]
            
            # 修改文件名，添加处理方法标记
            output_file_name = os.path.splitext(file_name)[0] + '_cwt_rbp.npy'
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # 保存处理后的数据
            np.save(output_file_path, data_freq_bands)

```


5. FTBT + BRP
FTBT
```
import os
import numpy as np
from scipy.fftpack import fft, ifft
from tqdm import tqdm

# 采样频率
fs = 500

# 输入和输出文件夹路径列表
input_folders = ['data_npy_cut/train/AD/', 'data_npy_cut/train/CN/','data_npy_cut/train/FDT/']
output_folders = ['FBFT/train/AD', 'FBFT/train/CN','FBFT/train/FDT']

# 确保文件夹数量相同
assert len(input_folders) == len(output_folders), "输入和输出文件夹数量不匹配"

# 创建新文件夹
for output_folder in output_folders:
    os.makedirs(output_folder, exist_ok=True)

def fbft(signal):
    N = len(signal)
    forward_fft = fft(signal)
    backward_fft = fft(ifft(forward_fft))
    fbft_result = forward_fft + backward_fft
    return fbft_result

# 遍历输入文件夹
for input_folder, output_folder in zip(input_folders, output_folders):
    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith('.npy'):
            # 读取 .npy 文件
            data = np.load(os.path.join(input_folder, file_name))
            
            # 对每个通道的数据进行 FBFT 操作
            fbft_data = []
            for ch_data in data:
                fbft_ch_data = fbft(ch_data)
                fbft_data.append(fbft_ch_data)
            
            # 将结果转换为 numpy 数组
            fbft_data = np.array(fbft_data)
            
            # 修改文件名，添加处理方法标记
            output_file_name = os.path.splitext(file_name)[0] + '_fbft.npy'
            output_file_path = os.path.join(output_folder, output_file_name)
            
            # 保存处理后的数据
            np.save(output_file_path, fbft_data)


```

使用matlab代码对数据进行CWT变换

```
% 设置数据文件夹和图像保存的根目录
dataDir = 'E:\数据处理\data_npy_cut\test\FDT'; % 数据文件夹路径
outputDir = 'E:\数据处理\处理后\test\2'; % 输出文件夹路径

% 添加 npy-matlab 工具包路径
addpath('E:\数据处理\npy-matlab-master\npy-matlab'); % 替换为 npy-matlab 工具包的实际路径

% 获取文件夹中所有 npy 文件
files = dir(fullfile(dataDir, '*.npy'));
% 设置采样率
fs = 500;

% 定义要处理的通道
channelsToProcess = [1, 4, 16,19];

% 处理每个文件
for k = 1:length(files)
    % 读取数据
    filename = files(k).name;
    filepath = fullfile(dataDir, filename);
    data = readNPY(filepath);
    
    % 创建输出文件夹
    [~, name, ~] = fileparts(filename);
    saveFolder = fullfile(outputDir, name);
    if ~exist(saveFolder, 'dir')
        mkdir(saveFolder);
    end
    
    % 对选定通道执行 CWT 并保存图像
    for i = channelsToProcess % 只处理指定通道
        if i <= size(data, 1) % 确保通道索引有效
            [cfs, frequencies] = cwt(data(i, :), fs);
            figure;
            imagesc(1:size(data, 2), frequencies, abs(cfs)); % 绘制小波系数的幅度
            axis xy;
            xlabel('Time (samples)');
            ylabel('Frequency (Hz)');
            title(['CWT Magnitude of Channel ', num2str(i)]);
            
            % 保存图像
            saveFileName = fullfile(saveFolder, sprintf('Channel_%d.png', i));
            saveas(gcf, saveFileName);
            close(gcf); % 关闭图像窗口以节省资源
        end
    end
end
```
图片转换为npy数据格式，二值化输入

```
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 输入的文件夹列表和保存位置列表
folder_list = ['data/train/AD', 'data/train/CN', 'data/train/FDT','data/test/AD','data/test/CN','data/test/FDT']
save_location_list = ['data-npy/train/AD', 'data-npy/train/CN', 'data-npy/train/FDT','data-npy/test/AD','data-npy/test/CN','data-npy/test/FDT']

# 遍历文件夹列表
for folder, save_location in zip(folder_list, save_location_list):
    # 遍历第二层文件夹
    for sub_folder in os.listdir(folder):
        sub_folder_path = os.path.join(folder, sub_folder)
        # 读取第二层文件夹中所有图片
        image_files = [f for f in os.listdir(sub_folder_path) if os.path.isfile(os.path.join(sub_folder_path, f))]
        images_data = []
        # 使用tqdm添加进度条
        for image_file in tqdm(image_files, desc=f"Processing {sub_folder}"):
            image_path = os.path.join(sub_folder_path, image_file)
            with Image.open(image_path).convert('L') as img:
                img_data = np.array(img)
                images_data.append(img_data)

        # 创建保存.npy文件的文件夹
        os.makedirs(save_location, exist_ok=True)
        # 保存为.npy文件，文件名为对应的保存位置
        output_path = os.path.join(save_location, f'{sub_folder}.npy')
        np.save(output_path, images_data)

```
## 任务寻找预处理代码
在paperwithcode上。


1. DWT 算法对将时间域上的数据转换为频率能量上的数据，找到这样的代码。
连续的小波变换 ：CWT
离散的小波变换 ：DWT
小波变换的基本知识：
不同的小波基函数，是由同一个基本小波函数经缩放和平移生成的。
小波变换是将原始图像与小波基函数以及尺度函数进行内积运算, 所以一个尺度函数和一个小波基函数就可以确定一个小波变换。



2. PSD
[参考链接](https://blog.csdn.net/frostime/article/details/106967703)
[mne怎么使用](https://iq.opengenus.org/eeg-signal-analysis-with-python/)
[mne函数psd——array-welch介绍](https://mne.tools/stable/generated/mne.time_frequency.psd_array_welch.html)
[mne函数psd-array-welch使用例子](https://mne.tools/stable/auto_examples/decoding/ssd_spatial_filters.html#sphx-glr-auto-examples-decoding-ssd-spatial-filters-py)
提取通道位置， TP9，AF7，AF8，TP10
['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']









```
## https://blog.csdn.net/SashiMoore/article/details/128599822 -->
import numpy as np
from matplotlib import pyplot as plt
 
from mne import create_info, Epochs
from mne.baseline import rescale
from mne.io import RawArray
from mne.time_frequency import (tfr_multitaper, tfr_stockwell, tfr_morlet,
                                tfr_array_morlet, AverageTFR)
from mne.viz import centers_to_edges
 
'''
Time-frequency on simulated data
(Multitaper vs. Morlet vs. Stockwell vs. Hilbert)
'''
 
'''
使用已知的频谱时间结构来模拟数据
'''
# 设定采样频率
sfreq = 1000.0
# 设定频道名称
ch_names = ['SIM0001', 'SIM0002']
# 设定频道类型
ch_types = ['grad', 'grad']
# 根据以上信息创建info
info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
 
n_times = 1024  # 时间采样点数量，构造epoch的时间长度多于1秒1000
n_epochs = 40   # 构建epoch数量
seed = 42       # 设定种子
rng = np.random.RandomState(seed)   # 根据种子生成随机数组，元素值在0-1之间
# 产生2行n_times*n_epochs+200列的标准正态分布随机数，长度为第二个参数所示
data = rng.randn(len(ch_names), n_times * n_epochs + 200)
 
# 添加一个50赫兹的正弦脉冲噪声和斜坡
# 返回0-1023的浮点数，除采样频率来表示时间采样点在以秒为单位的时间轴中的位置
t = np.arange(n_times, dtype=np.float64) / sfreq
# sin为数组中的每一个元素取正弦，t的系数为100pi，表示波的频率为100pi/2pi等于50hz
signal = np.sin(np.pi * 2. * 50. * t)
# 将信号中指定位置的t赋值为0，表示这些区域没有噪声,即只保留0.45s-0.55s的噪声信号
signal[np.logical_or(t < 0.45, t > 0.55)] = 0.  # Hard windowing
print(signal.shape)
on_time = np.logical_and(t >= 0.45, t <= 0.55)  # 设定取t范围
print(on_time)
# hanning生成长度为True数量的余弦窗口（即位于0.45s-0.55s的时间采样点个数），并乘在原数据区域上
signal[on_time] *= np.hanning(on_time.sum())  # Ramping
# 在data每一个频道的第一百个采样点到倒数第一百个采样点上加入噪声（持续1*20s）
print(data.shape)
data[:, 100:-100] += np.tile(signal, n_epochs)  # add signal
 
raw = RawArray(data, info)  # 建立raw结构
events = np.zeros((n_epochs, 3), dtype=int)     # 建立一个shape为(20, 3)的0数组
events[:, 0] = np.arange(n_epochs) * n_times    # 将第二个维度的第一个数值赋值为epoch所在的时间采样点位置
epochs = Epochs(raw, events, dict(sin50hz=0), tmin=0, tmax=n_times / sfreq,
                reject=dict(grad=4000), baseline=None)  # 建立epochs，将50hz波作为事件0输入
 
epochs.average().plot()
 
'''
计算时频表示
'''
# 多窗口变换
# 生成感兴趣的频率数组
freqs = np.arange(5., 100., 3.)
print('freqs shape is ', freqs.shape)
vmin, vmax = -3., 3.  # Define our color limits.
# 生成组合图，维度为(1, 3)，详解可见之前博客或网上资料
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for n_cycles, time_bandwidth, ax, title in zip(
        [freqs / 2, freqs, freqs / 2],  # 周期数 时间窗口长度 T=n_cycle/freqs，T过大时间精度不够，过小频率精度不够
        [2.0, 4.0, 8.0],  # 时间带宽积，越大计算次数越多，时间分辨率升高，频率分辨率降低，符合不确定性方程
        axs,
        ['Sim: Least smoothing, most variance',
         'Sim: Less frequency smoothing,\nmore time smoothing',
         'Sim: Less time smoothing,\nmore frequency smoothing']):
    power = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles,
                           time_bandwidth=time_bandwidth, return_itc=False)
    ax.set_title(title)
    # Plot results. Baseline correct based on first 100 ms.
    power.plot([0], baseline=(0., 0.1), mode='mean', vmin=vmin, vmax=vmax,
               axes=ax, show=False, colorbar=False)
plt.tight_layout()
plt.show()
 
'''
Stockwell (S) transform
'''
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fmin, fmax = freqs[[0, -1]]
for width, ax in zip((0.2, 0.7, 3.0), axs):
    power = tfr_stockwell(epochs, fmin=fmin, fmax=fmax, width=width)
    power.plot([0], baseline=(0., 0.1), mode='mean', axes=ax, show=False,
               colorbar=False)
    ax.set_title('Sim: Using S transform, width = {:0.1f}'.format(width))
plt.tight_layout()
plt.show()
 
# 小波变换
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
all_n_cycles = [1, 3, freqs / 2.]
for n_cycles, ax in zip(all_n_cycles, axs):
    power = tfr_morlet(epochs, freqs=freqs,
                       n_cycles=n_cycles, return_itc=False)
    power.plot([0], baseline=(0., 0.1), mode='mean', vmin=vmin, vmax=vmax,
               axes=ax, show=False, colorbar=False)
    # 若n_cycle不为int，则赋值为字符串
    n_cycles = 'scaled by freqs' if not isinstance(n_cycles, int) else n_cycles
    ax.set_title(f'Sim: Using Morlet wavelet, n_cycles = {n_cycles}')
plt.tight_layout()
plt.show()
 
# Narrow-bandpass Filter and Hilbert Transform
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
bandwidths = [1., 2., 4.]   # 带通宽度
for bandwidth, ax in zip(bandwidths, axs):
    data = np.zeros((len(ch_names), freqs.size, epochs.times.size),
                    dtype=complex)
    for idx, freq in enumerate(freqs):
        # 过滤原始数据并重新epoch以避免过滤器的时间过长
        # 重新构造低频率和短周期的epoch数据
        raw_filter = raw.copy()
        # 注意:过滤器的带宽从默认值改变
        # 夸大差异。使用默认的转换带宽，
        # 这些都非常相似，因为过滤器几乎是一样的。
        # 在实践中，使用默认值通常是明智的选择。
        # 滤波器设置
        raw_filter.filter(
            l_freq=freq - bandwidth / 2, h_freq=freq + bandwidth / 2,
            # 对于大带宽和低频率计算没有负值
            # 过渡带宽度设定
            l_trans_bandwidth=min([4 * bandwidth, freq - bandwidth]),
            h_trans_bandwidth=4 * bandwidth)
        # 该函数计算通道子集的分析信号或包络
        raw_filter.apply_hilbert()
        epochs_hilb = Epochs(raw_filter, events, tmin=0, tmax=n_times / sfreq,
                             baseline=(0, 0.1))
        tfr_data = epochs_hilb.get_data()
        tfr_data = tfr_data * tfr_data.conj()  # compute power conj获取共轭复数
        tfr_data = np.mean(tfr_data, axis=0)  # average over epochs
        data[:, idx] = tfr_data
    power = AverageTFR(info, data, epochs.times, freqs, nave=n_epochs)
    power.plot([0], baseline=(0., 0.1), mode='mean', vmin=-0.1, vmax=0.1,
               axes=ax, show=False, colorbar=False)
    n_cycles = 'scaled by freqs' if not isinstance(n_cycles, int) else n_cycles
    ax.set_title('Sim: Using narrow bandpass filter Hilbert,\n'
                 f'bandwidth = {bandwidth}, '
                 f'transition bandwidth = {4 * bandwidth}')
plt.tight_layout()
plt.show()
 
# Calculating a TFR without averaging over epochs
n_cycles = freqs / 2.
power = tfr_morlet(epochs, freqs=freqs,
                   n_cycles=n_cycles, return_itc=False, average=False)
print(type(power))
avgpower = power.average()
avgpower.plot([0], baseline=(0., 0.1), mode='mean', vmin=vmin, vmax=vmax,
              title='Using Morlet wavelets and EpochsTFR', show=False)
plt.show()

```














划分数据集

```
import os
import csv
import numpy as np

train_path = "train_data.csv"
val_path = "test_data.csv"



def create_data_text(path,train_percent = 0.9):
    """建立数据data列表,划分数据集"""
    f_train = open(train_path,"w",newline='') #
    #并将文件对象赋给了变量 f_train。open(train_path, "w", newline='')
    #  的意思是以写入模式打开名为 train_path 的文件，
    # 并且在写入文本时不插入额外的换行符（newline=''）。open是python的内置函数
    f_val = open(val_path,"w",newline='')
    train_writer = csv.writer(f_train)
    # 创建了一个 CSV writer 对象 train_writer，
    # 用于向文件对象 f_train 中写入 CSV 格式的数据。csv.writer() 接受一个文件对象作为参数，
    # 并返回一个 CSV writer 对象，可以使用该对象的方法将数据写入文件。
    val_writer = csv.writer(f_val)
    # enumerate() 是 Python 内置函数，
    # 用于将一个可迭代对象（如列表、元组、字符串等）
    # 组合为一个索引序列，同时列出数据和数据下标。
    # 函数返回一个枚举对象，其中每个元素是一个包含索引和对应元素的元组。
    for cls,dirname in enumerate(os.listdir(path)):
        flist = os.listdir(os.path.join(path,dirname))
        # os.path.join(path, dirname) 是一个函数，用于将多个路径组合成一个完整的路径。
        # 它会根据当前操作系统的规则使用正确的路径分隔符来连接路径。
        np.random.shuffle(flist)
        # np.random.shuffle(flist) 是 NumPy 库中的一个函数，用于随机打乱列表 flist 中的元素顺序。
        # 这个函数会改变原始列表的顺序，使得列表中的元素随机排列。
        fnum = len(flist)
        # len函数返回
        for i,filename in enumerate(flist):
            # 使用了 enumerate() 函数来遍历列表 flist 中的元素，同时获取元素的索引和值。具体来说，
            # enumerate(flist) 返回一个枚举对象，其中每个元素是一个元组，包含元素在列表中的索引和元素的值。
            if i < fnum * train_percent:
                train_writer.writerow([os.path.join(path,dirname,filename),str(cls)])
            # 是将一个包含两个元素的列表写入CSV文件的操作。这个列表包含两个元素：
            # os.path.join(path,dirname,filename) 返回一个完整的文件路径，其中 path 是主目录路径，dirname 是子目录路径，filename 是文件名。这个路径表示要写入CSV文件的文件的完整路径。
            # str(cls) 将整数 cls 转换为字符串，cls 表示类别编号。
            else:
                val_writer.writerow([os.path.join(path,dirname,filename),str(cls)])
    f_train.close()
    # f_train.close() 是关闭文件 f_train 的方法。在使用完文件后，
    # 应该调用这个方法来关闭文件，以释放资源并确保文件被正确关闭。
    f_val.close()

if __name__ == "__main__":
    create_data_text("FFT_data")
    
```

## 填坑

train-loss 和test-loss之间的关系
变化趋势分析：train loss 不断下降，test loss不断下降，说明网络仍在学习;（最好的）train loss 不断下降，test loss趋于不变，说明网络过拟合；train loss 趋于不变，test loss不断下降，说明数据集100%有问题;（检查dataset）train loss 趋于不变，test loss趋于不变，说明学习遇到瓶颈，需要减小学习率或批量数目;（减少学习率）train loss 不断上升，test loss不断上升，说明网络结构设计不当，训练超参数设置不当，数据集经过清洗等问题；（最不好的情况）train_loss 不断下降， test_loss 不断上升，和第2种情况类似说明网络过拟合了。


1. 调高batch——size有提高点的趋势 32好于 0.0005的参数。

2. 动态调整学习率
使用 StepLR、ExponentialLR 或 ReduceLROnPlateau

StepLR
```
import torch.optim as optim

# 定义优化器
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# 定义学习率调度器，每隔 step_size 个 epoch，将学习率乘以 gamma
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 训练循环
num_epochs = 50
for epoch in range(num_epochs):
    # 训练步骤
    train(...)
    
    # 更新学习率
    scheduler.step()
    
    # 打印当前学习率
    print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {scheduler.get_last_lr()}")


```

ExponentialLR 以指数方式衰减学习率。
```
# 定义优化器
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# 定义学习率调度器
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
```

ReduceLROnPlateau 在监测指标停滞时降低学习率，适用于验证损失等指标。
```
# 定义优化器
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

# 定义学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


```


3. 替换模型的输出层

```

```
### 密码
Qwer123@
Qwer123.
