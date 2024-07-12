---
title: 高等数学 —— 级数
date: 2024-06-09 11:11:07
tags: 高等数学
---

## 傅里叶变换解决的问题


## 傅里叶变换
1. 傅里叶级数： 周期性的函数f(t) 都考研变换维一系列的正（余）弦函数的组合。 

时域 ——傅里叶变换—— 频率，相位，增幅。 
连续的

![](pic/gdsx-flybh.png)
2. 傅里叶变换 （非连续）
欧拉公式


3. 应用： 声音的处理 ，图像的处理 


| |
| :------ | 
|![](pic/gdsx-flybh2.png)|
|![](pic/gdsx-hbs_56.png)|
|![](pic/gdsx-hbs_57.png)|
|![](pic/gdsx-hbs_58.png)|
|![](pic/gdsx-hbs_59.png)|
|![](pic/gdsx-hbs_60.png)|
|![](pic/gdsx-hbs_61.png)|
|![](pic/gdsx-hbs_62.png)|
|![](pic/gdsx-hbs_63.png)|
|![](pic/gdsx-hbs_64.png)|
|![](pic/gdsx-hbs_65.png)|
|![](pic/gdsx-hbs_66.png)|
|![](pic/gdsx-hbs_67.png)|
|![](pic/gdsx-hbs_68.png)|
|![](pic/gdsx-hbs_69.png)|
|![](pic/gdsx-hbs_70.png)|
|![](pic/gdsx-hbs_71.png)|
|![](pic/gdsx-hbs_72.png)|

## 波段

提取脑电波（EEG）中的不同波段通常使用数字滤波器来分离特定频率范围的信号。以下是一些常见的脑电波段及其频率范围：

- Delta波：0.5–4 Hz
- Theta波：4–8 Hz
- Alpha波：8–13 Hz
- Beta波：13–25 Hz
- Gamma波：25–45 Hz


```
import os
import mne
import numpy as np

# 定义所需通道的列表
channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
# 输入和输出文件夹路径列表
input_folders = ['公开数据集/AD', '公开数据集/CN']
output_folders = [ 'guding_channl/AD', 'guding_channl/CN']


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
    
    # 提取Delta波段
    Delta_data = raw.copy().filter(0.5, 4, fir_design='firwin').get_data()
    
    # 提取Theta波段 (4-8 Hz)
    theta_data = raw.copy().filter(4, 8, fir_design='firwin').get_data()

    # 提取Alpha波段 (8-13 Hz)
    alpha_data = raw.copy().filter(8, 13, fir_design='firwin').get_data()
    
    # 提取Beta
    Beta_data = raw.copy().filter(13, 25, fir_design='firwin').get_data()

    # 提取Gammma
    Gamma_data = raw.copy().filter(25, 45, fir_design='firwin').get_data()
    
    # 堆叠Alpha和Theta波段的数据
    stacked_data = np.stack((Delta_data, theta_data,alpha_data,Beta_data,Gamma_data),axis=0)
    print(stacked_data.shape)
    # 获取文件名（不包含扩展名）
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 构建保存路径
    save_path = os.path.join(output_folder, f'{file_name}_processed.npy')
    
    # 保存处理后的EEG数据为npy文件
    np.save(save_path, stacked_data)
    print(f"处理并保存文件: {save_path}")


def process_set_file(file_path, output_folder):
    # 读取SET文件
    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    
    # 上采样到512采样率
    raw.resample(512)
    
    # 检查通道数量
    if not all(ch in raw.ch_names for ch in channels):
        print(f"文件 {file_path} 不包含所有所需通道，跳过处理。")
        return
    
    # 删除不需要的通道，只保留指定的通道
    raw.pick_channels(channels)
    
    # 按照指定顺序重新排列通道
    raw.reorder_channels(channels)
    
  # 提取Delta波段
    Delta_data = raw.copy().filter(0.5, 4, fir_design='firwin').get_data()
    
    # 提取Theta波段 (4-8 Hz)
    theta_data = raw.copy().filter(4, 8, fir_design='firwin').get_data()

    # 提取Alpha波段 (8-13 Hz)
    alpha_data = raw.copy().filter(8, 13, fir_design='firwin').get_data()
    
    # 提取Beta
    Beta_data = raw.copy().filter(13, 25, fir_design='firwin').get_data()

    # 提取Gammma
    Gamma_data = raw.copy().filter(25, 45, fir_design='firwin').get_data()
    
    # 堆叠Alpha和Theta波段的数据
    stacked_data = np.stack((Delta_data, theta_data,alpha_data,Beta_data,Gamma_data),axis=0)
    print(stacked_data.shape)
    # 获取文件名（不包含扩展名）
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # 构建保存路径
    save_path = os.path.join(output_folder, f'{file_name}_processed.npy')
    
    # 保存处理后的EEG数据为npy文件
    np.save(save_path, stacked_data)
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

PSD处理方法已经被验证，上述代码没有问题。