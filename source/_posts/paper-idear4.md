---
title: 论文思路——代码
date: 2024-05-09 22:04:42
tags: 论文思路
---

## 数据划分
1. 数据集分类

将set的文件转换为npy格式
```
import os
import mne
import numpy as np

def find_set_files(root_folder):
    set_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.set'):
                set_files.append(os.path.join(root, file))
    return set_files

# 指定您的 AD 文件夹路径
ad_folder = r'FDT'

# 指定保存 .npy 文件的新文件夹路径
output_folder = r'data_onremove_npy/FDT'
os.makedirs(output_folder, exist_ok=True)

# 找到所有的 .set 文件
set_files = find_set_files(ad_folder)

# 读取 .set 文件并保存为 .npy 文件
for file_path in set_files:
    # 读取 .set 文件
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    # 获取原始数据
    data = raw.get_data()

    # 构造新的文件路径
    npy_file_name = os.path.basename(file_path).replace('.set', '.npy')
    npy_file_path = os.path.join(output_folder, npy_file_name)

    # 保存为 .npy 文件
    np.save(npy_file_path, data)
    print(f'Saved {npy_file_path}')

```


将数据剪切为想要的长度
```
import os
import numpy as np
from tqdm import tqdm

# 指定包含原始npy文件的文件夹路径列表
input_folders = ['guding_channl/test/AD', 'guding_channl/test/CN','guding_channl/train/AD', 'guding_channl/train/CN',]  # 输入文件夹列表
output_folders = ['data_npy_cut/test/AD', 'data_npy_cut/test/CN','data_npy_cut/train/AD', 'data_npy_cut/train/CN',]  # 对应的输出文件夹列表

# 确定剪切后的数据长度
cut_length = 2500

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
对数据进行fft变换

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

显示fft数据

```
import numpy as np
import os

# 替换为包含.npy文件的文件夹路径列表
folder_paths = ["data_FFT_npy/AD", "data_FFT_npy/CN","data_FFT_npy/FDT"]  # 替换为实际的文件夹路径列表
# ["normal_save", "open_normal_save", "open_patient_save",'patient_save']
# 循环遍历每个文件夹
for folder_path in folder_paths:
    # 获取文件夹中的所有.npy文件
    npy_files = [file for file in os.listdir(folder_path) if file.endswith('.npy')]

    # 循环加载每个.npy文件并显示其形状
    for npy_file in npy_files:
        npy_file_path = os.path.join(folder_path, npy_file)
        data = np.load(npy_file_path)
        print(f"Folder: {folder_path}, File: {npy_file}, Shape: {data.shape}")

```

## 配置文件

训练代码

```
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import EEGDataset,EEGDataset_Batch_normal
from net import IntegratedNet
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt


# 定义归一化操作
def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

transform = transforms.Compose([
        transforms.Lambda(normalize),  # 使用Lambda函数应用自定义归一化操作
        transforms.ToTensor()
    ])

def train_identityformer_model(model, model_name, num_epochs=100, num_classes=3, batch_size=16, learning_rate=0.0001, w_wight=1025, chennal=33):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    m = nn.Softmax(dim=1)

    train_dataset = EEGDataset(csv_file='train_data.csv', transform=transform)
    test_dataset = EEGDataset(csv_file='test_data.csv', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    save_dir = 'loss'
    os.makedirs(save_dir, exist_ok=True)

    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        train_loss_total = 0
        train_acc_total = 0
        val_loss_total = 0
        val_acc_total = 0

        model.train()
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (train_x, train_y) in progress_bar:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            train_x = train_x.unsqueeze(1)
            train_x = train_x.view(batch_size, 1, chennal, w_wight)

            train_y_pred = model(train_x)
            train_loss = loss_fn(train_y_pred, train_y)

            train_acc = (m(train_y_pred).max(dim=1)[1] == train_y).sum() / train_y.shape[0]
            train_loss_total += train_loss.data.item()
            train_acc_total += train_acc.data.item()

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Train Loss: {train_loss.data.item():.4f}, Train Acc: {train_acc.data.item():.4f}")

        train_loss_arr.append(train_loss_total / len(train_loader))
        train_acc_arr.append(train_acc_total / len(train_loader))

        model.eval()
        for j, (val_x, val_y) in enumerate(test_loader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_x = val_x.unsqueeze(1)
            val_x = val_x.view(batch_size, 1, chennal, w_wight)

            val_y_pred = model(val_x)
            val_loss = loss_fn(val_y_pred, val_y)
            val_acc = (m(val_y_pred).max(dim=1)[1] == val_y).sum() / val_y.shape[0]
            val_loss_total += val_loss.data.item()
            val_acc_total += val_acc.data.item()

        val_loss_arr.append(val_loss_total / len(test_loader))
        val_acc_arr.append(val_acc_total / len(test_loader))

        if val_acc_arr[-1] > best_val_acc:
            best_val_acc = val_acc_arr[-1]
            best_epoch = epoch
            torch.save(model.state_dict(), f"{model_name}_best.pth")

        print("epoch:{} val_loss:{} val_acc:{}".format(epoch, val_loss_arr[-1], val_acc_arr[-1]))

    np.save(os.path.join(save_dir, 'train_loss_arr.npy'), np.array(train_loss_arr))
    np.save(os.path.join(save_dir, 'train_acc_arr.npy'), np.array(train_acc_arr))
    np.save(os.path.join(save_dir, 'val_loss_arr.npy'), np.array(val_loss_arr))
    np.save(os.path.join(save_dir, 'val_acc_arr.npy'), np.array(val_acc_arr))

    plt.subplot(1, 2, 1)
    plt.title("loss")
    plt.plot(train_loss_arr, "r", label="train")
    plt.plot(val_loss_arr, "b", label="val")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("acc")
    plt.plot(train_acc_arr, "r", label="train")
    plt.plot(val_acc_arr, "b", label="val")
    plt.legend()

    plt.savefig("loss_acc.png")
    plt.show()

    print(f"Best model at epoch {best_epoch+1}, val_acc={best_val_acc:.4f}")
    print('Training completed!')


# 创建模型并训练
model = IntegratedNet(input_size=1,in_feature=157,num_classes=2)  # 确保模型的输出层适用于三分类问题
train_identityformer_model(model, model_name='MLPFormer_betch_16_fft_opendata',chennal=19,w_wight=2500)


```

测试代码

```
import torch.cuda
from torchvision import models
from dataset import *
from torch.utils.data import DataLoader
from torch import optim, nn
from dataset import *
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix, accuracy_score
from matplotlib import rcParams
import os
from net import *

labels = os.listdir("data_FFT_npy")
m = nn.Softmax(dim=1)
rcParams['font.family'] = 'SimHei'
def val(batch_size=16,w_wight=2500):
    # 数据集和数据加载器
    val_dataset = EEGDataset(csv_file='test_data.csv',transform=transform)

    val_data_loader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntegratedNet(input_size=1,in_feature=157,num_classes=2).to(device)
    model.load_state_dict(torch.load("MLPFormer_betch_16_fft_opendata.pth"))

    arr_y = []
    arr_y_pred = []
    for val_x, val_y in val_data_loader:
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        val_x = val_x.unsqueeze(1)
        val_x = val_x.view(batch_size, 1, 19, w_wight)
        val_y_pred = model(val_x)
        arr_y.extend(val_y.cpu().numpy())
        pred_result = m(val_y_pred).max(dim=1)[1]
        arr_y_pred.extend(pred_result.cpu().numpy())

    p = precision_score(arr_y, arr_y_pred, average="macro")
    recall = recall_score(arr_y, arr_y_pred, average="macro")
    f1 = f1_score(arr_y, arr_y_pred, average="macro")

    cm = confusion_matrix(arr_y, arr_y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    print("Precision: {:.5f}, Recall: {:.5f}, F1-score: {:.5f}, Specificity: {:.5f}".format(p, recall, f1, specificity))

    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(2), labels=labels)
    plt.yticks(range(2), labels=labels)

    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    thresh = cm.mean()
    for i in range(2):
        for j in range(2):
            info = cm[j, i]
            plt.text(i, j, info, color="white" if info > thresh else "black")
    plt.savefig("confusion_matrix.jpg")
    plt.show()





import matplotlib.pyplot as plt

def eval_single_sample(csv_file='test_data.csv', model_path='model/MLPFormer_betch_2_normal.pth'):
    # 数据集和数据加载器
    val_dataset = EEGDataset_eval(csv_file=csv_file, transform=transform)

    val_data_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntegratedNet().to(device)
    model.load_state_dict(torch.load(model_path))

    labels = ['认知功能障碍','轻度认知功能障碍','认知功能正常']
    m = nn.Softmax(dim=1)
    rcParams['font.family'] = 'SimHei'

    # 初始化一个列表来保存每个预测的概率
    all_probs = []

    for val_x in val_data_loader:
        val_x = val_x.to(device)
        val_x = val_x.unsqueeze(1)
        val_x = val_x.view(1, 1, 33, 1025)
        val_y_pred = model(val_x)
        pred_probs = m(val_y_pred).squeeze().detach().cpu().numpy()

        # 保存预测的概率
        all_probs.append(pred_probs)

    # 计算平均概率
    avg_probs = np.mean(all_probs, axis=0)

    # 绘制平均预测概率的柱状图
    plt.figure(figsize=(10, 6))
    plt.bar(labels, avg_probs)
    plt.xlabel('Classes')
    plt.ylabel('Average Predicted Probability')
    plt.title('Average Predicted Probabilities for Each Class')
    plt.show()

    pred_label = labels[np.argmax(avg_probs)]
    print(f"Most probable class: {pred_label} with average probability {np.max(avg_probs)*100:.2f}%")




if __name__ == "__main__":
    val()
    # eval_single_sample()


```


## 第二版-自动化

代码复现工程
92准确率复现

1. 提取set文件

```

import os
import shutil

def find_and_copy_set_files(source_folders, destination_folders):
    for source_folder, destination_folder in zip(source_folders, destination_folders):
        # 确保目标文件夹存在
        os.makedirs(destination_folder, exist_ok=True)
        
        # 遍历源文件夹中的所有文件和子文件夹
        for root, dirs, files in os.walk(source_folder):
            for file in files:
                if file.endswith('.set'):
                    # 构建源文件路径
                    source_file_path = os.path.join(root, file)
                    # 构建目标文件路径
                    destination_file_path = os.path.join(destination_folder, file)
                    
                    # 复制文件
                    shutil.copy2(source_file_path, destination_file_path)
                    print(f'Copied {source_file_path} to {destination_file_path}')

# 使用示例
source_folders = [
    'derivatives_class\AD',
    'derivatives_class\CN',
    'derivatives_class\FDT'
    # 添加更多源文件夹路径
]

destination_folders = [
    'data-set/AD',
    'data-set/CN',
    'data-set/FDT'
    # 添加更多目标文件夹路径
]

find_and_copy_set_files(source_folders, destination_folders)


```

2. 划分数据集

```

import os
import shutil
import random

# 指定包含数据文件的文件夹列表
data_folders = [
    "guding_channl/AD",
    "guding_channl/CN",
    'guding_channl/FDT'
]

# 对应的保存训练集和测试集的文件夹路径
train_folders = [
    "data/train/AD",
    "data/train/CN",
    'data/train/FDT'
   
]

test_folders = [
    "data/test/AD",
    "data/test/CN",
    'data/test/FDT'
   
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


2. 固定通道数


```
import os
import mne
import numpy as np

# 定义所需通道的列表
## 混合数据集共同的通道
channels = ['C3', 'Fz', 'F8', 'F4', 'C4', 'F3', 'Pz', 'P4', 'Cz', 'P3', 'F7']

## 公开数据集通道
# channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
# 输入和输出文件夹路径列表
input_folders = ['私有数据集/AD', '私有数据集/MCI', '私有数据集/NC', '公开数据集/AD', '公开数据集/CN']
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

3. 按照一定长度剪切数据集

```
import os
import numpy as np
from tqdm import tqdm

# 指定包含原始npy文件的文件夹路径列表
input_folders = ['data/test/AD', 'data/test/CN','data/test/FDT','data/train/AD', 'data/train/CN','data/train/FDT']  # 输入文件夹列表
output_folders = ['data_npy_cut/test/AD', 'data_npy_cut/test/CN','data_npy_cut/test/FDT','data_npy_cut/train/AD', 'data_npy_cut/train/CN','data_npy_cut/train/FDT']  # 对应的输出文件夹列表

# 确定剪切后的数据长度
cut_length = 5120

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



3. 第二种处理方法 按照psd处理

```
import mne
import numpy as np
import os


def eeg_power_band(epochs):
    """
    根据epochs的特定频段中的相对功率来创建EEG特征。
    """
    FREQ_BANDS = {"delta": [0.5, 4],
                  "theta": [4, 8],
                  "alpha": [8, 13],
                  "sigma": [13, 25],
                  "beta": [25, 45]}
    
    spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=45., n_fft=256, n_overlap=10)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    psds /= np.sum(psds, axis=-1, keepdims=True)
    
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        # psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)]
        X.append(psds_band.reshape(len(psds), -1))
    
    return np.concatenate(X, axis=1)




def process_eeg_data(input_folders, output_folders):
    """
    处理输入文件夹中的所有.set文件，并将提取的特征保存到输出文件夹中。
    """
    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith(".set"):
                set_file_path = os.path.join(input_folder, filename)
                
                # Load the .set file using MNE
                raw = mne.io.read_raw_eeglab(set_file_path, preload=True)
                
                raw.resample(512)
                # Set EEG reference
                raw.set_eeg_reference('average', projection=True)
                # Filter the data
                raw.filter(1., 45., fir_design='firwin')
                
                # Create events (you may need to modify this based on your data)
                events = mne.make_fixed_length_events(raw, start=0, duration=5.0)
                
                # Create Epochs object
                epochs = mne.Epochs(raw, events, tmin=0, tmax=4.0, baseline=None, preload=True)
                # print('epochs-info',epochs.info)
                # # Get original events data
                # original_events = epochs.get_data()
                # Extract features
                # print('original_events',original_events.shape)
                
                features = eeg_power_band(epochs)
                
                # Save features to output folder
                output_file_path = os.path.join(output_folder, filename.replace(".set", "_features.npy"))
                print(features.shape)
                np.save(output_file_path, features)
                
                print(f"Processed {filename} and saved features to {output_file_path}")

# Example usage
input_folders = ["data-remove/test/AD", "data-remove/test/CN",'data-remove/train/AD','data-remove/train/CN']
output_folders = ["psd/test/AD", "psd//test/CN",'psd/train/AD','psd/train/CN']
process_eeg_data(input_folders, output_folders)



```

4. 建立列表

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


5. PSD对应的CNN处理算法

```
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

# 自定义数据集类，从CSV文件中加载数据
class CsvDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        data = np.load(file_path)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# 定义简单的卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 22, 128)  # 根据实际情况调整输入大小
        self.fc2 = nn.Linear(128, 2)  # 假设有2个类别
        self.futool = StarReLU()

    def forward(self, x):
        x = self.conv1(x)
        # print('1',x.shape)
        x = self.futool(x)
        x = self.pool(x)
        # print('2',x.shape)
        x = self.conv2(x)
        # print('3',x.shape)
        x = self.futool(x)
        x = self.pool(x)
        # print('4',x.shape)
        x = x.view(-1, 32 * 22)
        # print('4-1',x.shape)
        x = self.fc1(x)
        # print('5',x.shape)
        x = self.futool(x)
        # print('6',x.shape)
        x = self.fc2(x)
        # print('7',x.shape)
        return x


import torch

import torch
from tqdm import tqdm

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, save_path='best_model.pth'):
    model.train()
    total_samples = len(train_loader.dataset)
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)

        for data, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(data.unsqueeze(1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix({'loss': running_loss / total_samples, 'accuracy': correct_predictions / total_samples})

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        # Validate the model
        model.eval()
        total_samples_val = len(val_loader.dataset)
        running_loss_val = 0.0
        correct_predictions_val = 0

        with torch.no_grad():
            for data, labels in val_loader:
                outputs = model(data.unsqueeze(1))
                loss_val = criterion(outputs, labels)
                running_loss_val += loss_val.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions_val += (predicted == labels).sum().item()

            epoch_loss_val = running_loss_val / total_samples_val
            epoch_accuracy_val = correct_predictions_val / total_samples_val

            print(f'Validation Loss: {epoch_loss_val:.4f}, Accuracy: {epoch_accuracy_val:.4f}')

            if epoch_accuracy_val > best_val_acc:
                best_val_acc = epoch_accuracy_val
                torch.save(model.state_dict(), save_path)
                print(f'Saved the model with the best validation accuracy: {best_val_acc:.4f}')

    print('Training finished')





# CSV训练集文件路径
train_csv_file_path = 'train_data.csv'
# CSV测试集文件路径
test_csv_file_path = 'test_data.csv'

# 创建训练集和测试集数据集实例
train_dataset = CsvDataset(train_csv_file_path)
test_dataset = CsvDataset(test_csv_file_path)

# 创建训练集和测试集数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False,drop_last=True)

# 初始化模型、损失函数和优化器
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.0001)

# 训练模型
train(model, train_loader, test_loader, criterion, optimizer, num_epochs=100)




```

6. PSD+CNN 绘制混淆矩阵

```
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True, 
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias

# 自定义数据集类，从CSV文件中加载数据
class CsvDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        data = np.load(file_path)
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# 定义简单的卷积神经网络模型
class CNN(nn.Module):
    def __init__(self,num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 22, 128)  # 根据实际情况调整输入大小
        self.fc2 = nn.Linear(128, num_classes)  # 假设有2个类别
        self.futool = StarReLU()

    def forward(self, x):
        x = self.conv1(x)
        # print('1',x.shape)
        x = self.futool(x)
        x = self.pool(x)
        # print('2',x.shape)
        x = self.conv2(x)
        # print('3',x.shape)
        x = self.futool(x)
        x = self.pool(x)
        # print('4',x.shape)
        x = x.view(-1, 32 * 22)
        # print('4-1',x.shape)
        x = self.fc1(x)
        # print('5',x.shape)
        x = self.futool(x)
        # print('6',x.shape)
        x = self.fc2(x)
        # print('7',x.shape)
        return x



# 设置中文显示
rcParams['font.family'] = 'SimHei'

labels = ['CN', 'FDT']
softmax = nn.Softmax(dim=1)

def val(batch_size=16):
    # 数据集和数据加载器
    val_dataset =  CsvDataset(csv_file='test_data.csv')
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(num_classes=3).to(device)
    model.load_state_dict(torch.load("best_model.pth"))

    arr_y = []
    arr_y_pred = []
    for val_x, val_y in val_data_loader:
        val_x = val_x.to(device)
        val_y = val_y.to(device)
        val_x = val_x.unsqueeze(1)
        val_y_pred = model(val_x)
        arr_y.extend(val_y.cpu().numpy())
        pred_result = softmax(val_y_pred).max(dim=1)[1]
        arr_y_pred.extend(pred_result.cpu().numpy())

    accuracy = accuracy_score(arr_y, arr_y_pred)
    precision = precision_score(arr_y, arr_y_pred, average="macro")
    recall = recall_score(arr_y, arr_y_pred, average="macro")
    f1 = f1_score(arr_y, arr_y_pred, average="macro")

    # 计算特异度
    cm = confusion_matrix(arr_y, arr_y_pred)
    specificity = []
    for i in range(len(labels)):
        TN = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        FP = cm[:, i].sum() - cm[i, i]
        specificity.append(TN / (TN + FP))
    avg_specificity = np.mean(specificity)

    print(f"Accuracy: {accuracy:.5f}, Precision: {precision:.5f}, Recall: {recall:.5f}, F1: {f1:.5f}, Specificity: {avg_specificity:.5f}")

    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels=labels)
    plt.yticks(range(len(labels)), labels=labels)

    plt.colorbar()
    plt.xlabel("预测值")
    plt.ylabel("真实值")
    thresh = cm.mean()
    for i in range(len(labels)):
        for j in range(len(labels)):
            info = cm[j, i]
            prob = info / np.sum(cm[j])
            plt.text(i, j, f"{info}\n({prob*100:.2f}%)", color="white" if info > thresh else "black", ha='center', va='center')
    plt.savefig("confusion_matrix.jpg")
    plt.show()

if __name__ == "__main__":
    val()

```
