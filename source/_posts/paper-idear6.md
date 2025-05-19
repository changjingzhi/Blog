---
title:  PSD-公开数据集
date: 2024-06-11 13:10:58
tags: 论文思路
---

1. psd-分解
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
                
                # Set EEG reference
                raw.set_eeg_reference('average', projection=True)
                
                # Filter the data
                raw.filter(1., 45., fir_design='firwin')
                
                # Create events (you may need to modify this based on your data)
                events = mne.make_fixed_length_events(raw, start=0, duration=5.0)
                
                # Create Epochs object
                epochs = mne.Epochs(raw, events, tmin=0, tmax=4.0, baseline=None, preload=True)
                
                # Extract features
                features = eeg_power_band(epochs)
                
                # Save features to output folder
                output_file_path = os.path.join(output_folder, filename.replace(".set", "_features.npy"))
                print(features.shape)
                np.save(output_file_path, features)
                
                print(f"Processed {filename} and saved features to {output_file_path}")

# Example usage
input_folders = ["公开数据集/CN", "公开数据集/AD",'公开数据集/FDT']
output_folders = ["psd/CN", "psd/AD",'psd/FDT']
process_eeg_data(input_folders, output_folders)

```

2. 划分训练和测试

```
import os
import shutil
import random

# 指定包含数据文件的文件夹列表
data_folders = [
    "psd/AD",
    "psd/CN",
    "psd/FDT"
]

# 对应的保存训练集和测试集的文件夹路径
train_folders = [
    "psd_data/train/AD",
    "psd_data/train/CN",
    "psd_data/train/FDT"

]

test_folders = [
    "psd_data/test/AD",
    "psd_data/test/CN",
    "psd_data/test/FDT"
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
            if random.random() < 0.7:
                shutil.copy(source_file, train_folder)
            else:
                shutil.copy(source_file, test_folder)
        except Exception as e:
            print(f"无法复制文件 {file_name}: {e}")

    # 打印训练集和测试集的文件数量
    print(f"{data_folder} -> 训练集大小:", len(os.listdir(train_folder)))
    print(f"{data_folder} -> 测试集大小:", len(os.listdir(test_folder)))

```

2 划分数据集debug版本

```
import os
import shutil
import random

# 指定包含数据文件的文件夹列表
data_folders = [
    "data-remove/AD",
    "data-remove/CN",
    "data-remove/MCI"
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
    
    # 打乱文件列表
    random.shuffle(data_files)
    
    # 按8:2比例分配文件
    split_point = int(0.8 * len(data_files))
    train_files = data_files[:split_point]
    test_files = data_files[split_point:]

    # 复制文件到训练集文件夹
    for file_name in train_files:
        source_file = os.path.join(data_folder, file_name)
        try:
            shutil.copy(source_file, train_folder)
        except Exception as e:
            print(f"无法复制文件 {file_name} 到训练集: {e}")

    # 复制文件到测试集文件夹
    for file_name in test_files:
        source_file = os.path.join(data_folder, file_name)
        try:
            shutil.copy(source_file, test_folder)
        except Exception as e:
            print(f"无法复制文件 {file_name} 到测试集: {e}")

    # 打印训练集和测试集的文件数量
    print(f"{data_folder} -> 训练集大小:", len(os.listdir(train_folder)))
    print(f"{data_folder} -> 测试集大小:", len(os.listdir(test_folder)))


```


3.建立列表

```

import os
import numpy as np

# 输入文件夹路径列表
input_folder_paths = ['psd_data/test/AD', 'psd_data/test/CN', 'psd_data/test/FDT','psd_data/train/AD','psd_data/train/CN','psd_data/train/FDT']

# 输出文件夹路径列表
output_folder_paths = ['psd_data_cut/test/AD','psd_data_cut/test/CN','psd_data_cut/test/FDT','psd_data_cut/train/AD', 'psd_data_cut/train/CN','psd_data_cut/train/FDT']

# 如果输出文件夹不存在，则创建它们
for output_folder_path in output_folder_paths:
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

# 遍历每个输入文件夹
for input_folder_path, output_folder_path in zip(input_folder_paths, output_folder_paths):
    # 获取输入文件夹中的所有.npy文件
    npy_files = [f for f in os.listdir(input_folder_path) if f.endswith('.npy')]

    for npy_file in npy_files:
        # 完整文件路径
        file_path = os.path.join(input_folder_path, npy_file)
        
        # 加载数据
        data = np.load(file_path)
        
        # 确保数据的第二个维度是95
        if data.shape[1] != 95:
            print(f"文件 {npy_file} 不是期望的形状 {data.shape}")
            continue
        
        # 保存每个通道的数据到单独的.npy文件中
        base_filename = os.path.splitext(npy_file)[0]
        for i in range(data.shape[0]):
            output_filename = f"{base_filename}_channel_{i}.npy"
            output_path = os.path.join(output_folder_path, output_filename)
            np.save(output_path, data[i])
            print(f"保存 {output_path} 成功!")

print("所有文件处理完成!")


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
    create_data_text("psd_data_cut")


```

5. 训练模型

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
optimizer = optim.RMSprop(model.parameters(), lr=0.0005)

# 训练模型
train(model, train_loader, test_loader, criterion, optimizer, num_epochs=100)

```


6. 测试代码

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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3)
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
    model = CNN(num_classes=2).to(device)
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


## 实践过程中的问题。

1. 对于不同的学习率，激活函数，模型结构所组合出来的实验的ACC都不一样。


2. 针对一个问题进行优化




## 混合数据集处理流程

### PSD+CNN

1. 进行PSD+CNN处理

```

import mne
import numpy as np
import os

channels = ['C3', 'Fz', 'F8', 'F4', 'C4', 'F3', 'Pz', 'P4', 'Cz', 'P3', 'F7']

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
        X.append(psds_band.reshape(len(psds), -1))
    return np.concatenate(X, axis=1)

def process_eeg_files(input_folders, output_folders):
    """
    处理输入文件夹中的所有 .set 和 .edf 文件，并将提取的特征保存到输出文件夹中。
    """
    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            file_path = os.path.join(input_folder, filename)

            if filename.endswith(".set"):
                # Load the .set file using MNE
                raw = mne.io.read_raw_eeglab(file_path, preload=True)
            elif filename.endswith(".edf"):
                # Load the .edf file using MNE
                raw = mne.io.read_raw_edf(file_path, preload=True)
            else:
                continue  # 如果文件不是.set或.edf格式，则跳过

            if not all(ch in raw.ch_names for ch in channels):
                print(f"文件 {file_path} 不包含所有所需通道，跳过处理。")
                continue

            raw.pick_channels(channels)
    
            # 按照指定顺序重新排列通道
            raw.reorder_channels(channels)
            raw.resample(512)
            # 设置EEG参考
            raw.set_eeg_reference('average', projection=True)
            # 过滤数据
            raw.filter(1., 45., fir_design='firwin')

            # 创建事件
            events = mne.make_fixed_length_events(raw, start=0, duration=5.0)

            # 创建Epochs对象
            epochs = mne.Epochs(raw, events, tmin=0, tmax=5.0, baseline=None, preload=True)

            # 提取特征
            features = eeg_power_band(epochs)

            # 保存特征到输出文件夹
            output_file_path = os.path.join(output_folder, filename.replace(".set", "_features.npy").replace(".edf", "_features.npy"))
            np.save(output_file_path, features)

            print(f"Processed {filename} and saved features to {output_file_path}")

# 示例用法
input_folders = ["数据集划分8-2数据划分-编号2-去除伪迹版混合数据集/test/AD", "数据集划分8-2数据划分-编号2-去除伪迹版混合数据集/test/CN", "数据集划分8-2数据划分-编号2-去除伪迹版混合数据集/test/MCI", "数据集划分8-2数据划分-编号2-去除伪迹版混合数据集/train/AD", "数据集划分8-2数据划分-编号2-去除伪迹版混合数据集/train/CN", "数据集划分8-2数据划分-编号2-去除伪迹版混合数据集/train/MCI"]
output_folders = ["psd/test/AD", "psd/test/CN", "psd/test/MCI", "psd/train/AD", "psd/train/CN", "psd/train/MCI"]
process_eeg_files(input_folders, output_folders)


```

2. 按照要求剪切数据
```

import os
import numpy as np

# 输入文件夹路径列表
input_folder_paths = ["psd/test/AD", "psd//test/CN",'psd//test/MCI','psd/train/AD','psd/train/CN','psd/train/MCI']
# 输出文件夹路径列表
output_folder_paths = ['psd_data_cut/test/AD','psd_data_cut/test/CN','psd_data_cut/test/MCI','psd_data_cut/train/AD', 'psd_data_cut/train/CN','psd_data_cut/train/MCI']

# 如果输出文件夹不存在，则创建它们
for output_folder_path in output_folder_paths:
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

# 遍历每个输入文件夹
for input_folder_path, output_folder_path in zip(input_folder_paths, output_folder_paths):
    # 获取输入文件夹中的所有.npy文件
    npy_files = [f for f in os.listdir(input_folder_path) if f.endswith('.npy')]

    for npy_file in npy_files:
        # 完整文件路径
        file_path = os.path.join(input_folder_path, npy_file)
        
        # 加载数据
        data = np.load(file_path)
        
        # 确保数据的第二个维度是95
        if data.shape[1] != 55:
            print(f"文件 {npy_file} 不是期望的形状 {data.shape}")
            continue
        
        # 保存每个通道的数据到单独的.npy文件中
        base_filename = os.path.splitext(npy_file)[0]
        for i in range(data.shape[0]):
            output_filename = f"{base_filename}_channel_{i}.npy"
            output_path = os.path.join(output_folder_path, output_filename)
            np.save(output_path, data[i])
            print(f"保存 {output_path} 成功!")

print("所有文件处理完成!")


```


3. 建立列表

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
    create_data_text("psd_data_cut")


```



4. 训练CNN
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
        self.fc1 = nn.Linear(32 * 12, 128)  # 根据实际情况调整输入大小
        self.fc2 = nn.Linear(128, 3)  # 假设有2个类别
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
        x = x.view(-1, 32 * 12)
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


5. 测试CNN
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
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=32, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 22, 128)  # 根据实际情况调整输入大小
        self.fc2 = nn.Linear(128, 3)  # 假设有2个类别
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
    model = CNN(num_classes=2).to(device)
    model.load_state_dict(torch.load("87.12.pth"))

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



### mlpformer

