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

将npy数据剪切到想要的长度
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

