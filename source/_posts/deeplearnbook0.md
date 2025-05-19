---
title: 深度学习环境配置-pytroch
date: 2024-05-20 20:20:27
tags: 深度学习
---


## 环境配置
查看显卡能支持最高的CUDA版本：`nvidia-smi`
查看安装的CUDA `nvcc -V`

1. 安装NVIDIA驱动,[3080](https://www.nvidia.cn/content/DriverDownloads/confirmation.php?url=/Windows/552.44/552.44-desktop-win10-win11-64bit-international-dch-whql.exe&lang=cn&type=GeForce)
2. 安装CUDA Toolkit [](https://developer.nvidia.com/cuda-toolkit-archive)











































```
absl-py==2.0.0
addict==2.4.0
asttokens==2.4.1
backcall==0.2.0
cachetools==5.3.2
certifi==2023.7.22
chardet==5.2.0
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
comm==0.2.0
contourpy==1.1.1
cycler==0.12.1
debugpy==1.8.0
decorator==5.1.1
defusedxml==0.7.1
executing==2.0.1
filelock==3.13.1
fonttools==4.45.1
fsspec==2023.10.0
google-auth==2.23.4
google-auth-oauthlib==1.0.0
graphviz==0.20.1
grpcio==1.59.3
huggingface-hub==0.19.4
idna==3.4
imageio==2.33.0
importlib-metadata==6.8.0
importlib-resources==6.1.1
ipykernel==6.26.0
ipython==8.12.3
jedi==0.19.1
Jinja2==3.1.2
joblib==1.3.2
jupyter_client==8.6.0
jupyter_core==5.5.0
kiwisolver==1.4.5
lazy_loader==0.3
lxml==4.9.3
Markdown==3.5.1
MarkupSafe==2.1.3
matplotlib==3.7.4
matplotlib-inline==0.1.6
mne==1.6.0
nest-asyncio==1.5.8
netron==7.2.9
networkx==3.1
nltk==3.8.1
numpy==1.24.4
oauthlib==3.2.2
opencv-python==4.2.0.32
packaging==23.2
pandas==2.0.3
parso==0.8.3
pickleshare==0.7.5
Pillow==10.1.0
platformdirs==4.0.0
pooch==1.8.0
prompt-toolkit==3.0.41
protobuf==4.25.1
psutil==5.9.6
pure-eval==0.2.2
pyasn1==0.5.1
pyasn1-modules==0.3.0
pycocotools==2.0.7
pyemd==1.0.0
pygame==2.5.2
Pygments==2.16.1
pyparsing==3.1.1
python-dateutil==2.8.2
pytz==2024.1
PyWavelets==1.4.1
pywin32==306
PyYAML==6.0.1
pyzmq==25.1.1
regex==2023.10.3
requests==2.31.0
requests-oauthlib==1.3.1
rsa==4.9
safetensors==0.4.0
scikit-image==0.21.0
scikit-learn==1.3.2
scipy==1.10.1
seaborn==0.13.2
six==1.16.0
stack-data==0.6.3
summary==0.2.0
tensorboard==2.14.0
tensorboard-data-server==0.7.2
thop==0.1.1.post2209072238
threadpoolctl==3.2.0
tifffile==2023.7.10
timm==0.6.13
torch==1.12.1+cu116
torch-tb-profiler==0.4.3
torchaudio==0.12.1+cu116
torchsummary==1.5.1
torchvision==0.13.1+cu116
torchviz==0.0.2
tornado==6.3.3
tqdm==4.66.1
traitlets==5.13.0
typing_extensions==4.8.0
tzdata==2024.1
urllib3==2.1.0
wcwidth==0.2.10
Werkzeug==3.0.1
zipp==3.17.0

```