---
title: github
date: 2024-05-13 14:00:52
tags:  填坑
---


[github工作流程代码解释](https://ndpsoftware.com/git-cheatsheet.html#loc=index;)

## 背景介绍
GitHub是一个代码托管平台。一个共享和开源软件的流行平台
1. 软件开源，即编写软件的代码对所有人公开，所有人可以在现有代码的基础上进行二次开发，减少不必要的重复劳动（ IT 行话简称不要重复「造轮子」）
2. 方便团队协作。这个过程有点像是我们把文档放在石墨或语雀这类支持团队协作的平台上，而 GitHub 上存放的是代码，参与编写软件的人可以通过 Git（版本控制工具）从 GitHub 拉取或往 GitHub 上传代码

类似的代码托管平台还有gitee，开源中国。

仓库（Repository） 仓库 是 GitHub 最基本的元素。 它们很容易被想象为项目的文件夹。 仓库包含所有项目文件（包括文档），并存储每个文件的修改历史记录。 仓库可以有多个协作者，仓库可以是公开的，也可以设置为私有的。更详细的请看[参考文档](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories)
分支（Branch） 分支是仓库的并行版本。默认情况下，您的仓库具有一个名为 main 的主分支。我们可以复制主分支创建其他分支，您安全地进行任何更改而不会影响”线上“主分支。 完成所需更改后，可以将分支合并回主分支以发布你的更改。
README：GitHub 个人主页资料上 “关于我” 的介绍。 内容一般包含：介绍您的工作和兴趣，您引以为豪的贡献以及这些贡献的背景信息，在您参与的社区获得帮助的指南
## 配套软件
[git for windown ](https://git-scm.com/download/win)
## 使用命令
初始命令，建立.git文件
```
git init 
```

选取上传文件，如果需要选取所有则使用`git add .`
```
git add README.md
git add .
```

书写提交信息
```
git commit -m "first commit              "
```
选取分支
```
git branch -M main
```
添加目标位置，
```
git remote add origin git@github.com:changjingzhi/test.git
```

(origin 表示名字)
git push 提交文件到远端仓库
```
git push -u origin main
```

git clone 克隆远端仓库。
```
git clone 远端仓库地址
```


## python

使用pip freeze命令生成requirements.txt文件：
```
pip freeze > requirements.txt

```


model环境
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