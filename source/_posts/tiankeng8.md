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