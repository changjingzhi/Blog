---
title: 论文写作解析
date: 2024-06-24 11:21:16
tags: 论文思路
---

写作论文，也是需要具体的格式的，在[论文思路——论文阅读文献](https://chenlidbk.xyz/2024/04/30/paper-idear3/#%E8%AE%BA%E6%96%87%E7%BB%93%E6%9E%84) 大概描述了论文的结构，但是一个欠缺点是这只是一个架构，只是骨架，没有血肉，下面做的工作就是来填充论文内容过程的详细阐述。写论文，首先得想好自己的创新点是什么，然后验证自己的创新点是否有用，这里需要注意的是实验的完备性，不要给自己挖坑。

## 论文内容

### 论文的标题 关于题目：题目的书写有三个方案，

1.可以写的长并且有好的解释性：Linear-Time External Multipass Sorting with Approximation Guarantees

2.可以写的甜美小巧（ short and sweet）Approximate External Sort

3.或者一种折中的思路，写一个俏皮幽默的小标题，去吸人眼球，直击人心（ plus a cute name that sticks in people's minds）Floosh: A Linear-Time Algorithm for Approximate External Sort


### 论文的abstract

abstract要陈述问题，你的方法和解决方案，以及论文的主要贡献。几乎不包括任何背景和动机。事实要真实，但要全面。摘要中的内容以后不得逐字重复。

### The Introduction

强调写在前面：导言非常重要（crucially important），审稿人可能会在看完introduction之后产生了一个initial decision决定要不要接受paper，他们会阅读论文的其余部分，目的其实就是为了印证他们的看法

关于写法紧跟在后：这是斯坦福InfoLab的五点介绍结构。除非有更好的，否则引言应该由五个段落组成，回答以下五个问题:

What is the problem?
Why is it interesting and important?
Why is it hard? (E.g., why do naive approaches fail?)
Why hasn't it been solved before? (Or, what's wrong with previous proposed solutions? How does mine differ?)
What are the key components of my approach and results? Also include any specific limitations.

### Related Work

关于相关工作，（这个有一个说法（mentor说的），说是这个是最不重要的内容，然后一般就是长短可以自适应。）这里探讨的是，related work是应该放在文章开头还是结尾？


放开头，如果内容简短又详细，或者是对前面的工作是一个防御立场，并且比较重要，在这种情况下，相关的工作可以是引言末尾的一个小节，或者是它自己的第2节。
Beginning, if it can be short yet detailed enough, or if it's critical to take a strong defensive stance about previous work right away. In this case Related Work can be either a subsection at the end of the Introduction, or its own Section 2.
 

放到末尾，如果introduction直接就能够把故事讲圆了，不是很需要放到前面，那就放到后面去

End, if it can be summarized quickly early on (in the Introduction or Preliminaries), or if sufficient comparisons require the technical content of the paper. In this case Related Work should appear just before the Conclusions, possibly in a more general section "Discussion and Related Work".

我的一般做法是放在introduct中


## 主体

1.文章里提到了一个重点，一个清晰的新的重要的技术贡献应该在读者完成第3页的时候表达出来

2.另一个重点，文章每一部分都应该讲述一个完整的故事

话不多说，让我开始写字吧

## 实操部分
工具台： 浏览器， word， vision，  绘制网络结构的网站 [链接](https://alexlenail.me/NN-SVG/LeNet.html)



## 最近觉得

最近觉得自己做的课题没有什么意义，一种原因是文献看的不够多，多关注领域内文章的discussion部分，一般来说，这部分最能体现出作者的观点的，而且会提到一些可能再深入的方向。