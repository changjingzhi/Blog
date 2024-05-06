---
title: MetaFormer Baselines for Vision
date: 2024-05-04 15:33:28
tags: 深度学习论文
---

[参考博客-知乎](https://zhuanlan.zhihu.com/p/577709208)
[论文原文链接](https://arxiv.org/pdf/2210.13452)
[读论文博客](https://zhuanlan.zhihu.com/p/579175302)
[原文代码](https://zhuanlan.zhihu.com/p/579175302)


## 原文参考
Nevertheless, some work [17](https://arxiv.org/pdf/2105.01601), [18](https://arxiv.org/pdf/2105.03824), [19](https://arxiv.org/pdf/2108.13002),[20](https://arxiv.org/pdf/2106.04263), [21](https://arxiv.org/pdf/2107.00645) found that, by replacing the attention module in Transformers with simple operators like spatial MLP [17](https://arxiv.org/pdf/2105.01601),[22](https://arxiv.org/pdf/2105.03404), [23](https://arxiv.org/pdf/2005.00743) or Fourier transform [18], the resultant models still produce encouraging performance.

基于此这篇论文团队之前的工作在[24](https://arxiv.org/pdf/2111.11418)这篇论文中他们提出了MetaFormer，为了更进一步的验证MetaFormer，Our goal is to push the limits of MetaFomer, based on which we may have a comprehensive picture of its capacity. 团队选取了最basic or common token mixers ，such as ，identity mapping or global random mixing， swparable convolution [6],[7] ，[8]  and vanilla self-attention [9]


## 挖坑

### 代码解析工程