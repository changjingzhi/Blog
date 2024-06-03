---
title: 分类模型-MetaFormer Baselines for Vision
date: 2024-05-04 15:33:28
tags: 深度学习论文
---

[参考博客-知乎](https://zhuanlan.zhihu.com/p/577709208)
[论文原文链接](https://arxiv.org/pdf/2210.13452)
[读论文博客](https://zhuanlan.zhihu.com/p/579175302)
[原文代码](https://github.com/sail-sg/metaformer)


## 原文参考
Nevertheless, some work [17](https://arxiv.org/pdf/2105.01601), [18](https://arxiv.org/pdf/2105.03824), [19](https://arxiv.org/pdf/2108.13002),[20](https://arxiv.org/pdf/2106.04263), [21](https://arxiv.org/pdf/2107.00645) found that, by replacing the attention module in Transformers with simple operators like spatial MLP [17](https://arxiv.org/pdf/2105.01601),[22](https://arxiv.org/pdf/2105.03404), [23](https://arxiv.org/pdf/2005.00743) or Fourier transform [18], the resultant models still produce encouraging performance.
最近，一些工作，17，18，19，20 发现，通过在Transformers中的自注意力为简单的operators 为MLP或者其他的 Fourier transform，模型仍然保持着良好的表现。


基于此这篇论文团队之前的工作在[24](https://arxiv.org/pdf/2111.11418)这篇论文中他们提出了MetaFormer，为了更进一步的验证MetaFormer，Our goal is to push the limits of MetaFomer, based on which we may have a comprehensive picture of its capacity. 团队选取了最basic or common token mixers ，such as ，identity mapping or global random mixing， swparable convolution [6],[7] ，[8]  and vanilla self-attention [9]

![模型结构图](pic/MetaFormer1.png)
![模型实验图](pic/MateFormer2.png)
![模型实验图](pic/MateFormer3.png)
## 挖坑

MACs（Multiply-Accumulate Operations，乘加运算）是衡量神经网络计算复杂性的一个重要指标。它表示神经网络在进行一次前向传播（即推理）时所需的乘加运算次数。乘加运算是神经网络中最基本的操作，尤其在卷积层和全连接层中广泛使用。

### 代码解析工程

metaformer_baselines.py
```
# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MetaFormer baselines including IdentityFormer, RandFormer, PoolFormerV2,
ConvFormer and CAFormer.
Some implementations are modified from timm (https://github.com/rwightman/pytorch-image-models).
"""
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers.helpers import to_2tuple


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 1.0, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'identityformer_s12': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s12.pth'),
    'identityformer_s24': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s24.pth'),
    'identityformer_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_s36.pth'),
    'identityformer_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m36.pth'),
    'identityformer_m48': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/identityformer/identityformer_m48.pth'),


    'randformer_s12': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s12.pth'),
    'randformer_s24': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s24.pth'),
    'randformer_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_s36.pth'),
    'randformer_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m36.pth'),
    'randformer_m48': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/randformer/randformer_m48.pth'),

    'poolformerv2_s12': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s12.pth'),
    'poolformerv2_s24': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s24.pth'),
    'poolformerv2_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_s36.pth'),
    'poolformerv2_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m36.pth'),
    'poolformerv2_m48': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/poolformerv2/poolformerv2_m48.pth'),



    'convformer_s18': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18.pth'),
    'convformer_s18_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384.pth',
        input_size=(3, 384, 384)),
    'convformer_s18_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21ft1k.pth'),
    'convformer_s18_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'convformer_s18_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s18_in21k.pth',
        num_classes=21841),

    'convformer_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36.pth'),
    'convformer_s36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384.pth',
        input_size=(3, 384, 384)),
    'convformer_s36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21ft1k.pth'),
    'convformer_s36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'convformer_s36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_s36_in21k.pth',
        num_classes=21841),

    'convformer_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36.pth'),
    'convformer_m36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384.pth',
        input_size=(3, 384, 384)),
    'convformer_m36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21ft1k.pth'),
    'convformer_m36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'convformer_m36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_m36_in21k.pth',
        num_classes=21841),

    'convformer_b36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36.pth'),
    'convformer_b36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384.pth',
        input_size=(3, 384, 384)),
    'convformer_b36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21ft1k.pth'),
    'convformer_b36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'convformer_b36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/convformer/convformer_b36_in21k.pth',
        num_classes=21841),


    'caformer_s18': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18.pth'),
    'caformer_s18_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384.pth',
        input_size=(3, 384, 384)),
    'caformer_s18_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21ft1k.pth'),
    'caformer_s18_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'caformer_s18_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s18_in21k.pth',
        num_classes=21841),

    'caformer_s36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36.pth'),
    'caformer_s36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384.pth',
        input_size=(3, 384, 384)),
    'caformer_s36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21ft1k.pth'),
    'caformer_s36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'caformer_s36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_s36_in21k.pth',
        num_classes=21841),

    'caformer_m36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36.pth'),
    'caformer_m36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384.pth',
        input_size=(3, 384, 384)),
    'caformer_m36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21ft1k.pth'),
    'caformer_m36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'caformer_m36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_m36_in21k.pth',
        num_classes=21841),

    'caformer_b36': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36.pth'),
    'caformer_b36_384': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384.pth',
        input_size=(3, 384, 384)),
    'caformer_b36_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21ft1k.pth'),
    'caformer_b36_384_in21ft1k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_384_in21ft1k.pth',
        input_size=(3, 384, 384)),
    'caformer_b36_in21k': _cfg(
        url='https://huggingface.co/sail/dl/resolve/main/caformer/caformer_b36_in21k.pth',
        num_classes=21841),
}


class Downsampling(nn.Module):
    """
    Downsampling implemented by a layer of convolution.
    """
    def __init__(self, in_channels, out_channels, 
        kernel_size, stride=1, padding=0, 
        pre_norm=None, post_norm=None, pre_permute=False):
        super().__init__()
        self.pre_norm = pre_norm(in_channels) if pre_norm else nn.Identity()
        self.pre_permute = pre_permute
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding)
        self.post_norm = post_norm(out_channels) if post_norm else nn.Identity()

    def forward(self, x):
        x = self.pre_norm(x)
        if self.pre_permute:
            # if take [B, H, W, C] as input, permute it to [B, C, H, W]
            x = x.permute(0, 3, 1, 2)
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
        x = self.post_norm(x)
        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale
        

class SquaredReLU(nn.Module):
    """
        Squared ReLU: https://arxiv.org/abs/2109.08668
    """
    def __init__(self, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
    def forward(self, x):
        return torch.square(self.relu(x))


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


class Attention(nn.Module):
    """
    Vanilla self-attention from Transformer: https://arxiv.org/abs/1706.03762.
    Modified from timm.
    """
    def __init__(self, dim, head_dim=32, num_heads=None, qkv_bias=False,
        attn_drop=0., proj_drop=0., proj_bias=False, **kwargs):
        super().__init__()

        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.num_heads = num_heads if num_heads else dim // head_dim
        if self.num_heads == 0:
            self.num_heads = 1
        
        self.attention_dim = self.num_heads * self.head_dim

        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        
    def forward(self, x):
        B, H, W, C = x.shape
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RandomMixing(nn.Module):
    def __init__(self, num_tokens=196, **kwargs):
        super().__init__()
        self.random_matrix = nn.parameter.Parameter(
            data=torch.softmax(torch.rand(num_tokens, num_tokens), dim=-1), 
            requires_grad=False)
    def forward(self, x):
        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        x = torch.einsum('mn, bnc -> bmc', self.random_matrix, x)
        x = x.reshape(B, H, W, C)
        return x


class LayerNormGeneral(nn.Module):
    r""" General LayerNorm for different situations.

    Args:
        affine_shape (int, list or tuple): The shape of affine weight and bias.
            Usually the affine_shape=C, but in some implementation, like torch.nn.LayerNorm,
            the affine_shape is the same as normalized_dim by default. 
            To adapt to different situations, we offer this argument here.
        normalized_dim (tuple or list): Which dims to compute mean and variance. 
        scale (bool): Flag indicates whether to use scale or not.
        bias (bool): Flag indicates whether to use scale or not.

        We give several examples to show how to specify the arguments.

        LayerNorm (https://arxiv.org/abs/1607.06450):
            For input shape of (B, *, C) like (B, N, C) or (B, H, W, C),
                affine_shape=C, normalized_dim=(-1, ), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, ), scale=True, bias=True.

        Modified LayerNorm (https://arxiv.org/abs/2111.11418)
            that is idental to partial(torch.nn.GroupNorm, num_groups=1):
            For input shape of (B, N, C),
                affine_shape=C, normalized_dim=(1, 2), scale=True, bias=True;
            For input shape of (B, H, W, C),
                affine_shape=C, normalized_dim=(1, 2, 3), scale=True, bias=True;
            For input shape of (B, C, H, W),
                affine_shape=(C, 1, 1), normalized_dim=(1, 2, 3), scale=True, bias=True.

        For the several metaformer baslines,
            IdentityFormer, RandFormer and PoolFormerV2 utilize Modified LayerNorm without bias (bias=False);
            ConvFormer and CAFormer utilizes LayerNorm without bias (bias=False).
    """
    def __init__(self, affine_shape=None, normalized_dim=(-1, ), scale=True, 
        bias=True, eps=1e-5):
        super().__init__()
        self.normalized_dim = normalized_dim
        self.use_scale = scale
        self.use_bias = bias
        self.weight = nn.Parameter(torch.ones(affine_shape)) if scale else None
        self.bias = nn.Parameter(torch.zeros(affine_shape)) if bias else None
        self.eps = eps

    def forward(self, x):
        c = x - x.mean(self.normalized_dim, keepdim=True)
        s = c.pow(2).mean(self.normalized_dim, keepdim=True)
        x = c / torch.sqrt(s + self.eps)
        if self.use_scale:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x


class LayerNormWithoutBias(nn.Module):
    """
    Equal to partial(LayerNormGeneral, bias=False) but faster, 
    because it directly utilizes otpimized F.layer_norm
    """
    def __init__(self, normalized_shape, eps=1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.bias = None
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """
    def __init__(self, dim, expansion_ratio=2,
        act1_layer=StarReLU, act2_layer=nn.Identity, 
        bias=False, kernel_size=7, padding=3,
        **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=padding, groups=med_channels, bias=bias) # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer: https://arxiv.org/abs/2111.11418
    Modfiled for [B, H, W, C] input
    """
    def __init__(self, pool_size=3, **kwargs):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = self.pool(y)
        y = y.permute(0, 2, 3, 1)
        return y - x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MlpHead(nn.Module):
    """ MLP classification head
    """
    def __init__(self, dim, num_classes=1000, mlp_ratio=4, act_layer=SquaredReLU,
        norm_layer=nn.LayerNorm, head_dropout=0., bias=True):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)
        self.head_dropout = nn.Dropout(head_dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.head_dropout(x)
        x = self.fc2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """
    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()
        
    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


r"""
downsampling (stem) for the first stage is a layer of conv with k7, s4 and p2
downsamplings for the last 3 stages is a layer of conv with k3, s2 and p1
DOWNSAMPLE_LAYERS_FOUR_STAGES format: [Downsampling, Downsampling, Downsampling, Downsampling]
use `partial` to specify some arguments
"""
DOWNSAMPLE_LAYERS_FOUR_STAGES = [partial(Downsampling,
            kernel_size=7, stride=4, padding=2,
            post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6)
            )] + \
            [partial(Downsampling,
                kernel_size=3, stride=2, padding=1, 
                pre_norm=partial(LayerNormGeneral, bias=False, eps=1e-6), pre_permute=True
            )]*3


class MetaFormer(nn.Module):
    r""" MetaFormer
        A PyTorch impl of : `MetaFormer Baselines for Vision`  -
          https://arxiv.org/abs/2210.13452

    Args:
        in_chans (int): Number of input image channels. Default: 3.
        num_classes (int): Number of classes for classification head. Default: 1000.
        depths (list or tuple): Number of blocks at each stage. Default: [2, 2, 6, 2].
        dims (int): Feature dimension at each stage. Default: [64, 128, 320, 512].
        downsample_layers: (list or tuple): Downsampling layers before each stage.
        token_mixers (list, tuple or token_fcn): Token mixer for each stage. Default: nn.Identity.
        mlps (list, tuple or mlp_fcn): Mlp for each stage. Default: Mlp.
        norm_layers (list, tuple or norm_fcn): Norm layers for each stage. Default: partial(LayerNormGeneral, eps=1e-6, bias=False).
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_dropout (float): dropout for MLP classifier. Default: 0.
        layer_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: None.
            None means not use the layer scale. Form: https://arxiv.org/abs/2103.17239.
        res_scale_init_values (list, tuple, float or None): Init value for Layer Scale. Default: [None, None, 1.0, 1.0].
            None means not use the layer scale. From: https://arxiv.org/abs/2110.09456.
        output_norm: norm before classifier head. Default: partial(nn.LayerNorm, eps=1e-6).
        head_fn: classification head. Default: nn.Linear.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 320, 512],
                 downsample_layers=DOWNSAMPLE_LAYERS_FOUR_STAGES,
                 token_mixers=nn.Identity,
                 mlps=Mlp,
                 norm_layers=partial(LayerNormWithoutBias, eps=1e-6), # partial(LayerNormGeneral, eps=1e-6, bias=False),
                 drop_path_rate=0.,
                 head_dropout=0.0, 
                 layer_scale_init_values=None,
                 res_scale_init_values=[None, None, 1.0, 1.0],
                 output_norm=partial(nn.LayerNorm, eps=1e-6), 
                 head_fn=nn.Linear,
                 **kwargs,
                 ):
        super().__init__()
        self.num_classes = num_classes

        if not isinstance(depths, (list, tuple)):
            depths = [depths] # it means the model has only one stage
        if not isinstance(dims, (list, tuple)):
            dims = [dims]

        num_stage = len(depths)
        self.num_stage = num_stage

        if not isinstance(downsample_layers, (list, tuple)):
            downsample_layers = [downsample_layers] * num_stage
        down_dims = [in_chans] + dims
        self.downsample_layers = nn.ModuleList(
            [downsample_layers[i](down_dims[i], down_dims[i+1]) for i in range(num_stage)]
        )
        
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlps, (list, tuple)):
            mlps = [mlps] * num_stage

        if not isinstance(norm_layers, (list, tuple)):
            norm_layers = [norm_layers] * num_stage
        
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        if not isinstance(layer_scale_init_values, (list, tuple)):
            layer_scale_init_values = [layer_scale_init_values] * num_stage
        if not isinstance(res_scale_init_values, (list, tuple)):
            res_scale_init_values = [res_scale_init_values] * num_stage

        self.stages = nn.ModuleList() # each stage consists of multiple metaformer blocks
        cur = 0
        for i in range(num_stage):
            stage = nn.Sequential(
                *[MetaFormerBlock(dim=dims[i],
                token_mixer=token_mixers[i],
                mlp=mlps[i],
                norm_layer=norm_layers[i],
                drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_values[i],
                res_scale_init_value=res_scale_init_values[i],
                ) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = output_norm(dims[-1])

        if head_dropout > 0.0:
            self.head = head_fn(dims[-1], num_classes, head_dropout=head_dropout)
        else:
            self.head = head_fn(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'norm'}

    def forward_features(self, x):
        for i in range(self.num_stage):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([1, 2])) # (B, H, W, C) -> (B, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x



@register_model
def identityformer_s12(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_s12']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def identityformer_s24(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_s24']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def identityformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def identityformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def identityformer_m48(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=nn.Identity,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['identityformer_m48']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_s12(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_s12']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_s24(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_s24']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def randformer_m48(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=[nn.Identity, nn.Identity, RandomMixing, partial(RandomMixing, num_tokens=49)],
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['randformer_m48']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model



@register_model
def poolformerv2_s12(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_s12']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_s24(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[4, 4, 12, 4],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_s24']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[64, 128, 320, 512],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[6, 6, 18, 6],
        dims=[96, 192, 384, 768],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def poolformerv2_m48(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[8, 8, 24, 8],
        dims=[96, 192, 384, 768],
        token_mixers=Pooling,
        norm_layers=partial(LayerNormGeneral, normalized_dim=(1, 2, 3), eps=1e-6, bias=False),
        **kwargs)
    model.default_cfg = default_cfgs['poolformerv2_m48']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s18_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_s36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_s36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_m36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_m36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_m36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_m36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_m36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def convformer_b36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=SepConv,
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['convformer_b36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s18_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 3, 9, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s18_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_s36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[64, 128, 320, 512],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_s36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m36_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_m364_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[96, 192, 384, 576],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_m364_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36_384(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_384']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36_384_in21ft1k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_384_in21ft1k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model


@register_model
def caformer_b36_in21k(pretrained=False, **kwargs):
    model = MetaFormer(
        depths=[3, 12, 18, 3],
        dims=[128, 256, 512, 768],
        token_mixers=[SepConv, SepConv, Attention, Attention],
        head_fn=MlpHead,
        **kwargs)
    model.default_cfg = default_cfgs['caformer_b36_in21k']
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            url= model.default_cfg['url'], map_location="cpu", check_hash=True)
        model.load_state_dict(state_dict)
    return model

```


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
from dataset import EEGDataset, EEGDataset_Batch_normal
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

def train_identityformer_model(model, model_name, num_epochs=100, num_classes=3, batch_size=16, learning_rate=0.0001, w_wight=2560, chennal=32):
    # Checking CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    m = nn.Softmax(dim=1)  # 只对样本的维度做softmax
    # Creating datasets and data loaders
    train_dataset = EEGDataset(csv_file='train_data.csv', transform=transform)
    test_dataset = EEGDataset(csv_file='test_data.csv', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    save_dir = os.path.join('model', model_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0
    best_model_path = os.path.join(save_dir, "{}_best_model.pth".format(model_name))

    # 训练
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    for epoch in range(num_epochs):
        train_loss_total = 0  # 所有batch的loss累加值
        train_acc_total = 0   # 所有batch的acc累加值`
        val_loss_total = 0
        val_acc_total = 0

        model.train()    # 标志模型的模式是什么，因为dropout只在训练时启用
        for i, (train_x, train_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            train_x = train_x.unsqueeze(1)
            train_x = train_x.view(batch_size, 1, chennal, w_wight)
            # 前向传播
            train_y_pred = model(train_x)
            train_loss = loss_fn(train_y_pred, train_y)

            # 通过模型每个样本得到3个实数值（train_y_pred）,通过softmax将实数值转换成概率值，通过max取概率最大的下标，最后用下标和标签做比较
            train_acc = (m(train_y_pred).max(dim=1)[1] == train_y).sum()/train_y.shape[0]
            train_loss_total += train_loss.data.item()
            train_acc_total += train_acc.data.item()
            # 反向传播
            train_loss.backward()
            # 梯度下降
            optimizer.step()
            optimizer.zero_grad()

            # print("epoch:{} train_loss:{} train_acc:{}".format(epoch, train_loss.data.item(), train_acc.data.item()))

        train_loss_arr.append(train_loss_total / len(train_loader))  # 平均值
        train_acc_arr.append(train_acc_total / len(train_loader))
        print("epoch:{} train_loss:{} train_acc:{}".format(epoch, train_loss_arr[-1], train_acc_arr[-1]))
        # 测试集
        model.eval()
        for j, (val_x, val_y) in enumerate(test_loader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_x = val_x.unsqueeze(1)
            val_x = val_x.view(batch_size, 1, chennal, w_wight)
            # 前向传播
            val_y_pred = model(val_x)
            val_loss = loss_fn(val_y_pred, val_y)
            val_acc = (m(val_y_pred).max(dim=1)[1] == val_y).sum()/val_y.shape[0]
            val_loss_total += val_loss.data.item()
            val_acc_total += val_acc.data.item()

        val_loss_arr.append(val_loss_total / len(test_loader)) # 平均值
        val_acc_arr.append(val_acc_total / len(test_loader))
        print("epoch:{} val_loss:{} val_acc:{}".format(epoch, val_loss_arr[-1], val_acc_arr[-1]))

        # 保存最佳模型
        if val_acc_arr[-1] > best_val_acc:
            best_val_acc = val_acc_arr[-1]
            torch.save(model.state_dict(), best_model_path)

        # 保存训练和验证过程中的loss和acc
    np.save(os.path.join(save_dir, f"{model_name}_train_loss.npy"), np.array(train_loss_arr))
    np.save(os.path.join(save_dir, f"{model_name}_train_acc.npy"), np.array(train_acc_arr))
    np.save(os.path.join(save_dir, f"{model_name}_val_loss.npy"), np.array(val_loss_arr))
    np.save(os.path.join(save_dir, f"{model_name}_val_acc.npy"), np.array(val_acc_arr))

    print("保存模型成功!")
    print('Training completed!')

```

###  运行代码
```

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import EEGDataset, transform1
from net import IntegratedNet
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt



def train_identityformer_model(model, model_name, num_epochs=100, num_classes=3, batch_size=4, learning_rate=0.00001, w_wight=2560, chennal=32):
    # Checking CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    m = nn.Softmax(dim=1)  # 只对样本的维度做softmax
    # Creating datasets and data loaders
    train_dataset = EEGDataset(csv_file='train_data.csv', transform=transform1)
    test_dataset = EEGDataset(csv_file='test_data.csv', transform=transform1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    save_dir = os.path.join('model', model_name)
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0
    best_model_path = os.path.join(save_dir, "{}_best_model.pth".format(model_name))

    # 训练
    train_loss_arr = []
    train_acc_arr = []
    val_loss_arr = []
    val_acc_arr = []

    for epoch in range(num_epochs):
        train_loss_total = 0  # 所有batch的loss累加值
        train_acc_total = 0   # 所有batch的acc累加值`
        val_loss_total = 0
        val_acc_total = 0

        model.train()    # 标志模型的模式是什么，因为dropout只在训练时启用
        for i, (train_x, train_y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            train_x = train_x.to(device)
            # print('train_x',train_x.shape)
            train_y = train_y.to(device)
            train_x = train_x.unsqueeze(1)
            train_x = train_x.view(batch_size, 5, chennal, w_wight)
            # 前向传播
            train_y_pred = model(train_x)
            train_loss = loss_fn(train_y_pred, train_y)

            # 通过模型每个样本得到4个实数值（train_y_pred）,通过softmax将实数值转换成概率值，通过max取概率最大的下标，最后用下标和标签做比较
            train_acc = (m(train_y_pred).max(dim=1)[1] == train_y).sum()/train_y.shape[0]
            train_loss_total += train_loss.data.item()
            train_acc_total += train_acc.data.item()
            # 反向传播
            train_loss.backward()
            # 梯度下降
            optimizer.step()
            optimizer.zero_grad()

            # print("epoch:{} train_loss:{} train_acc:{}".format(epoch, train_loss.data.item(), train_acc.data.item()))

        train_loss_arr.append(train_loss_total / len(train_loader))  # 平均值
        train_acc_arr.append(train_acc_total / len(train_loader))
        print("epoch:{} train_loss:{} train_acc:{}".format(epoch, train_loss_arr[-1], train_acc_arr[-1]))
        # 测试集
        model.eval()
        for j, (val_x, val_y) in enumerate(test_loader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_x = val_x.unsqueeze(1)
            val_x = val_x.view(batch_size, 5, chennal, w_wight)
            # 前向传播
            val_y_pred = model(val_x)
            val_loss = loss_fn(val_y_pred, val_y)
            val_acc = (m(val_y_pred).max(dim=1)[1] == val_y).sum()/val_y.shape[0]
            val_loss_total += val_loss.data.item()
            val_acc_total += val_acc.data.item()

        val_loss_arr.append(val_loss_total / len(test_loader)) # 平均值
        val_acc_arr.append(val_acc_total / len(test_loader))
        print("epoch:{} val_loss:{} val_acc:{}".format(epoch, val_loss_arr[-1], val_acc_arr[-1]))

        # 保存最佳模型
        if val_acc_arr[-1] > best_val_acc:
            best_val_acc = val_acc_arr[-1]
            torch.save(model.state_dict(), best_model_path)

        # 保存训练和验证过程中的loss和acc
    np.save(os.path.join(save_dir, f"{model_name}_train_loss.npy"), np.array(train_loss_arr))
    np.save(os.path.join(save_dir, f"{model_name}_train_acc.npy"), np.array(train_acc_arr))
    np.save(os.path.join(save_dir, f"{model_name}_val_loss.npy"), np.array(val_loss_arr))
    np.save(os.path.join(save_dir, f"{model_name}_val_acc.npy"), np.array(val_acc_arr))

    print("保存模型成功!")
    print('Training completed!')

# 创建模型并训练
model = IntegratedNet(input_size=5,in_feature=29)  # 确保模型的输出层适用于三分类问题
train_identityformer_model(model, model_name='MLPFormer_betch_16_fft_opendata',chennal=19,w_wight=453)

```