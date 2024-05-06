---
title: 填坑——强化学习——使用智能体来玩游戏
date: 2024-04-30 19:28:01
tags: 填坑
---

## 参考文献

[flappy-bird-gymnasium环境](https://github.com/markub3327/flappy-bird-gymnasium)
[github上的代码](https://github.com/luozhiyun993/FlappyBird-PPO-pytorch)
[动手强化学习](https://hrl.boyuai.com/chapter/)
[tensorflow实现的强化学习](https://github.com/alexzhuustc/gym-flappybird)
[tensorflow+gym实现的flyappybird](https://github.com/alexzhuustc/gym-flappybird)
## 开篇
（广义的讲）强化学习是机器通过与环境交互来实现目标的一种计算方法。机器和环境的一轮交互是指，机器在环境的一个状态下做一个动作决策，把这个动作作用到环境当中，这个环境发生相应的改变并且将相应的奖励反馈和下一轮状态传回机器。这种交互是迭代进行的，机器的目标是最大化在多轮交互过程中获得的累积奖励的期望。强化学习用智能体（agent）这个概念来表示做决策的机器。相比于有监督学习中的“模型”，强化学习中的“智能体”强调机器不但可以感知周围的环境信息，还可以通过做决策来直接改变这个环境，而不只是给出一些预测信号。
在每一轮交互中，智能体感知到环境目前所处的状态，经过自身的计算给出本轮的动作，将其作用到环境中；环境得到智能体的动作后，产生相应的即时奖励信号并发生相应的状态转移。智能体则在下一轮交互中感知到新的环境状态，依次类推。
在这个过程中，智能体有3种控制要素、即感知、决策和奖励
感知：智能体在某种程度上感知环境的状态，从而知道自己所处的现状。
决策： 智能体根据感知的现状计算出达到目标需要采取的动作的过程叫做决策。比如，针对当前棋盘决定下一颗落子的位置。
奖励： 环境根据状态和智能体采取的动作，产生一个标量信号作为奖励反馈。这个标量信号衡量智能体的好坏。（类似于在深度学习中的损失函数，）

## 游戏选择 FlappyBird
flappy bird》是一款由来自越南的独立游戏开发者Dong Nguyen所开发的作品，游戏于2013年5月24日上线，并在2014年2月突然暴红。2014年2月，《Flappy Bird》被开发者本人从苹果及谷歌应用商店（Google Play）撤下。2014年8月份正式回归App Store，正式加入Flappy迷们期待已久的多人对战模式。游戏中玩家必须控制一只小鸟，跨越由各种不同长度水管所组成的障碍。

## 环境配置
1. 使用miniconda 配置环境，命令`conda create -n rl_python python=3.10`，配置环境，如果配置错误使用`conda remove rl_python --all `来删除环境，使用`conda activate rl_python`来启动环境。
2. 配置游戏环境，使用`pip install flappy-bird-gymnasium`安装游戏
3. 安装tersorflow深度学习框架[官网连接](https://tensorflow.google.cn/install/gpu?hl=zh-cn)
4. 安装pytorch-GPU环境 [官网链接](https://pytorch.org/get-started/previous-versions/) 通过 nvcc --version 来查看CUDA版本。安装10.2版本的pytorch`pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102` python 的3.10不兼容，换11.8的pytorch版本`pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118`

## ppo算法
PPO（Proximal Policy Optimization）近端策略优化算法，是一种基于策略（policy-based）的强化学习算法，是一种off-policy算法。由OpenAI于2017年提出，主要用于解决强化学习中的策略优化问题。它是Trust Region Policy Optimization（TRPO）的简化版本，旨在保持TRPO的优点，同时降低其计算复杂性。
核心原理：PPO的核心在于限制策略更新的步长，确保新策略不会偏离旧策略太远。这通过引入一个剪辑的目标函数来实现，该函数可以最小化策略更新过程中的风险。PPO算法通过这种方式平衡了探索与利用，提高了算法的稳定性和效率。PPO算法的核心在于更新策略梯度，主流方法有两种，分别是KL散度做penalty，另一种是Clip剪裁，它们的主要作用都是限制策略梯度更新的幅度，从而推导出不同的神经网络参数更新方式


## 代码解析工程

下面这份代码实现了使用ppo算法训练模型来玩Flappy Bird游戏。
get_args()函数用于解析命令行参数，包括学习率、折扣因子、迭代次数等。
PolicyNet类定义了策略网络，用于输出动作的概率分布。
ValueNet类定义了值函数网络，用于评估状态的价值。
compute_advantage()函数计算优势值，用于策略更新。
train()函数是训练的主要逻辑，包括初始化网络和优化器，采样游戏轨迹，计算优势值，更新策略网络和值函数网络等过程。
在训练过程中，使用TensorBoard记录训练过程中的奖励和其他指标。
在每轮训练结束后，保存表现最好的模型。
if __name__ == "__main__":部分用于执行整个训练过程。

```

import argparse # 用于解析命令行参数，方便地从命令行中读取参数值
import os 
from torch.utils.data import   BatchSampler, SubsetRandomSampler # BatchSampler 用于批次采样，subsetRandomSampler用于随机采样子集
import torch
import torch.nn as nn 
from torch.utils.tensorboard import SummaryWriter # 提供了与TensorBoard集成的功能，用于可视化训练过程和结果。

from src.flappy_bird import FlappyBird 

def get_args():
    # 该函数用于解析命令行参数，包括学习率、折扣因子、迭代次数
    parser = argparse.ArgumentParser(
        """Implementation of PPO to play Flappy Bird""")
    parser.add_argument("--lr", type=float, default=1e-5) # 学习率
    parser.add_argument("--gamma", type=float, default=0.99) # 折扣因子
    parser.add_argument("--num_iters", type=int, default=20000) # 迭代次数
    parser.add_argument("--log_path", type=str, default="tensorboard_ppo") # 保存日志
    parser.add_argument("--saved_path", type=str, default="trained_models") # 模型保存路径
    parser.add_argument("--lmbda", type=float, default=0.95) # lmbda 参数
    parser.add_argument("--epochs", type=int, default=10) # 训练轮数
    parser.add_argument("--eps", type=float, default=0.2) # eps ppo算法中的参数，默认值为0.2
    parser.add_argument("--batch_size",type=int, default=2048 ) # 批处理大小，默认值为2048
    parser.add_argument("--mini_batch_size",type=int, default=64 ) # 小批量大小

    args = parser.parse_args()
    return args


class PolicyNet(nn.Module):
    # 定义策略网络，用于输出动作的概率分布，表演者。
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU())
        self.flat = nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.Tanh())
        self.drop = nn.Dropout(0.5)
        self.fc3 = nn.Sequential(nn.Linear(512, 2))

    def forward(self, input):
        # 前向传播
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.flat(output)
        output = self.drop(output)
        output = self.fc1(output)
        return nn.functional.softmax(self.fc3(output), dim=1)


class ValueNet(nn.Module):
    # 定义值函数网络，用于评估状态的价值，评论家
    def __init__(self):
        super(ValueNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 512), nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, input):
        return self.net(input)

def compute_advantage(gamma, lmbda, td_delta):
    # 函数计算优势值，用于策略更新
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

def train(opt):
    # 训练的主要逻辑，包括初始化网络和优化器，采样游戏轨迹，计算优势值，更新策略网络和值函数网络等过程。
    if torch.cuda.is_available(): # 检查CUDA是否可以使用，
        torch.cuda.manual_seed(1993) # 设计随机种子，确保实验的可用性。
    else:
        torch.manual_seed(123)

    actor = PolicyNet().cuda() # 初始化策略函数，演员
    critic = ValueNet().cuda() # 初始化值函数，评论家

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=opt.lr) # 定义策略函数网络的优化器器
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=opt.lr) # 定义值函数网络的优化器


    writer = SummaryWriter(opt.log_path) # 创建一个TensorBoard的SummaryWriter对象，用于记录训练过程中的指标
    game_state = FlappyBird("ppo") # 
    state, reward, terminal = game_state.step(0)
    max_reward = 0
    iter = 0
    replay_memory = [] # 初始化回放内存，用于存储游戏轨迹
    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = [] # 初始化评估次数和评估列表，
    while iter < opt.num_iters:
        terminal = False
        episode_return = 0.0

        while not terminal:
            prediction = actor(state)
            print(prediction)
            action_dist = torch.distributions.Categorical(prediction)
            action_sample = action_dist.sample()
            action = action_sample.item()
            next_state, reward, terminal = game_state.step(action) # 执行游戏环境的一步，获取下一个状态、奖励和终止状态。
            replay_memory.append([state, action, reward, next_state, terminal])
            state = next_state
            episode_return += reward

            if len(replay_memory) > opt.batch_size:
                state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*replay_memory)
                states = torch.cat(state_batch, dim=0).cuda()
                actions = torch.tensor(action_batch).view(-1, 1).cuda()
                rewards = torch.tensor(reward_batch).view(-1, 1).cuda()
                dones = torch.tensor(terminal_batch).view(-1, 1).int().cuda()
                next_states = torch.cat(next_state_batch, dim=0).cuda()

                with torch.no_grad():
                    td_target = rewards + opt.gamma * critic(next_states) * (1 - dones)
                    td_delta = td_target - critic(states)
                    advantage = compute_advantage(opt.gamma, opt.lmbda, td_delta.cpu()).cuda()
                    old_log_probs = torch.log(actor(states).gather(1, actions)).detach()

                for _ in range(opt.epochs):
                    for index in BatchSampler(SubsetRandomSampler(range(opt.batch_size)), opt.mini_batch_size, False):
                        log_probs = torch.log(actor(states[index]).gather(1, actions[index]))
                        ratio = torch.exp(log_probs - old_log_probs[index])
                        surr1 = ratio * advantage[index]
                        surr2 = torch.clamp(ratio, 1 - opt.eps, 1 + opt.eps) * advantage[index]  # 截断
                        actor_loss = torch.mean(-torch.min(surr1, surr2))
                        critic_loss = torch.mean(
                            nn.functional.mse_loss(critic(states[index]), td_target[index].detach()))
                        actor_optimizer.zero_grad()
                        critic_optimizer.zero_grad()
                        actor_loss.backward()
                        critic_loss.backward()
                        actor_optimizer.step()
                        critic_optimizer.step()
                replay_memory = []

        if episode_return > max_reward:
            max_reward = episode_return
            print(" max_reward Iteration: {}/{}, Reward: {}".format(iter + 1, opt.num_iters, episode_return))

        iter += 1
        if (iter+1) % 10 == 0:
            evaluate_num += 1
            evaluate_rewards.append(episode_return)
            print("evaluate_num:{} \t episode_return:{} \t".format(evaluate_num, episode_return))
            writer.add_scalar('step_rewards', evaluate_rewards[-1], global_step= iter)
        if (iter+1) % 1000 == 0:
            actor_dict = {"net": actor.state_dict(), "optimizer": actor_optimizer.state_dict()}
            critic_dict = {"net": critic.state_dict(), "optimizer": critic_optimizer.state_dict()}
            torch.save(actor_dict, "{}/flappy_bird_actor_good".format(opt.saved_path))
            torch.save(critic_dict, "{}/flappy_bird_critic_good".format(opt.saved_path))



if __name__ == "__main__":
    opt = get_args()
    train(opt)


```
问题是上面代码使用了自定义的Flappy Bird来实现游戏环境，没有使用flappy-bird-gymnasium环境来创建游戏环境，具有一定的参考意义。注： 代码暂时不能移植到Flappy Bird，上我需要解析上面的代码的逻辑。

###  移植工程。
#### 第一步，明白怎么调用flappy_Bird环境
