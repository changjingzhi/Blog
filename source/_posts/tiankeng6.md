---
title: 填坑——强化学习——使用智能体来玩游戏
date: 2024-04-30 19:28:01
tags: 填坑
---

## 参考文献

[flappy-bird-gymnasium环境](https://github.com/markub3327/flappy-bird-gymnasium)
[Gymnasim官网](https://gymnasium.farama.org/)
[github上的代码](https://github.com/luozhiyun993/FlappyBird-PPO-pytorch)
[动手强化学习](https://hrl.boyuai.com/chapter/)
[tensorflow实现的强化学习](https://github.com/alexzhuustc/gym-flappybird)
[tensorflow+gym实现的flyappybird](https://github.com/alexzhuustc/gym-flappybird)
[问答网站](https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym)
## 开篇
（广义的讲）强化学习是机器通过与环境交互来实现目标的一种计算方法。机器和环境的一轮交互是指，机器在环境的一个状态下做一个动作决策，把这个动作作用到环境当中，这个环境发生相应的改变并且将相应的奖励反馈和下一轮状态传回机器。这种交互是迭代进行的，机器的目标是最大化在多轮交互过程中获得的累积奖励的期望。强化学习用智能体（agent）这个概念来表示做决策的机器。相比于有监督学习中的“模型”，强化学习中的“智能体”强调机器不但可以感知周围的环境信息，还可以通过做决策来直接改变这个环境，而不只是给出一些预测信号。
在每一轮交互中，智能体感知到环境目前所处的状态，经过自身的计算给出本轮的动作，将其作用到环境中；环境得到智能体的动作后，产生相应的即时奖励信号并发生相应的状态转移。智能体则在下一轮交互中感知到新的环境状态，依次类推。
在这个过程中，智能体有3种控制要素、即感知、决策和奖励
感知：智能体在某种程度上感知环境的状态，从而知道自己所处的现状。
决策： 智能体根据感知的现状计算出达到目标需要采取的动作的过程叫做决策。比如，针对当前棋盘决定下一颗落子的位置。
奖励： 环境根据状态和智能体采取的动作，产生一个标量信号作为奖励反馈。这个标量信号衡量智能体的好坏。（类似于在深度学习中的损失函数，）
强化学习中模型和环境交互，对于模型的目的来讲，就是从环境中取得最大的奖励值，主要在于策略的更新方法。
参数更新方法：价值更新、梯度更新。输出的值，连续的值，离散的值。


目前已经有的算法; 
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
```
import flappy_bird_gymnasium
import gymnasium as gym
env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=True)

obs, _ = env.reset() #  reset() 用于重置环境并返回初始观测值
for i in range(1):
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample() 
    print(action)
    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    # obs: 这是一个变量，用于存储执行动作后的观测值（observation）。在强化学习中，代理根据观测值来决定下一步的动作。
    # reward: 这是一个变量，用于存储执行动作后获得的奖励值（reward）。奖励值表示环境对代理执行动作的评价，可以是正数、负数或零。
    # terminated: 这是一个布尔变量，用于表示游戏是否结束（terminated）。如果游戏结束，则该变量为 True；否则为 False。
    # _: 下划线是一个通用的占位符，通常用于表示我们不关心的变量。在这里，它被用作占位符，因为 env.step(action) 返回的是一个元组，我们只关心其中的前三个元素（obs、reward 和 terminated），而不关心其他的返回值。
    # info: 这是一个字典，用于存储额外的环境信息（info）。这些信息可能包括调试信息、性能指标等，可以帮助我们更好地理解环境的运行情况。
    print(env.step(action))
    # Checking if the player is still alive
    if terminated:
        break

env.close()
```

在github上，作者表明 在论文中有更详细的参考：[论文链接](https://www.mdpi.com/1424-8220/24/6/1905)

1. 状态空间：
在"FlappyBird-v0"环境中，提供了代表游戏屏幕的观测数据，这些数据为游戏状态提供了简单的数值信息。

FlappyBird-v0
存在两种观测选项：

选项一
激光雷达传感器的180个读数（论文：使用变压器模型和激光雷达传感器进行运动识别玩Flappy Bird）
选项二
最后一个管道的水平位置
最后一个顶部管道的垂直位置
最后一个底部管道的垂直位置
下一个管道的水平位置
下一个顶部管道的垂直位置
下一个底部管道的垂直位置
下下一个管道的水平位置
下下一个顶部管道的垂直位置
下下一个底部管道的垂直位置
玩家的垂直位置
玩家的垂直速度
玩家的旋转

2. 动作空间：
0 - 什么都不做
1 - 拍打翅膀

3. 奖励：
+0.1 - 每帧保持存活状态
+1.0 - 成功通过一根管道
-1.0 - 死亡
−0.5 - 触摸屏幕顶部


## 第二步，使用一个简单的游戏来实践dqn

上代码：实践了打砖块的游戏，游戏输入为（4，80，80）图片架构，

rl_utils代码
```
from tqdm import tqdm
import numpy as np
import torch
import collections
import random

class ReplayBuffer: # 记忆池子
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                

```
dqn_breakout 文件包括dqn网络的构建，模型的训练[介绍的链接](https://gymnasium.farama.org/environments/atari/breakout/)，

```
import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
    
class Qnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):
        super(Qnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4) # 卷积层，输入
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        self.head = torch.nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.head(x)

class DQN:
    ''' DQN算法 '''
    def __init__(self, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(in_channels=4, action_dim=action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def save(self, model_file="mymodel"):
        torch.save(self.q_net, model_file)

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

lr = 2e-3
num_episodes = 500
hidden_dim = 128
gamma = 0.98
epsilon = 0.01
target_update = 10
buffer_size = 100000
minimal_size = 500
batch_size = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = 'BreakoutNoFrameskip-v4' # 雅达利的弹球游戏
env = gym.make(env_name)
env = gym.wrappers.AtariPreprocessing(env)
env = gym.wrappers.FrameStack(env, num_stack=4)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(buffer_size)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
print('state shape:', env.observation_space.shape)
agent = DQN(action_dim, lr, gamma, epsilon,target_update, device)

return_list = []
for i in range(10):
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):
            episode_return = 0
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminate, truncated, _ = env.step(action)
                done = terminate or truncated
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > minimal_size:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)

agent.save()
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

```

## 第三步解析上面实现的弹球游戏

打砖块的游戏的输入为(4, 84, 84)，动作有4个有四个动作( 0 , 1 , 2 , 3 )，奖励为分数。对于Flappy Bird for Gymnasium环境中游戏action 为（180，）The LIDAR sensor 180 readings， action space 为 0 do nothing， 1 为 flap 奖励为0.1 - every frame it stays alive，1.0 - successfully passing a pipe，1.0 - dying， 0.5 - touch the top of the screen 输入有点少。

## 第四步，使用dqn网络 

笔记： 函数的方法（DQN）和基于策略的方法（REINFORCE），其中基于值函数的方法只学习一个价值函数，而基于策略的方法只学习一个策略函数。那么，一个很自然的问题是，有没有什么方法既学习价值函数，又学习策略函数呢？答案就是 Actor-Critic。在 REINFORCE 算法中，目标函数的梯度中有一项轨迹回报，用于指导策略的更新。Actor 的更新采用策略梯度的原则，之前路走偏了使用了价值策略来更新参数，不容易收敛。

model
```
import torch
import torch.nn.functional as F

class ACnet(torch.nn.Module):
    ''' 加入卷积层的Q网络 '''
    def __init__(self, action_dim, in_channels=4):
        super(ACnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = torch.nn.Flatten()
        self.fc4 = torch.nn.Linear(7 * 7 * 64, 512)
        #Actor                
        self.actor_fc = torch.nn.Linear(512, action_dim)
        #Critic
        self.policy_fc = torch.nn.Linear(512, 1)
    
    def policy(self, obs):
        """
        Args:
            obs: A float32 tensor array of shape [B, C, H, W]

        Returns:
            policy_logits: B * ACT_DIM
        """
        obs = obs / 255.0
        conv1 = F.relu(self.conv1(obs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        flatten = self.flatten(conv3)
        fc_output = F.relu(self.fc(flatten))
        policy_logits = self.policy_fc(fc_output)

        return policy_logits

    def value(self, obs):
        """
        Args:
            obs: A float32 tensor of shape [B, C, H, W]

        Returns:
            values: B
        """
        obs = obs / 255.0
        conv1 = F.relu(self.conv1(obs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        flatten = self.flatten(conv3)
        fc_output = F.relu(self.fc(flatten))
        values = self.value_fc(fc_output)
        values = torch.squeeze(values, axis=1)
        return values

    def policy_and_value(self, obs):
        """
        Args:
            obs: A tensor array of shape [B, C, H, W]

        Returns:
            policy_logits: B * ACT_DIM
            values: B
        """
        obs = obs / 255.0
        conv1 = F.relu(self.conv1(obs))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        flatten = self.flatten(conv3)
        fc_output = F.relu(self.fc(flatten))

        policy_logits = self.policy_fc(fc_output)

        values = self.value_fc(fc_output)
        values = torch.squeeze(values, axis=1)
        return policy_logits, values
    

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.softmax(self.fc4(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


```

agent
```
import torch
import os
import torch.nn.functional as F
from model import PolicyNet, ValueNet

class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, device):
        # 策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        
                # 策略网络优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # 价值网络优化器
        #学习率衰减
        # self.actor_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.99)
        # self.critic_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.99)
        self.actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.actor_optimizer, T_max=50, eta_min=1e-6)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.critic_optimizer, T_max=50, eta_min=1e-6)
        self.gamma = gamma
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # 时序差分目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 -
                                                                       dones)
        td_delta = td_target - self.critic(states)  # 时序差分误差
        log_probs = torch.log(self.actor(states).gather(1, actions))
        actor_loss = torch.mean(-log_probs * td_delta.detach())
        # 均方误差损失函数
        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()  # 计算策略网络的梯度
        critic_loss.backward()  # 计算价值网络的梯度
        self.actor_optimizer.step()  # 更新策略网络的参数
        self.critic_optimizer.step()  # 更新价值网络的参数
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

    def save(self, path='mymodel', epoch=0):
        '''保存模型'''
        checkpoint = {"actor": self.actor.state_dict(),
                      "critic": self.critic.state_dict(),
                      "actor_opt": self.actor_optimizer.state_dict(),
                      "critic_opt": self.critic_optimizer.state_dict(),
                      "epoch": epoch}        
        if not os.path.exists(path):
            os.makedirs(path, True)
        path = os.path.join(path, f'checkpoint_{epoch}.ckpt')
        torch.save(checkpoint, path)

    def load(self, path='mymodel'):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
        return checkpoint['epoch']

```

rl_utils 文件，用于测试代码

```
from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import time
from tensorboardX import SummaryWriter

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) 

    def add(self, state, action, reward, next_state, done): 
        self.buffer.append((state, action, reward, next_state, done)) 

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    current_time = time.strftime("logs/%Y-%m-%dT%H_%M", time.localtime())
    writer = SummaryWriter(log_dir=current_time)
    return_list = []
    global_step = 0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state,_ = env.reset()
                # print(state)
                done = False
                while not done:
                    action = agent.take_action(state)
                    # print(action)
                    next_state, reward, terminated,truncated, _ = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                
                return_list.append(episode_return)
                writer.add_scalar('reward', episode_return, global_step)
                writer.add_scalar('actor LR', agent.actor_lr_scheduler.get_last_lr(),global_step)
                writer.add_scalar('critic LR', agent.critic_lr_scheduler.get_last_lr(),global_step)
                global_step += 1
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
        agent.save(path='mymodel', epoch=i)

    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                
```


训练代码，使用Ac—Critic算法来进行梯度更新
```
import flappy_bird_gymnasium
import gymnasium as gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from model import ACnet
from agent import ActorCritic



actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 4000
hidden_dim = 256
gamma = 0.96
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'FlappyBird-v0'
env = gym.make(env_name)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                    gamma, device)

#恢复训练
# agent.load('mymodel/checkpoint_9')

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)

agent.save()


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Actor-Critic on {}'.format(env_name))
plt.show()

```
结果不怎么样，一直维持到0.8 左右。agent一直没有行动。某种程度上是数据输入太少了，论文中的结果怎么好我看不懂，真的服了。


## 补充知识


价值方法： Sarsa、Qlearning、DQN
策略方法： Reinfore、Actor-Critic、A2C、TRPO、PPO、DDPC


on-policy（在策略） 的算法： Sarsa、Reinforce、Actor-Critic、A2C、TRPO、PPO
非on-policy算法：Qlearning，DQN，Double DQN
两者区别在于： on-policy是在线训练的，采样做预测做动作的思路。非on-policy离线训练。对于on-policy算法，采样先预测然后计算概率来采取动作。（on-policy是激进的，非on-policy是保守的。）


## 挖坑
### 怎么使用tensorboar来进行可视化过程。

TensorBoard是一个可视化工具，它可以用来展示网络图、张量的指标变化、张量的分布情况等。特别是在训练网络的时候，我们可以设置不同的参数（比如：权重W、偏置B、卷积层数、全连接层数等），使用TensorBoader可以很直观的帮我们进行参数的选择。它通过运行一个本地服务器，来监听6006端口。在浏览器发出请求时，分析训练时记录的数据，绘制训练过程中的图像。
