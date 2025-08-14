---
title: '马尔可夫决策过程'
publishDate: '2025-08-14'
updatedDate: '2025-08-14'
description: '深入强化学习'
tags:
  - 强化学习
language: 'Chinese'
---

# 马尔可夫决策过程

## 1 马尔可夫过程

### 1.1 随机过程

**随机过程（stochastic process）** 是概率论的“动力学”部分。

概率论的研究对象是静态的==随机现象==（例如投掷硬币），而==随机过程==的研究对象是随时间演变的随机现象（例如天气随时间的变化、城市交通随时间的变化）。

> **随机现象**关注的是单次试验结果，其概率分布被认为是不随时间演变的。例如投掷硬币：即使你今天掷、明天掷、后天掷，概率规律本身不变。
> **随机过程**关注的不是单次试验结果，而是随机现象随时间演变的轨迹。不仅样本空间可能随时间变化，概率分布也可以随时间变化（例如天气随时间变化：早上只可能是晴天或雨天，晚上只可能是阴天或雨天；早上的下雨概率是 20%，晚上是 60%）。

在随机过程中，随机现象在某时刻的取值是一个向量随机变量，用 $\boldsymbol{S}_t$ 表示，所有可能的状态组成状态集合 $\mathcal{S}$。随机现象便是状态的变化过程。在某时刻 $t$ 的状态 $\boldsymbol{S}_t$ 通常取决于 $t$ 时刻之前的状态。我们将已知历史信息 $(\boldsymbol{S}_1,\cdots,\boldsymbol{S}_t)$ 时下一个时刻状态为 $\boldsymbol{S}_{t+1}$ 的概率表示成 $P(\boldsymbol{S}_{t+1}|\boldsymbol{S}_1,\cdots,\boldsymbol{S}_t)$。

> 例如运动的机器人在 $t$ 时刻的位置，就是 3 维**向量随机变量**，即 $\boldsymbol{S}_{t}=(X_t,Y_t,Z_t)$，其中 $X_t$、$Y_t$ 、$Z_t$ 分别是机器人在 $X$、 $Y$ 、$Z$ 三个方向上的随机变量。

### 1.2 马尔可夫性质

当且仅当某时刻的状态只取决于上一时刻的状态时，一个随机过程被称为具有**马尔可夫性质（Markov property）**，用公式表示为 $P(\boldsymbol{S}_{t+1}|\boldsymbol{S}_{t})=P(\boldsymbol{S}_{t+1}|\boldsymbol{S}_1,\cdots,\boldsymbol{S}_t)$。

也就是说，当前状态是未来的充分统计量，即下一个状态只取决于当前状态，而不会受到过去状态的影响。需要明确的是，具有马尔可夫性并不代表这个随机过程就和历史完全没有关系。因为虽然 $t+1$ 时刻的状态只与 $t$ 时刻的状态有关，但是 $t$ 时刻的状态其实包含了 $t-1$ 时刻的状态的信息，通过这种链式的关系，历史的信息被传递到了现在。

> 马尔可夫性可以大大简化运算，因为只要当前状态可知，所有的历史信息都不再需要了，利用当前状态信息就可以决定未来。

### 1.3 马尔可夫过程

**马尔可夫过程（Markov process）** 指具有马尔可夫性质的随机过程，也被称为**马尔可夫链（Markov chain）**。

我们通常用元组 $\langle \mathcal{S}, \boldsymbol{\mathcal{P}}\rangle$ 描述一个马尔可夫过程，其中 $\mathcal{S}$ 是有限数量的状态集合，$\boldsymbol{\mathcal{P}}$ 是**状态转移矩阵（state transition matrix）**。

假设一共有 $n$ 个状态，此时 $\mathcal{S}=\{\boldsymbol{s}_1,\boldsymbol{s}_2,\cdots,\boldsymbol{s}_n\}$。状态转移矩阵 $\boldsymbol{\mathcal{P}}$ 定义了所有状态对之间的转移概率，即：

$$
\boldsymbol{\mathcal{P}}=
\begin{bmatrix}
P(\boldsymbol{s}_1|\boldsymbol{s}_1) & \cdots & P(\boldsymbol{s}_n|\boldsymbol{s}_1) \\
\vdots & \ddots & \vdots \\
P(\boldsymbol{s}_1|\boldsymbol{s}_n) & \cdots & P(\boldsymbol{s}_n|\boldsymbol{s}_n) 
\end{bmatrix}
$$

矩阵 $\boldsymbol{\mathcal{P}}$ 中第 $i$ 行第 $j$ 列元素表示从状态 $\boldsymbol{s}_i$ 转移到状态 $\boldsymbol{s}_j$ 的概率，我们称 $P(\boldsymbol{s}'|\boldsymbol{s})$ 为**状态转移函数**。

> 从某个状态出发，到达其他状态的概率和必须为 1，即状态转移矩阵 $\boldsymbol{\mathcal{P}}$ 的每一行的和为 1。

下图是一个具有 6 个状态的马尔可夫过程的简单例子。

- 每个绿色圆圈表示一个状态，每个状态都有一定概率（包括概率为 0）转移到其他状态。
- 状态之间的虚线箭头表示状态的转移，箭头旁的数字表示该状态转移发生的概率。
- 从每个状态出发转移到其他状态的概率总和为 1。
 
其中 $\boldsymbol{s}_6$ 通常被称为**终止状态（terminal state）**，因为它不会再转移到其他状态，可以理解为它永远以概率 1 转移到自己。

![20250814091903-2025-08-14-09-19-03](https://ozzyc.oss-cn-shenzhen.aliyuncs.com/NotePicture/20250814091903-2025-08-14-09-19-03.png)

我们可以写出这个马尔可夫过程的状态转移矩阵：

$$
\boldsymbol{\mathcal{P}}=
\begin{bmatrix}
0.9 & 0.1 & 0 & 0 & 0 & 0 \\
0.5 & 0 & 0.5 & 0 & 0 & 0 \\
0 & 0 & 0 & 0.6 & 0 & 0.4 \\
0 & 0 & 0 & 0 & 0.3 & 0.7 \\
0 & 0.2 & 0.3 & 0.5 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1
\end{bmatrix}
$$

其中，第 $i$ 行 $j$ 列的值 $\boldsymbol{\mathcal{P}}_{i,j}$ 则代表从状态 $\boldsymbol{s}_i$ 转移到 $\boldsymbol{s}_j$ 的概率。

给定一个马尔可夫过程，我们就可以从某个状态出发，根据它的状态转移矩阵生成一个状态**序列（episode）**，这个步骤也被叫做**采样（sampling）**。例如，从 $\boldsymbol{s}_1$ 出发，可以生成序列 $\boldsymbol{s}_1 \rightarrow \boldsymbol{s}_2 \rightarrow \boldsymbol{s}_3 \rightarrow \boldsymbol{s}_6$ 或序列 $\boldsymbol{s}_1 \rightarrow \boldsymbol{s}_1 \rightarrow \boldsymbol{s}_2 \rightarrow \boldsymbol{s}_3 \rightarrow \boldsymbol{s}_4 \rightarrow \boldsymbol{s}_5 \rightarrow \boldsymbol{s}_3 \rightarrow \boldsymbol{s}_6$ 等。显然，生成这些序列的概率和状态转移矩阵有关。

## 2 马尔可夫奖励过程

在马尔可夫过程的基础上加入奖励函数 $r$ 和折扣因子 $\gamma$，就可以得到**马尔可夫奖励过程（Markov Reward Process，MRP）**。

一个马尔可夫奖励过程由 $\langle \mathcal{S}, \boldsymbol{\mathcal{P}}, r , \gamma \rangle$ 构成，各个组成元素的含义如下所示：

- $\mathcal{S}$ 是有限状态的集合。
- $\boldsymbol{\mathcal{P}}$ 是状态转移矩阵。
- $r$ 是==奖励函数==，某个状态 $\boldsymbol{s}$ 的奖励 $r(\boldsymbol{s})$ 指转移到该状态时可以获得奖励的期望。
- $\gamma$ 是==折扣因子（discount factor）==，取值范围为 $[0,1)$。引入折扣因子的理由为远期利益具有一定不确定性，有时我们更希望能够尽快获得一些奖励，所以我们需要对远期利益打一些折扣。接近 1 的 $\gamma$ 更关注长期的累计奖励，接近 0 的 $\gamma$ 更考虑短期奖励。

> **为什么奖励 $r$ 是“函数”而不是“数”？**
> 首先，转移到不同状态可能带来不同的即时奖励，因此需要表达为状态的函数 $r(\boldsymbol{s})$；其次，即使是转移到同一个状态 $\boldsymbol{s}$，获得的奖励也是随机的（表达为一个概率分布），因此用期望表达转移到状态 $\boldsymbol{s}$ 的平均奖励。例如每天从家步行通勤到公司，即使到同一个路口，也会因为天气/施工/红绿灯等原因获取不同的奖励。

### 2.1 回报

在一个马尔可夫奖励过程中，从第 $t$ 时刻状态 $\boldsymbol{S}_t$ 开始，直到终止状态时，所有奖励的衰减之和称为**回报（Return）**，公式如下：

$$
\begin{aligned}
G_t &= R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots \\
&=\sum_{k=0}^{\infty} \gamma^k R_{t+k}
\end{aligned}
$$

其中，$R_{t}$ 表示在时刻 $t$ 获得的奖励（这个奖励是转移到某个状态 $\boldsymbol{S}_t$ 获得的奖励期望）。

如下图，我们在刚才基础上添加奖励函数，构建成一个马尔可夫奖励过程。

![20250814114145-2025-08-14-11-41-45](https://ozzyc.oss-cn-shenzhen.aliyuncs.com/NotePicture/20250814114145-2025-08-14-11-41-45.png)

例如，进入状态 $\boldsymbol{s}_2$ 可以得到奖励期望为 -2，表明我们不希望进入 $\boldsymbol{s}_2$；进入 $\boldsymbol{s}_4$ 可以获得最高奖励期望 10，但是进入 $\boldsymbol{s}_6$ 之后奖励期望为 0，此时序列。

比如选取 $\boldsymbol{s}_1$ 为起始状态，设置 $\gamma=0.5$，采样到一条状态序列为 $\boldsymbol{s}_1 \rightarrow \boldsymbol{s}_2 \rightarrow \boldsymbol{s}_3 \rightarrow \boldsymbol{s}_6$，此时可以计算 $\boldsymbol{s}_1$ 的回报 $G_1$：

$$
G_1=-1 + 0.5 \times (-1) +0.5^2 \times (-2) + 0.5^3 \times 0 = -2.5
$$

> 直观来看，**回报**就是从当前时刻开始，==未来所有奖励的加权和==。在判断路径优劣时，我们可以通过比较不同路径的回报大小来选择长期收益更高的路线；回报越大，说明沿着这条路径获得的累计奖励越多，从而表明该路径在长期来看更有价值或更优。
