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

概率论的研究对象是静态的**随机现象**（例如投掷硬币），而**随机过程**的研究对象是随时间演变的随机现象（例如天气随时间的变化、城市交通随时间的变化）。

在随机过程中，随机现象在某时刻的取值是一个**向量随机变量**，用 $\boldsymbol{S}_t$ 表示，所有可能的状态组成状态集合 $\mathcal{S}$。例如运动的机器人在 $t$ 时刻的位置，就是 3 维向量随机变量，即 $\boldsymbol{S}_{t}=(X_t,Y_t,Z_t)$，其中 $X_t$、$Y_t$ 、$Z_t$ 分别是机器人在 $X$、 $Y$ 、$Z$ 三个方向上的随机变量。

随机现象是状态的变化过程。在某时刻 $t$ 的状态 $\boldsymbol{S}_t$ 通常取决于 $t$ 时刻之前的状态。我们将已知历史信息 $(\boldsymbol{S}_1,\cdots,\boldsymbol{S}_t)$ 时下一个时刻状态为 $\boldsymbol{S}_{t+1}$ 的概率表示成 $P(\boldsymbol{S}_{t+1}|\boldsymbol{S}_1,\cdots,\boldsymbol{S}_t)$。

> **随机现象**关注的是单次试验结果，其概率分布被认为是不随时间演变的。例如投掷硬币：即使你今天掷、明天掷、后天掷，概率规律本身不变。
> **随机过程**关注的不是单次试验结果，而是随机现象随时间演变的轨迹。不仅样本空间可能随时间变化，概率分布也可以随时间变化（例如天气随时间变化：早上只可能是晴天或雨天，晚上只可能是阴天或雨天；早上的下雨概率是 20%，晚上是 60%）。

### 1.2 马尔可夫性质

当且仅当某时刻的状态只取决于上一时刻的状态时，一个随机过程被称为具有**马尔可夫性质（Markov property）**，用公式表示为 $P(\boldsymbol{S}_{t+1}|\boldsymbol{S}_{t})=P(\boldsymbol{S}_{t+1}|\boldsymbol{S}_1,\cdots,\boldsymbol{S}_t)$。

也就是说，当前状态是未来的充分统计量，即下一个状态只取决于当前状态，而不会受到过去状态的影响。需要明确的是，具有马尔可夫性并不代表这个随机过程就和历史完全没有关系。因为虽然 $t+1$ 时刻的状态只与 $t$ 时刻的状态有关，但是 $t$ 时刻的状态其实包含了 $t-1$ 时刻的状态的信息，通过这种链式的关系，历史的信息被传递到了现在。

> 马尔可夫性可以大大简化运算，因为只要当前状态可知，所有的历史信息都不再需要了，利用当前状态信息就可以决定未来。

### 1.3 马尔可夫过程

**马尔可夫过程（Markov process）** 指具有马尔可夫性质的随机过程，也被称为**马尔可夫链（Markov chain）**。通常用元组 $\langle \mathcal{S}, \boldsymbol{\mathcal{P}}\rangle$ 描述一个马尔可夫过程，其中 $\mathcal{S}$ 是有限数量的状态集合，$\boldsymbol{\mathcal{P}}$ 是**状态转移矩阵（state transition matrix）**。

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

![![20250814091903-2025-08-14-09-19-03](httpsozzyc.oss-cn-shenzhen.aliyuncs.comNotePicture20250814091903-2025-08-14-09-19-03.png)-2025-08-15-09-41-24](https://ozzyc.oss-cn-shenzhen.aliyuncs.com/NotePicture/![20250814091903-2025-08-14-09-19-03](httpsozzyc.oss-cn-shenzhen.aliyuncs.comNotePicture20250814091903-2025-08-14-09-19-03.png)-2025-08-15-09-41-24.png)

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
- $r$ 是**奖励函数**，某个状态 $\boldsymbol{s}$ 的奖励 $r(\boldsymbol{s})$ 指转移到该状态时可以获得奖励的期望。
- $\gamma$ 是**折扣因子（discount factor）**，取值范围为 $[0,1)$。接近 1 的 $\gamma$ 更关注长期的累计奖励，接近 0 的 $\gamma$ 更考虑短期奖励。

> **为什么奖励 $r$ 是“函数”而不是“数”？**
> 首先，转移到不同状态可能带来不同的即时奖励，因此需要表达为状态的函数 $r(\boldsymbol{s})$；其次，即使是转移到同一个状态 $\boldsymbol{s}$，获得的奖励也是随机的（表达为一个概率分布）。例如每天从家步行通勤到公司，即使到同一个路口，也会因为天气/施工/红绿灯等原因得到不同的奖励。

### 2.1 回报

在一个马尔可夫奖励过程中，从第 $t$ 时刻状态 $\boldsymbol{S}_t$ 开始，直到终止状态时，所有奖励的累积称为**回报（Return）**，公式如下：

$$
\begin{aligned}
G_t &= R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots \\
&=\sum_{k=0}^{\infty} \gamma^k R_{t+k}
\end{aligned}
$$

其中，$R_{t}$ 表示在时刻 $t$ 获得的奖励。如下图，我们在刚才基础上添加奖励函数，构建成一个马尔可夫奖励过程。

![![20250814114145-2025-08-14-11-41-45](httpsozzyc.oss-cn-shenzhen.aliyuncs.comNotePicture20250814114145-2025-08-14-11-41-45.png)-2025-08-15-09-56-40](https://ozzyc.oss-cn-shenzhen.aliyuncs.com/NotePicture/![20250814114145-2025-08-14-11-41-45](httpsozzyc.oss-cn-shenzhen.aliyuncs.comNotePicture20250814114145-2025-08-14-11-41-45.png)-2025-08-15-09-56-40.png)

例如，进入状态 $\boldsymbol{s}_2$ 可以得到奖励期望为 -2，表明我们不希望进入 $\boldsymbol{s}_2$；进入 $\boldsymbol{s}_4$ 可以获得最高奖励期望 10，但是进入 $\boldsymbol{s}_6$ 之后奖励期望为 0，此时序列终止。比如选取 $\boldsymbol{s}_1$ 为起始状态，设置 $\gamma=0.5$，采样到一条状态序列为 $\boldsymbol{s}_1 \rightarrow \boldsymbol{s}_2 \rightarrow \boldsymbol{s}_3 \rightarrow \boldsymbol{s}_6$，此时可以计算 $\boldsymbol{s}_1$ 的回报 $G_1$：

$$
G_1=-1 + 0.5 \times (-1) +0.5^2 \times (-2) + 0.5^3 \times 0 = -2.5
$$

> 直观来看，**回报**就是从当前时刻开始，未来所有奖励的加权和。在判断路径优劣时，我们可以通过比较不同路径的回报大小来选择长期收益更高的路线；回报越大，说明沿着这条路径获得的累计奖励越多，从而表明该路径在长期来看更有价值或更优。

### 2.2 价值函数

在马尔可夫奖励过程中，一个状态的**期望回报**（即从这个状态出发的未来累积奖励的期望）被称为这个状态的**价值（value）**。所有状态的价值就组成了**价值函数（value function）**，价值函数的输入为某个状态，输出为这个状态的价值。我们将价值函数写成 $V(\boldsymbol{s})=\mathbb{E}[G_t | \boldsymbol{S}_t = \boldsymbol{s}]$，展开为：

$$
\begin{aligned}
V(\boldsymbol{s})&=\mathbb{E}[G_t | \boldsymbol{S}_t = \boldsymbol{s}] \\
&=\mathbb{E}[R_t + \gamma R_{t+1} + \gamma^2 R_{t+2} + \cdots | \boldsymbol{S}_t = \boldsymbol{s}] \\
&=\mathbb{E}[R_t + \gamma (R_{t+1} + \gamma R_{t+2} + \cdots) | \boldsymbol{S}_t = \boldsymbol{s}] \\
&=\mathbb{E}[R_t + \gamma G_{t+1} | \boldsymbol{S}_t = \boldsymbol{s}]\\
&=\mathbb{E}[R_t | \boldsymbol{S}_t = \boldsymbol{s}]+
\gamma \mathbb{E}[G_{t+1} | \boldsymbol{S}_t = \boldsymbol{s}]
\end{aligned}
$$

- $\mathbb{E}[R_t | \boldsymbol{S}_t = \boldsymbol{s}]$ 是转移到状态 $\boldsymbol{s}$ 获得的即时奖励期望：$r(\boldsymbol{s})$；
- $\mathbb{E}[G_{t+1} | \boldsymbol{S}_t = \boldsymbol{s}]$ 的含义是：在当前时刻 $t$ 处于状态 $\boldsymbol{s}$ 的前提下，从下一时刻 $t+1$ 开始直到终止的未来回报 $G_{t+1}$ 的期望值。因为我们只知道当前状态 $\boldsymbol{s}$，不知道下一步 $t+1$ 时刻会到达哪个状态，因此我们要根据从状态 $\boldsymbol{s}$ 出发的转移概率来求期望：

$$
\begin{aligned}
\mathbb{E}[G_{t+1} | \boldsymbol{S}_t = \boldsymbol{s}] 
&=\sum_{\boldsymbol{s}' \in \mathcal{S}}  
\mathbb{E}[G_{t+1} | \boldsymbol{S}_t = \boldsymbol{s},\boldsymbol{S}_{t+1} = \boldsymbol{s}'] \cdot P(\boldsymbol{s}'|\boldsymbol{s}) \\
&=\sum_{\boldsymbol{s}' \in \mathcal{S}}  
\mathbb{E}[G_{t+1} | \boldsymbol{S}_{t+1} = \boldsymbol{s}'] \cdot P(\boldsymbol{s}'|\boldsymbol{s}) \\
&=\sum_{\boldsymbol{s}' \in \mathcal{S}}V(\boldsymbol{s}')\cdot P(\boldsymbol{s}'|\boldsymbol{s})
\end{aligned}
$$

> 由于马尔可夫性质：未来的奖励仅依赖于当前状态，与先前的状态无关。因此有 $\mathbb{E}[G_{t+1} | \boldsymbol{S}_t = \boldsymbol{s},\boldsymbol{S}_{t+1} = \boldsymbol{s}']=\mathbb{E}[G_{t+1} | \boldsymbol{S}_{t+1} = \boldsymbol{s}']$。

### 2.3 贝尔曼方程

综上，我们可以推导出**贝尔曼方程（Bellman equation）**：

$$
V(\boldsymbol{s}) = r(\boldsymbol{s}) + \gamma \sum_{\boldsymbol{s}' \in \mathcal{S}}P(\boldsymbol{s}'|\boldsymbol{s})V(\boldsymbol{s}')
$$

若一个马尔可夫奖励过程一共有 $n$ 个状态，即 $\mathcal{S}=\{\boldsymbol{s}_1, \boldsymbol{s}_2 , \cdots , \boldsymbol{s}_n \}$，将所有状态的价值表示成一个列向量 $\boldsymbol{\mathcal{V}}=\begin{bmatrix}V(\boldsymbol{s}_1),V(\boldsymbol{s}_2),\cdots, V(\boldsymbol{s}_n)\end{bmatrix}^{\mathrm{T}}$，将奖励函数表示成一个列向量 $\boldsymbol{\mathcal{R}}=\begin{bmatrix}r(\boldsymbol{s}_1),r(\boldsymbol{s}_2),\cdots, r(\boldsymbol{s}_n)\end{bmatrix}^{\mathrm{T}}$。于是我们可以将贝尔曼方程写成矩阵的形式：

$$
\begin{aligned}
&\boldsymbol{\mathcal{V}}=\boldsymbol{\mathcal{R}}+\gamma \boldsymbol{\mathcal{P}} \boldsymbol{\mathcal{V}} \\
\begin{bmatrix}
V(\boldsymbol{s}_1)\\
V(\boldsymbol{s}_2)\\
\cdots \\
V(\boldsymbol{s}_n)
\end{bmatrix} =
\begin{bmatrix}
r(\boldsymbol{s}_1)\\
r(\boldsymbol{s}_2)\\
\cdots \\
r(\boldsymbol{s}_n)
\end{bmatrix}+\gamma
&\begin{bmatrix}
P(\boldsymbol{s}_1|\boldsymbol{s}_1) & \cdots & P(\boldsymbol{s}_n|\boldsymbol{s}_1) \\
\vdots & \ddots & \vdots \\
P(\boldsymbol{s}_1|\boldsymbol{s}_n) & \cdots & P(\boldsymbol{s}_n|\boldsymbol{s}_n) 
\end{bmatrix}
\begin{bmatrix}
V(\boldsymbol{s}_1)\\
V(\boldsymbol{s}_2)\\
\cdots \\
V(\boldsymbol{s}_n)
\end{bmatrix}
\end{aligned}
$$

我们可以根据矩阵运算，得到以下解析解：

$$
\begin{aligned}
\boldsymbol{\mathcal{V}}&=\boldsymbol{\mathcal{R}}+\gamma \boldsymbol{\mathcal{P}} \boldsymbol{\mathcal{V}} \\
(\boldsymbol{\mathcal{I}}-\gamma \boldsymbol{\mathcal{P}})\boldsymbol{\mathcal{V}}
&=\boldsymbol{\mathcal{R}}\\
\boldsymbol{\mathcal{V}}&=(\boldsymbol{\mathcal{I}}-\gamma \boldsymbol{\mathcal{P}})^{-1}\boldsymbol{\mathcal{R}}
\end{aligned}
$$

以上解析解的计算复杂度是 $O(n^3)$，其中 $n$ 是状态个数，因此这种方法只适用很小的马尔可夫奖励过程。求解较大规模的马尔可夫奖励过程中的价值函数时，可以使用**动态规划（dynamic programming）**、**蒙特卡洛方法（Monte-Carlo method）**和**时序差分（temporal difference）**。

## 3 马尔可夫决策过程

**马尔可夫过程**和**马尔可夫奖励过程**都是自发改变的随机过程；而如果有一个外界的“刺激”来共同改变这个随机过程，就有了**马尔可夫决策过程（Markov decision process，MDP）**。我们将这个来自外界的刺激称为 **智能体（agent）** 的动作，在马尔可夫奖励过程（MRP）的基础上加入动作，就得到了马尔可夫决策过程（MDP）。马尔可夫决策过程由元组 $\langle \mathcal{S},\mathcal{A},P,r,\gamma \rangle$ 构成，其中：

- $\mathcal{S}$ 是状态的集合；
- $\mathcal{A}$ 是动作的集合;
- $\gamma$ 是折扣因子;
- $r(\boldsymbol{s},\boldsymbol{a})$ 是奖励函数，此时奖励可以同时取决于状态 $\boldsymbol{s}$ 和动作 $\boldsymbol{a}$，在奖励函数只取决于状态 $\boldsymbol{s}$ 时，退化为 $r(\boldsymbol{s})$；
- $P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a})$ 是状态转移函数，表示在状态 $\boldsymbol{s}$ 执行动作 $\boldsymbol{a}$ 之后达到状态 $\boldsymbol{s}'$ 的概率。

> 在上面 MDP 的定义中，不再使用类似 MRP 定义中的状态转移矩阵方式，而是直接表示成了状态转移函数。这样做一是因为此时状态转移与动作也有关，变成了一个三维数组，而不再是一个矩阵（二维数组）；二是因为状态转移函数更具有一般意义：如果状态集合不是有限的，就无法用数组表示，但仍然可以用状态转移函数表示。

不同于马尔可夫奖励过程，在马尔可夫决策过程中，通常存在一个 **智能体（agent）** 来执行动作。例如，一艘小船在大海中随着水流自由飘荡的过程就是一个马尔可夫奖励过程，它如果凭借运气漂到了一个目的地，就能获得比较大的奖励；如果有个水手在控制着这条船往哪个方向前进，就可以主动选择前往目的地获得比较大的奖励。

![![20250814163413-2025-08-14-16-34-13](httpsozzyc.oss-cn-shenzhen.aliyuncs.comNotePicture20250814163413-2025-08-14-16-34-13.png)-2025-08-15-10-46-41](https://ozzyc.oss-cn-shenzhen.aliyuncs.com/NotePicture/![20250814163413-2025-08-14-16-34-13](httpsozzyc.oss-cn-shenzhen.aliyuncs.comNotePicture20250814163413-2025-08-14-16-34-13.png)-2025-08-15-10-46-41.png)

马尔可夫决策过程是一个与时间相关的不断进行的过程，在智能体和环境 MDP 之间存在一个不断交互的过程。一般而言，它们之间的交互是如下图的循环过程：智能体根据当前状态 $\boldsymbol{S}_t$ 选择动作 $\boldsymbol{A}_t$；对于状态 $\boldsymbol{S}_t$ 和动作 $\boldsymbol{A}_t$，MDP 根据奖励函数和状态转移函数得到 $\boldsymbol{S}_{t+1}$ 和 $R_t$ 并反馈给智能体。

### 3.1 策略

智能体根据当前状态从动作的集合 $\mathcal{A}$ 中选择一个动作的函数，被称为 **策略（Policy）**，通常用字母 $\pi$ 表示。策略 $\pi(\boldsymbol{a}|\boldsymbol{s})=P(\boldsymbol{A}_t=\boldsymbol{a}|\boldsymbol{S}_t=\boldsymbol{s})$ 是一个函数，表示在输入状态为 $\boldsymbol{s}$ 情况下采取动作 $\boldsymbol{a}$ 的概率。

当一个策略是 **确定性策略（deterministic policy）** 时，它在每个状态时只输出一个确定性的动作，即只有该动作的概率为 1，其他动作的概率为 0；当一个策略是 **随机性策略（stochastic policy）** 时，它在每个状态时输出的是关于动作的概率分布，然后根据该分布进行采样就可以得到一个动作。

回顾一下在 MRP 中的价值函数，在 MDP 中也同样可以定义类似的价值函数。但此时的价值函数与策略有关，这意为着对于两个不同的策略来说，它们在同一个状态下的价值也很可能是不同的。这很好理解，因为不同的策略会采取不同的动作，从而之后会遇到不同的状态，以及获得不同的奖励，所以它们的累积奖励的期望也就不同，即状态价值不同。

> 在 MDP 中，由于马尔可夫性质的存在，策略只需要与当前状态有关，不需要考虑历史状态。

### 3.2 状态价值函数

我们用 $V^{\pi}(\boldsymbol{s})$ 表示在 MDP 中基于策略 $\pi$ 的**状态价值函数（state-value function）**，定义为从状态 $\boldsymbol{s}$ 出发遵循策略 $\pi$ 能获得的期望回报，数学表达为：

$$
V^{\pi}(\boldsymbol{s})=\mathbb{E}_{\pi}[G_t|\boldsymbol{S}_t=\boldsymbol{s}]
$$

> MDP 中，未来的状态转移不仅取决于环境，还取决于当前的策略 $\pi$。

### 3.3 动作价值函数

$V^{\pi}(\boldsymbol{s})$ 只能表达“按策略 $\pi$ 行动的平均前景”，但它不能比较不同动作的好坏。于是我们额外定义一个**动作价值函数（action-value function）**，用 $Q^{\pi}(\boldsymbol{s},\boldsymbol{a})$ 表示在 MDP 遵循策略 $\pi$ 时，对当前状态 $\boldsymbol{s}$ 执行动作 $\boldsymbol{a}$ 得到的期望回报：

$$
Q^{\pi}(\boldsymbol{s},\boldsymbol{a})=
\mathbb{E}_{\pi}[G_t|\boldsymbol{S}_t=\boldsymbol{s},\boldsymbol{A}_t=\boldsymbol{a}]
$$

状态价值函数和动作价值函数之间的关系为：在使用策略 $\pi$ 时，状态 $\boldsymbol{s}$ 的价值等于在该状态下基于策略 $\pi$ 采取所有动作的概率与相应的价值相乘再求和的结果：

$$
V^{\pi}(\boldsymbol{s}) = \sum_{\boldsymbol{a}\in \mathcal{A}}
\pi(\boldsymbol{a}|\boldsymbol{s}) \cdot Q^{\pi}(\boldsymbol{s},\boldsymbol{a})
$$

使用策略 $\pi$ 时，在状态 $\boldsymbol{s}$ 下采取动作 $\boldsymbol{a}$ 的价值等于即时奖励加上经过衰减的所有可能的下一个状态的状态转移概率与相应的价值的乘积：

$$
Q^{\pi}(\boldsymbol{s},\boldsymbol{a}) = 
r(\boldsymbol{s},\boldsymbol{a})+\gamma
\sum_{\boldsymbol{s}' \in \mathcal{S}}
P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) \cdot V^{\pi}(\boldsymbol{s}')
$$

### 3.4 贝尔曼期望方程

对于使用策略 $\pi$ 时的动作价值函数 $Q^{\pi}(\boldsymbol{s},\boldsymbol{a})$，有：

$$
\begin{aligned}
Q^{\pi}(\boldsymbol{s},\boldsymbol{a})&=
\mathbb{E}^{\pi}[R_t+\gamma R_{t+1}+\gamma^2 R_{t+2}+\cdots
| \boldsymbol{S}_t = \boldsymbol{s},\boldsymbol{A}_t = \boldsymbol{a}]\\
&=\mathbb{E}^{\pi}[R_t+\gamma (R_{t+1}+\gamma R_{t+2}+\cdots)
| \boldsymbol{S}_t = \boldsymbol{s},\boldsymbol{A}_t = \boldsymbol{a}]\\
&=\mathbb{E}^{\pi}[R_t+\gamma G_{t+1}
| \boldsymbol{S}_t = \boldsymbol{s},\boldsymbol{A}_t = \boldsymbol{a}]\\
&=\mathbb{E}^{\pi}[R_t| \boldsymbol{S}_t = \boldsymbol{s},\boldsymbol{A}_t = \boldsymbol{a}]+\gamma \mathbb{E}^{\pi}[G_{t+1}| \boldsymbol{S}_t = \boldsymbol{s},\boldsymbol{A}_t = \boldsymbol{a}]\\
&=r(\boldsymbol{s},\boldsymbol{a})+\gamma \sum_{\boldsymbol{s}'\in \mathcal{S}}P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) \cdot  \mathbb{E}^{\pi}[G_{t+1}| \boldsymbol{S}_t = \boldsymbol{s},\boldsymbol{S}_{t+1} = \boldsymbol{s}',\boldsymbol{A}_t = \boldsymbol{a},\boldsymbol{A}_{t+1} = \boldsymbol{a}'] \\
&=r(\boldsymbol{s},\boldsymbol{a})+\gamma \sum_{\boldsymbol{s}'\in \mathcal{S}}P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) \cdot  \sum_{\boldsymbol{a}' \in \mathcal{A}} \pi(\boldsymbol{a}'|\boldsymbol{s}') \cdot
\mathbb{E}^{\pi}[G_{t+1}| \boldsymbol{S}_{t+1} = \boldsymbol{s}',\boldsymbol{A}_{t+1} = \boldsymbol{a}']\\
&=r(\boldsymbol{s},\boldsymbol{a})+\gamma \sum_{\boldsymbol{s}'\in \mathcal{S}}P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) \cdot  \sum_{\boldsymbol{a}' \in \mathcal{A}} \pi(\boldsymbol{a}'|\boldsymbol{s}') \cdot
Q^{\pi}(\boldsymbol{s}',\boldsymbol{a}')
\end{aligned}
$$

对于使用策略 $\pi$ 时的动作价值函数 $V^{\pi}(\boldsymbol{s})$，有：

$$
\begin{aligned}
V^{\pi}(\boldsymbol{s})&=\mathbb{E}_{\pi}[G_t|\boldsymbol{S}_t=\boldsymbol{s}]\\
&=\mathbb{E}_{\pi}[R_t+\gamma R_{t+1}+\gamma^2 R_{t+2}+\cdots|\boldsymbol{S}_t=\boldsymbol{s}]\\
&=\mathbb{E}_{\pi}[R_t+\gamma (R_{t+1}+\gamma R_{t+2}+\cdots)|\boldsymbol{S}_t=\boldsymbol{s}]\\
&=\mathbb{E}_{\pi}[R_t+\gamma G_{t+1}|\boldsymbol{S}_t=\boldsymbol{s}]\\
&=\mathbb{E}_{\pi}[R_t|\boldsymbol{S}_t=\boldsymbol{s}]+\gamma\mathbb{E}_{\pi}[G_{t+1}|\boldsymbol{S}_t=\boldsymbol{s}]\\
&=\sum_{\boldsymbol{a} \in \mathcal{A}} \pi(\boldsymbol{a}|\boldsymbol{s})  r(\boldsymbol{s},\boldsymbol{a})+\gamma 
\sum_{\boldsymbol{a} \in \mathcal{A}} \pi(\boldsymbol{a}|\boldsymbol{s}) \mathbb{E}_{\pi}[G_{t+1}|\boldsymbol{S}_t=\boldsymbol{s},\boldsymbol{A}_t = \boldsymbol{a}]\\
&=\sum_{\boldsymbol{a} \in \mathcal{A}} \pi(\boldsymbol{a}|\boldsymbol{s})  r(\boldsymbol{s},\boldsymbol{a})+\gamma 
\sum_{\boldsymbol{a} \in \mathcal{A}} \pi(\boldsymbol{a}|\boldsymbol{s}) \sum_{\boldsymbol{s}'\in \mathcal{S}}P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) 
\mathbb{E}_{\pi}[G_{t+1}|\boldsymbol{S}_{t+1}=\boldsymbol{s}']\\
&=\sum_{\boldsymbol{a} \in \mathcal{A}} \pi(\boldsymbol{a}|\boldsymbol{s}) 
\left(  r(\boldsymbol{s},\boldsymbol{a})+\gamma  \sum_{\boldsymbol{s}'\in \mathcal{S}}P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) 
V^{\pi}(\boldsymbol{s}')
\right)
\end{aligned}
$$

以上是两个价值函数的**贝尔曼期望方程（Bellman expectation equation）**。