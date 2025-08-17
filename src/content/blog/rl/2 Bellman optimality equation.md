---
title: '贝尔曼最优方程'
publishDate: '2025-08-17'
updatedDate: '2025-08-17'
description: '深入强化学习'
tags:
  - 强化学习
language: 'Chinese'
---

# 贝尔曼最优方程

## 1 占用度量

**占用度量（occupancy measure）** 其实就是在 MDP 的框架下，统计策略在长期运行中，状态（或状态-动作对）出现的频率分布。假设你有一个策略 $\pi(\boldsymbol{a}|\boldsymbol{s})$，你从某个起始状态分布 $v_0(\boldsymbol{s})$ 开始，不断与环境交互：

- 有的状态你经常去到（比如起点附近的安全区）；
- 有的状态你几乎从不访问（比如很偏僻的角落）。

占用度量就是用一个概率分布，把这种“访问频率”精确描述下来。它是强化学习里的“交通流量统计表”，告诉你一个策略下，在哪些地方你花的时间多，哪些地方你几乎不去。

> 起始状态不一定是固定的，通常是由一个概率分布描述的。

### 1.1 状态占用度量

对于折扣因子 $\gamma \in [0,1)$，**状态占用度量（State Occupancy Measure）** 定义为：

$$
\nu^{\pi}(\boldsymbol{s})=(1-\gamma)\sum_{t=0}^{\infty} \gamma^{t}P_t^{\pi}(\boldsymbol{s})
$$

其中，$P_t^{\pi}(\boldsymbol{s})$ 表示采取策略 $\pi$ 使得智能体在 $t$ 时刻状态为 $\boldsymbol{s}$ 的概率；而 $1-\gamma$ 是为了使得 $\sum_{\boldsymbol{s} \in \mathcal{S}} \nu^{\pi}(\boldsymbol{s})=1$ 的归一化因子。

显然，对于状态 $\boldsymbol{s}$，在策略 $\pi$ 下，如果有更高的可能性早访问到，那么由于较小的时间步 $t$ 对应更大的权重 $\gamma^t$，它在占用度量中的值就会更大；反之，如果一个状态通常在较晚的时间步才被访问到，那么其对应的概率项会被较小的折扣权重削弱，从而在占用度量中的值较小。这说明占用度量不仅反映了状态被访问的频率，还体现了**访问的时间先后性**，即越早访问到的状态在占用度量中贡献越大。

> 如果一个状态总是很晚才出现，它的占用度量就小，这说明在折扣回报的视角下，这个状态的重要性低，策略优化会更关注前期就能到达、能获得奖励的状态。

### 1.2 状态-动作占用度量（State-Action Occupancy Measure）

$$
\begin{aligned}
\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})
&=(1-\gamma)\sum_{t=0}^{\infty} \gamma^{t}P_t^{\pi}(\boldsymbol{s},\boldsymbol{a})\\
&=(1-\gamma)\sum_{t=0}^{\infty} \gamma^{t}P_t^{\pi}(\boldsymbol{s})\pi (\boldsymbol{a}|\boldsymbol{s})
\end{aligned}
$$

其中，$P_t^{\pi}(\boldsymbol{s},\boldsymbol{a})$ 表示采取策略 $\pi$ 使得智能体在 $t$ 时刻状态为 $\boldsymbol{s}$ 且选择动作 $\boldsymbol{a}$ 的概率；$\pi (\boldsymbol{a}|\boldsymbol{s})$ 为策略，表示在状态 $\boldsymbol{s}$ 时，选择动作 $\boldsymbol{a}$ 的概率；$1-\gamma$ 是归一化因子。

**状态–动作占用度量** $\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})$ 表示在策略 $\pi$ 下，智能体在整个交互过程中以折扣加权的方式出现在状态–动作对 $(\boldsymbol{s},\boldsymbol{a})$ 的比例。状态占用度量 $\nu^{\pi}(\boldsymbol{s})$ 一样，它统计了某个状态在不同时间步出现的概率，并对不同时间步用折扣因子 $\gamma^t$ 加权。但在此基础上，还要考虑在该状态下采取某个动作 $\boldsymbol{a}$ 的概率 $\pi (\boldsymbol{a}|\boldsymbol{s})$，从而得到状态–动作对的加权访问概率。显然有：

$$
\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})=
\nu^{\pi}(\boldsymbol{s})\pi (\boldsymbol{a}|\boldsymbol{s})
$$

$\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})$ 可以看作是“策略 $\pi$ 在长期运行过程中，折扣归一化后，选择状态–动作对 $(\boldsymbol{s},\boldsymbol{a})$ 的频率”。如果某个状态 $\boldsymbol{s}$ 更可能在较早时间被访问到，且策略 $\pi$ 在该状态下较常选择动作 $\boldsymbol{a}$，那么 $\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})$ 的值就会更大。

**定理 1：** 智能体分别以策略 $\pi_1$ 和 $\pi_2$ 和同一个 MDP 交互得到的状态–动作占用度量 $\rho^{\pi_1}$ 和 $\rho^{\pi_2}$ 满足：

$$
\rho^{\pi_1}=\rho^{\pi_2} \leftrightharpoons \pi_1 =\pi_2
$$

以上表明：**状态–动作占用度量与策略是一一对应的**。

**定理 2：** 给定一合法的状态–动作占用度量 $\rho$，可生成该占用度量的唯一策略是：

$$
\begin{aligned}
\pi(\boldsymbol{a}|\boldsymbol{s})&=
\frac{\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})}
{\nu^{\pi}(\boldsymbol{s})}\\
&=\frac{\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})}
{(1-\gamma)\sum_{t=0}^{\infty} \gamma^{t}P_t^{\pi}(\boldsymbol{s})}\\
&=\frac{\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})}
{(1-\gamma)\sum_{t=0}^{\infty} \gamma^{t}
\sum_{\boldsymbol{a}}
P_t^{\pi}(\boldsymbol{s},\boldsymbol{a})}\\
&=\frac{\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})}
{\sum_{\boldsymbol{a}}[(1-\gamma)\sum_{t=0}^{\infty} \gamma^{t}
P_t^{\pi}(\boldsymbol{s},\boldsymbol{a})]}\\
&=\frac{\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})}
{\sum_{\boldsymbol{a}'}\rho^{\pi}(\boldsymbol{s},\boldsymbol{a}')}
\end{aligned}
$$

> 其中，分母中的 $\boldsymbol{a}'$ 仅仅是求和对象，用于遍历所有可能的动作，与分子中的 $\boldsymbol{a}$ 不同。也就是说，$\pi(\boldsymbol{a}|\boldsymbol{s})$ 依赖的动作变量是分子里的 $\boldsymbol{a}$。

以上说明了**状态–动作占用度量完全刻画了一个策略**，即有：

- 给定一个合法的状态–动作占用度量 $\rho^{\pi}(\boldsymbol{s},\boldsymbol{a})$，就可以唯一恢复出生成它的策略 $\pi(\boldsymbol{a}|\boldsymbol{s})$；
- 这也侧面证明了定理一：$\rho^{\pi_1}=\rho^{\pi_2} \leftrightharpoons \pi_1 =\pi_2$，不可能有不同的策略生成同一个 $\rho$；

> 在优化问题里，我们可以等价地把优化策略问题改写成优化占用度量问题。

## 2 贝尔曼最优方程

### 2.1 最优策略

**强化学习的目标通常是找到一个策略，使得智能体从初始状态出发能获得最多的期望回报。**

我们首先定义策略之间的偏序关系：当且仅当对于任意的状态 $\boldsymbol{s}$ 都有 $V^{\pi}(\boldsymbol{s})\geq V^{\pi '}(\boldsymbol{s})$ 时，则记 $\pi \geq \pi '$。于是在有限状态和动作集合的 MDP 中，至少存在一个策略比其他所有策略都好或者至少存在一个策略不差于其他所有策略，这个策略就是**最优策略（optimal policy）**。最有策略可能有多个，我们都将其表示为 $\pi^{*}(\boldsymbol{s})$。

最优策略都有相同的状态价值函数，称之为**最优状态价值函数**，表示为：

$$
V^{*}(\boldsymbol{s})=\max_{\pi}V^{\pi}(\boldsymbol{s}) \ \ \ \forall \boldsymbol{s} \in \mathcal{S}
$$

$V^*(\boldsymbol{s})$ 是状态 $s$ 的最优价值。它等于在所有可能的策略 $\pi$ 中，取那个能让 $V^\pi(s)$ 最大的值。

同理，定义**最优动作价值函数**：

$$
Q^{*}(\boldsymbol{s},\boldsymbol{a})
=\max_{\pi}Q^{\pi}(\boldsymbol{s},\boldsymbol{a})
\ \ \ \forall \boldsymbol{s} \in \mathcal{S},\boldsymbol{a} \in \mathcal{A}
$$

$Q^*(\boldsymbol{s},\boldsymbol{a})$ 是状态 $\boldsymbol{s}$ 下采取动作 $\boldsymbol{a}$ 的最优价值。它等于在所有可能的策略 $\pi$ 中，取那个能让 $Q^\pi(s,a)$ 最大的值。

为了使 $Q^*(\boldsymbol{s},\boldsymbol{a})$ 最大，我们需要在当前的状态动作对 $(\boldsymbol{s},\boldsymbol{a})$ 之后都执行最优策略。于是我们得到了最优状态价值函数和最优动作价值函数之间的关系：

$$
Q^{*}(\boldsymbol{s},\boldsymbol{a}) = 
r(\boldsymbol{s},\boldsymbol{a})+\gamma
\sum_{\boldsymbol{s}' \in \mathcal{S}}
P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) \cdot V^{*}(\boldsymbol{s}')
$$

这与在普通策略下的状态价值函数和动作价值函数之间的关系是一样的。另一方面，最优状态价值是选择此时使最优动作价值最大的那一个动作时的状态价值：

$$
V^{*}(\boldsymbol{s})=\max_{\boldsymbol{a} \in \mathcal{A}}Q^{*}(\boldsymbol{s},\boldsymbol{a})
$$

> $Q^{*}(\boldsymbol{s},\boldsymbol{a})$ 表示在状态 $\boldsymbol{s}$ 时，若当前采取动作 $\boldsymbol{a}$，然后从下一步开始都遵循最优策略，所能获得的最大期望回报；$V^{*}(\boldsymbol{s})$ 表示在状态 $\boldsymbol{s}$ 时，遵循最优策略能拿到的最大期望回报；从状态 $\boldsymbol{s}$ 出发，最优策略就是选择让 $Q^*(\boldsymbol{s},\boldsymbol{a})$ 最大的动作，所以状态的最优价值 $V^{*}(\boldsymbol{s})$ 就是候选动作里最大的那个值。

### 2.2 贝尔曼最优方程

根据 $V^{*}(\boldsymbol{s})$ 和 $Q^{*}(\boldsymbol{s},\boldsymbol{a})$ 的关系，我们可以得到**贝尔曼最优方程（Bellman optimality equation）**：

（1）最优状态价值函数的贝尔曼方程：

$$
V^{*}(\boldsymbol{s})=\max_{\boldsymbol{a} \in \mathcal{A}}\{
r(\boldsymbol{s},\boldsymbol{a})+\gamma
\sum_{\boldsymbol{s}'\in \mathcal{S}} 
P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a})
V^{*}(\boldsymbol{s}')
\}
$$

我们可以按照以下思路理解，首先状态价值函数如下：

$$
V^{\pi}(\boldsymbol{s})=\sum_{\boldsymbol{a} \in \mathcal{A}} \pi(\boldsymbol{a}|\boldsymbol{s}) 
\left(  r(\boldsymbol{s},\boldsymbol{a})+\gamma  \sum_{\boldsymbol{s}'\in \mathcal{S}}P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) 
V^{\pi}(\boldsymbol{s}')
\right)
$$

最优状态价值函数一定是在最优策略下取得的，此时我们不需要再用 $\sum_{\boldsymbol{a} \in \mathcal{A}} \pi(\boldsymbol{a}|\boldsymbol{s})$ 求动作的期望，而是直接选择使回报最大的动作。在状态 $\boldsymbol{s}$ 下，最优策略会选择这个动作，使当前的即时奖励 + 未来最优状态价值的期望最大。

（2）最优状态-动作价值函数的贝尔曼方程：

$$
Q^{*}(\boldsymbol{s},\boldsymbol{a}) = 
r(\boldsymbol{s},\boldsymbol{a})+\gamma
\sum_{\boldsymbol{s}' \in \mathcal{S}}
P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) \max_{\boldsymbol{a}' \in \mathcal{A}} Q^{*}(\boldsymbol{s}',\boldsymbol{a}')
$$

同样按照以下思路理解，首先状态-动作价值函数如下：

$$
Q^{\pi}(\boldsymbol{s},\boldsymbol{a})
=r(\boldsymbol{s},\boldsymbol{a})+\gamma \sum_{\boldsymbol{s}'\in \mathcal{S}}P(\boldsymbol{s}'|\boldsymbol{s},\boldsymbol{a}) \cdot  \sum_{\boldsymbol{a}' \in \mathcal{A}} \pi(\boldsymbol{a}'|\boldsymbol{s}') \cdot
Q^{\pi}(\boldsymbol{s}',\boldsymbol{a}')
$$

最优状态-动作价值函数一定是在最优策略下取得的，此时我们不需要再用 $\sum_{\boldsymbol{a}' \in \mathcal{A}} \pi(\boldsymbol{a}'|\boldsymbol{s}')$ 求下一步动作的期望，而是直接选择使未来回报最大的动作。在状态 $\boldsymbol{s}$ 下，若当前采取动作 $\boldsymbol{a}$，最优策略会在下一状态 $\boldsymbol{s}'$ 选择能够最大化状态-动作价值 $Q^{*}(\boldsymbol{s}',\boldsymbol{a}')$ 的动作，使当前的即时奖励 + 未来最优状态-动作价值的期望最大。

强化学习的目标是找到最优策略 $\pi^{*}$，但直接枚举所有策略计算期望回报是不现实的。而贝尔曼最优方程告诉我们：在每个状态，最优策略就是选择让 $V^{*}$ 或 $Q^{*}$ 最大的动作。也就是说，只要有了 $V^{*}$ 或 $Q^{*}$，就能立即导出最优策略。它把“最优策略求解”问题转化为“求最优价值函数”问题，极大简化了计算。
