---
title: '矩阵与线性变换'
publishDate: '2025-07-03'
updatedDate: '2025-07-10'
description: 'Test page for any purpose'
tags:
  - Test
language: 'Chinese'
---

# 矩阵与线性变换

## 1. 线性变换

A **transformation** $T$ assigns an output $T(\boldsymbol{v})$ to each input vector $\boldsymbol{v}$ in the whole space $\mathbf{V}$. The transformation is **linear** if it meets these requirements for all $\boldsymbol{v}$ and $\boldsymbol{w}$:

$$
\begin{aligned}
&(a):~T(\boldsymbol{v}+\boldsymbol{w}) = T(\boldsymbol{v})+T(\boldsymbol{w})\\
&(b):~T(c\boldsymbol{v}) = cT(\boldsymbol{v})~~~\mathrm{for~all}~c.
\end{aligned}
$$

> 摘自 Gilbert Strang 著的 《Introduction to Linear Algebra》

设 $T$ 是定义在向量空间 $\mathbf{V}$ 上的变换，若对任意向量 $\boldsymbol{v}, \boldsymbol{w} \in \mathbf{V}$ 及任意标量 $c$，$T$ 满足：

$$
\begin{aligned}
&(a):~T(\boldsymbol{v}+\boldsymbol{w}) = T(\boldsymbol{v})+T(\boldsymbol{w})\\
&(b):~T(c\boldsymbol{v}) = cT(\boldsymbol{v})
\end{aligned}
$$

则称 $T$ 为一个线性变换。

>如果输入为零向量 $\boldsymbol{v} = 0$，则线性变换的输出为 $T(\boldsymbol{v}) = 0$。

进一步地，线性性条件可统一表述为：

$$
T(c\boldsymbol{v}+d\boldsymbol{w}) = cT(\boldsymbol{v})+dT(\boldsymbol{w})
$$
  
其中 $c$ 和 $d$ 为任意标量，$\boldsymbol{v}, \boldsymbol{w} \in \mathbf{V}$。

推广到多个向量线性组合的情形：

$$
T(c_1\boldsymbol{v}_1+c_2\boldsymbol{v}_2+ \cdots c_n\boldsymbol{v}_n) = c_1 T(\boldsymbol{v}_1)+c_2 T(\boldsymbol{v}_2) + \cdots c_n T(\boldsymbol{v}_n)
$$

其中 $c_i$ 为任意标量，$\boldsymbol{v}_i  \in \mathbf{V}$。

**线性变换保持向量间的线性关系，即其对线性组合的作用等价于对各向量分别变换后再进行线性组合。**

## 2. 基底与线性变换

**基底**是向量空间 $\mathbf{V}$ 中一组线性无关的向量，记作 $(\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_n)$，它们的线性组合可以**张成**整个空间 $\mathbf{V}$，即有：

$$
\mathbf{V} = \mathrm{span}(\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_n)
$$

这意味着，向量空间 $\mathbf{V}$ 中的任意向量都可以唯一表示为这组基底向量的线性组合。

对任一向量 $\boldsymbol{x} = c_1\boldsymbol{v}_1+c_2\boldsymbol{v}_2+ \cdots c_n\boldsymbol{v}_n$，向量 $\boldsymbol{x}$ 在基底下的坐标表示为 $(c_1,c_2,\cdots,c_n)$。

上述提到的线性变换满足的条件：

$$
T(c_1\boldsymbol{v}_1+c_2\boldsymbol{v}_2+ \cdots c_n\boldsymbol{v}_n) = c_1 T(\boldsymbol{v}_1)+c_2 T(\boldsymbol{v}_2) + \cdots c_n T(\boldsymbol{v}_n)
$$

实际上表达的是：知道了 $T$ 在基底上的变换 $T(\boldsymbol{v}_1),T(\boldsymbol{v}_2), \cdots,T(\boldsymbol{v}_n)$，就能通过“坐标”来计算 $T$ 在任意向量上的变换。

## 3. 矩阵与线性变换

既然线性变换 $T$ 的行为完全由其对一组基底向量的作用决定，那么我们可以将 $T$ 在基底 $(\boldsymbol{v}_1, \boldsymbol{v}_2, \cdots, \boldsymbol{v}_n)$ 上的变换结果记录下来：

$$
T(\boldsymbol{v}_1)=\boldsymbol{w}_1,~
T(\boldsymbol{v}_2)=\boldsymbol{w}_2,~
\cdots,~
T(\boldsymbol{v}_n)=\boldsymbol{w}_n
$$

将这些基底的变换结果作为列向量排成一个矩阵：

$$
\boldsymbol{A}=
\begin{bmatrix}
\boldsymbol{w}_1
&\boldsymbol{w}_2
&\cdots
&\boldsymbol{w}_n
\end{bmatrix}
$$

于是，对任意向量 $\boldsymbol{x} = c_1\boldsymbol{v}_1 + c_2\boldsymbol{v}_2 + \cdots + c_n\boldsymbol{v}_n$，其在线性变换 $T$ 下的结果为：

$$
\begin{aligned}
T(c_1\boldsymbol{v}_1+c_2\boldsymbol{v}_2+ \cdots c_n\boldsymbol{v}_n) 
&= c_1 T(\boldsymbol{v}_1)+c_2 T(\boldsymbol{v}_2) + \cdots c_n T(\boldsymbol{v}_n)\\
&=c_1\boldsymbol{w}_1+c_2\boldsymbol{w}_2+\cdots+c_n\boldsymbol{w}_n\\
&=\boldsymbol{A}
\begin{bmatrix}
c_1
\\
c_2
\\
\cdots
\\
c_n
\end{bmatrix}
\end{aligned}
$$

矩阵乘以向量，实际上就是描述一个线性变换如何作用在向量上，由矩阵的列向量给出在基底上的变换结果。

我们来看一个二维空间中的例子：

设线性变换 $T$ 由下列矩阵 $\boldsymbol{A}$ 表示：

$$
\boldsymbol{A}=
\begin{bmatrix}
2&1\\
0&3
\end{bmatrix}
$$

这意味着：$T$ 作用在任意二维向量 $\boldsymbol{x} = \begin{bmatrix} x_1 \ x_2 \end{bmatrix}^\mathrm{T}$ 上的结果为：

$$
\begin{aligned}
T(\boldsymbol{x})&=\boldsymbol{A}\boldsymbol{x}\\
&=\begin{bmatrix}
2&1\\
0&3
\end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}\\
&=\begin{bmatrix} 2x_1+x_2 \\ 3x_2 \end{bmatrix}
\end{aligned}
$$

这个运算看似只是“矩阵乘法”，但它实际上体现了两个关键的线性变换特征：

- 第一列 $\begin{bmatrix} 2 \ 0 \end{bmatrix}$ 表示原始基向量 $\boldsymbol{v}_1 = \begin{bmatrix} 1 \ 0 \end{bmatrix}$ 在 $T$ 下的变换结果，即 $T(\boldsymbol{v}_1)$
- 第二列 $\begin{bmatrix} 1 \ 3 \end{bmatrix}$ 表示原始基向量 $\boldsymbol{v}_2 = \begin{bmatrix} 0 \ 1 \end{bmatrix}$ 在 $T$ 下的变换结果，即 $T(\boldsymbol{v}_2)$

任意一个线性变换都可以表示为某个矩阵的作用，而任意一个矩阵都对应一个线性变换。矩阵与线性变换之间是一一对应的，这正是线性代数研究矩阵的核心意义。
