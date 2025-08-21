---
title: '变分自编码器（VAE）'
publishDate: '2025-08-21'
updatedDate: '2025-08-21'
description: 'AE与VAE'
tags:
  - 生成式AI
language: 'Chinese'
---

# 变分自编码器

## 1 自编码器

**自编码器（Autoencoder, AE）** 是一种神经网络模型，其主要目标是通过无监督方式学习输入数据的紧凑且高效的表示。它常被应用于降维、特征提取、数据压缩以及生成建模等任务。

AE 主要可以分为两个部分：

- **编码器（Encoder）**：输入原始数据 $\boldsymbol{x}$（比如一张图像、一个文本向量），通过一系列神经网络层（如全连接层、卷积层）逐步压缩信息，得到一个低维的潜在向量 $\boldsymbol{z}$。这个 $\boldsymbol{z}$ 就是数据在 **潜在空间（latent space）** 中的表示，往往比原始数据维度低得多。

- **解码器（Decoder）**：以潜在向量 $\boldsymbol{z}$ 为输入，通过神经网络结构逐步还原出原始数据的近似重建 $\hat{\boldsymbol{x}}$。

![20250820161044-2025-08-20-16-10-44](https://ozzyc.oss-cn-shenzhen.aliyuncs.com/NotePicture/20250820161044-2025-08-20-16-10-44.png)

如下示例，将 MNIST 手写数字数据集图像展平（28 × 28 = 784 维），然后通过两个 Linear 层组成的 Encoder 压缩到 64 维，然后通过两个 Linear 层组成的 Decoder 还原图像。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2. 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 展平成784维
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# 3. 定义自编码器
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()   
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

# 4. 初始化模型与优化器
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 5. 训练
epochs = 40
for epoch in range(epochs):
    total_loss = 0
    for x, _ in train_loader:
        x = x.to(device)

        optimizer.zero_grad()
        x_recon = model(x)
        loss = criterion(x_recon, x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# 6. 测试与可视化
test_imgs, _ = next(iter(train_loader))
test_imgs = test_imgs.to(device)
with torch.no_grad():
    recon = model(test_imgs)

# 取前8张进行对比
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(8):
    axes[0, i].imshow(test_imgs[i].cpu().view(28, 28), cmap="gray")
    axes[0, i].axis("off")
   
    axes[1, i].imshow(recon[i].cpu().view(28, 28), cmap="gray")
    axes[1, i].axis("off")

    if i == 0:
        axes[0, i].set_title("Original",fontsize=12,loc="left")
        axes[1, i].set_title("Reconstructed",fontsize=12,loc="left")

plt.show()
```

经过训练后，模型能够将输入的高维图像有效地压缩到低维潜在表示 $\boldsymbol{z} \in \mathbb{R}^{64}$，同时通过解码器尽可能重建出原始图像。潜在表示 $\boldsymbol{z}$ 捕捉了手写数字的关键结构特征，例如数字的整体形状和笔画分布，使得在低维空间中就能保留图像的主要信息。

![ae_result-2025-08-20-16-58-59](https://ozzyc.oss-cn-shenzhen.aliyuncs.com/NotePicture/ae_result-2025-08-20-16-58-59.png)

**自编码器** 没有要求 $\boldsymbol{z}$ 必须服从某个特定分布（如高斯分布），也没有约束 $\boldsymbol{z}$ 的取值范围。

在 MNIST 手写数字数据集中，编码器可能会将数字 “0” 的图像映射到潜在空间（64 维高维空间）中的一片螺旋形区域，将数字 “1” 映射到一个完全不相邻的条状区域，而数字 “2” 又映射到另一块奇特的区域。相同类别的样本虽然在潜在空间中倾向于聚集，但这些表示仍然是 **离散分布的**——潜在空间中有许多位置并不对应任何有效的图像。

因此，当 AE 训练完成后，如果我们从潜在空间中 **随机采样** 一个向量 $\boldsymbol{z}^{\mathrm{sample}}$，再输入到解码器中，生成的结果往往是不可控的，甚至可能完全无效。也正因如此，普通 AE 更适合作为一种 **特征压缩或表示学习工具**，而不是一个真正的 **生成模型**，因为它的潜在空间并不具备良好的采样与生成能力。

## 2 变分自编码器

为了解决这一问题，**变分自编码器（VAE）** 在训练过程中引入了分布约束，强制潜在变量 $\boldsymbol{z}$ 服从一个连续、光滑的先验分布，通常是多维标准高斯 $p(\boldsymbol{z})=\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$。这样一来，潜在空间被“填满”并具有结构化特性。此时从该空间中进行随机采样，并通过解码器就可以生成连贯、有效的样本。

### 2.1 近似后验：变分分布

编码器的任务是进行**后验推断**：给定观测 $\boldsymbol{x}$，它输出潜在变量 $\boldsymbol{z}$ 的分布，即真实后验：

$$
\begin{aligned}
p_{\theta}(\boldsymbol{z}|\boldsymbol{x}) =
\cfrac{p_{\theta}(\boldsymbol{z},\boldsymbol{x})}{p_{\theta}(\boldsymbol{x})}=\cfrac{p_{\theta}(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})}{p_{\theta}(\boldsymbol{x})}
\end{aligned}
$$

在实际应用中，真实后验 $p_{\theta}(\boldsymbol{z}|\boldsymbol{x})$ 通常是不可解的，因为边缘似然 $p_{\theta}(\boldsymbol{x})=\int{p_{\theta}(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})\mathrm{d}\boldsymbol{z}}$ 在高维空间几乎无法计算。为了解决这一问题，VAE 引入了一个可参数化的分布  $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$ 去近似真实后验：

$$
q_{\phi}(\boldsymbol{z}|\boldsymbol{x})
\approx 
p_{\theta}(\boldsymbol{z}|\boldsymbol{x})
$$

真实后验可能是任意复杂分布（甚至多峰、不对称），直接用精确分布建模几乎不可能。在 **变分推断** 里，选一个简单可控的分布族（比如高斯）去近似复杂后验，训练过程中 **KL 散度** 会迫使它尽量贴近真实后验。即使真实后验不是单峰高斯，优化会让 $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$​ 尽量“抓住”真实后验的核心区域（概率质量最大的地方）。基于这一理论，VAE 选择高斯分布作为近似后验，因此其编码器的任务就是学习这个变分分布 $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$ 的参数（均值和方差），从而实现对潜在空间的概率化建模。

解码器的任务是生成观测数据，即学习给定潜在变量 $\boldsymbol{z}$ 时观测 $\boldsymbol{x}$ 的条件分布：

$$
p_{\theta}(\boldsymbol{x}|\boldsymbol{z})
$$

在训练过程中，解码器通过最大化似然（或等价地最小化重构误差）来学习如何从潜在空间生成尽可能接近真实观测的数据。

### 2.3 核心目标：变分下界

设真实数据都是从未知分布 $p_{\mathrm{data}}(\boldsymbol{x})$ 中采样出来的。生成模型的任务就是学习一个参数化分布 $p_{\theta}(\boldsymbol{x})$，使其尽可能逼近 $p_{\mathrm{data}}(\boldsymbol{x})$。因此，我们希望最大化边缘似然：

$$
\begin{aligned}
\log p_{\theta}(\boldsymbol{x})
&=\log \int{p_{\theta}(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})~\mathrm{d}\boldsymbol{z}}\\
&=\log \int{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})   
\cfrac{p_{\theta}(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}
~\mathrm{d}\boldsymbol{z}} ~~~~~~~~~~(引入变分分布)\\
&=\log \mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\cfrac{p_{\theta}(\boldsymbol{x}|\boldsymbol{z})p(\boldsymbol{z})}{q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}]\\
&\geq \mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z}) +\log p(\boldsymbol{z}) - \log q_{\phi}(\boldsymbol{z}|\boldsymbol{x})] ~~~~~~~~~~(\mathrm{Jensen}~不等式)\\
&=\mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})]- \mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[ \log q_{\phi}(\boldsymbol{z}|\boldsymbol{x})-\log p(\boldsymbol{z})]\\
&=\mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})]
-\mathrm{KL}[q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p(\boldsymbol{z})]~~~~~~~~~~(变分下界~\mathrm{ELBO})
\end{aligned}
$$

**变分下界（ELBO）** 由两个核心项组成：

- **重构项** $\mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})]$ ：通过优化解码器参数 $\theta$，模型能够在给定潜在变量 $\boldsymbol{z}$ 的情况下尽可能准确地还原原始输入 $\boldsymbol{x}$。这一过程保证了潜在表示中保留与输入数据相关的重要信息，从而使得解码器能够进行有效的重建。
- **正则项** $\mathrm{KL}[q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p(\boldsymbol{z})]$ ：它保证了编码器学到的潜在分布不会偏离设定的先验分布 $p(\boldsymbol{z})=\mathcal{N}(\boldsymbol{0},\boldsymbol{I})$。这一约束使得在训练完成后，即便我们直接从先验分布中随机采样潜在变量 $\boldsymbol{z}$，解码器也能生成连贯、合理的样本。

综上，**重构项** 保证潜在变量中携带与输入相关的信息，**正则项** 则将这些潜在表示与全局先验对齐，从而使得模型不仅能重建已有样本，还能在生成阶段通过“纯随机采样”产生多样化且合理的新样本。

### 2.4 可微采样：重参数化

为了优化 VAE，我们最大化变分下界：

$$
\mathcal{L}_{b}=\mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})]
-\mathrm{KL}[q_{\phi}(\boldsymbol{z}|\boldsymbol{x})||p(\boldsymbol{z})]
$$

其中：

- 重构项 $\mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})]$ 衡量通过解码器从隐变量 $\boldsymbol{z}$ 重构原始数据 $\boldsymbol{x}$ 的好坏。
- 正则项 $\mathbb{E}_{\boldsymbol{z}\sim q_{\phi}(\boldsymbol{z}|\boldsymbol{x})}[\log p_{\theta}(\boldsymbol{x}|\boldsymbol{z})]$ 约束编码器输出的分布 $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$ 与先验分布 $p(\boldsymbol{z})$ 之间的差异。

我们希望重构项尽可能大，正则项应尽可能小。正则项尽可能小，$q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$ 就要尽可能接近标准正态分布，就意味着 $\boldsymbol{z}$ 趋同，没有任何辨识度，这样的话重构项就小了；而如果重构项大的话，预测就准确，此时 $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$ 就不会太随机，正则项就不可能小了。所以这两部分的 loss 其实是**相互拮抗**的，要整体来看。

对于正则项的优化，因为潜在变量 $\boldsymbol{z}$ 是从编码器输出的分布 $q_{\phi}(\boldsymbol{z}|\boldsymbol{x})$ 中采样得到的（随机的过程），此时如果直接对 $\boldsymbol{z}$ 采样计算梯度，梯度无法顺利反向传播到编码器的参数。我们采用 **重参数化（reparameterization）** 技巧，将随机采样过程表示为一个可微函数：

$$
\boldsymbol{z}=\boldsymbol{\mu}_{\phi}(\boldsymbol{x})+
\boldsymbol{\sigma}_{\phi}(\boldsymbol{x})\odot \boldsymbol{\epsilon},~~~~~~~~~~\boldsymbol{\epsilon}\sim \mathcal{N}(\boldsymbol{0},\boldsymbol{I})
$$

- $\boldsymbol{\mu}_{\phi}(\boldsymbol{x})$ 和 $\boldsymbol{\sigma}_{\phi}(\boldsymbol{x})$ 是编码器输出的均值和方差。
- $\boldsymbol{\epsilon}$ 是独立于编码器参数的标准正态噪声。

通过这种方式，重构项和正则项的梯度都可以顺利回传到编码器，从而实现端到端训练。因此，重参数化不仅保证了 VAE 可以同时优化重构项和正则项，也解决了随机采样导致的梯度不可导问题，是 VAE 训练不可或缺的一步。

如下示例，基于 MNIST 手写数字数据集图像训练简单的 VAE 生成模型。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------------
# 超参数
# -------------------------------
batch_size = 128
latent_dim = 20   # 潜在变量维度
epochs = 200
learning_rate = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# -------------------------------
# 数据集
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1)) 
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# -------------------------------
# VAE 模型
# -------------------------------
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        # 编码器
        self.fc1 = nn.Linear(28*28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        # 解码器
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)
    
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        # 重参数化采样
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        x_hat = torch.sigmoid(self.fc4(h3))  # 输出 0~1
        return x_hat
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

# -------------------------------
# 损失函数：ELBO
# -------------------------------
def loss_function(x_hat, x, mu, logvar):
    # 重构损失（Binary Cross Entropy）
    recon_loss = F.binary_cross_entropy(x_hat, x, reduction='sum')
    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# -------------------------------
# 模型和优化器
# -------------------------------
model = VAE(latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------------------------------
# 训练
# -------------------------------
model.train()
for epoch in range(1, epochs+1):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 28*28).to(device)
        optimizer.zero_grad()
        x_hat, mu, logvar = model(data)
        loss = loss_function(x_hat, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {train_loss/len(train_loader.dataset):.4f}")

# -------------------------------
# 生成新数字样本
# -------------------------------
model.eval()
with torch.no_grad():
    z = torch.randn(64, latent_dim).to(device)  # 从标准正态采样
    sample = model.decode(z).cpu()
    sample = sample.view(64, 1, 28, 28)

# 显示生成结果
fig, axes = plt.subplots(8, 8, figsize=(8,8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(sample[i].squeeze(), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

以下是训练完成后，随机采样了 64 组潜在变量 $\boldsymbol{z}$ 生成的图像：

![vae_result-2025-08-21-16-12-30](https://ozzyc.oss-cn-shenzhen.aliyuncs.com/NotePicture/vae_result-2025-08-21-16-12-30.png)
