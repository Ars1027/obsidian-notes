---
title: Lecture 6 - Variational Autoencoder II
aliases:
  - VAE
  - 变分自编码器
tags:
  - Generative-Models
  - VAE
  - Latent-Variable-Model
  - Variational-Inference
---

# Lecture 6：Variational Autoencoder II

## 0. 这节课到底在讲什么？

上一节课解决的问题是：

> 引入潜变量 $z$ 后，真实的似然 $p_\theta(x)$ 很难计算，怎么办？

因为：

$$
p_\theta(x)
=
\int p(z)p_\theta(x|z)\,dz
$$

这个积分通常很难直接求。

于是上一节引入一个近似分布 $q(z)$，再利用 Jensen 不等式得到：

$$
\log p_\theta(x)\geq \mathrm{ELBO}
$$

因此不再直接优化难算的 $\log p_\theta(x)$，而是优化它的下界 ELBO。

---

而这一节进一步解决：

> **这个 $q(z)$ 到底怎么得到？怎么真正做成一个可以训练的神经网络？**

最终得到：

$$
\boxed{\text{Variational Autoencoder, VAE}}
$$

整节课主线：

```text
上一节得到 ELBO
        ↓
不同 x 需要不同的 q(z|x)
        ↓
不能每个样本都单独优化
        ↓
训练一个共享的 Encoder
        ↓
qφ(z|x)
        ↓
形成 VAE
        ↓
Reconstruction + KL
        ↓
随机采样无法直接反向传播
        ↓
Reparameterization Trick
        ↓
z = μ + σε
        ↓
得到可训练的 VAE
        ↓
构造连续、规整的潜空间
        ↓
可以随机生成新样本
```

---

# 1. 从 Variational Inference 到 Amortized Inference

## 1.1 原来的问题

对于不同的数据：

$$
x_1,x_2,x_3,\dots
$$

真正的后验：

$$
p(z|x_1),\quad p(z|x_2),\quad p(z|x_3)
$$

当然也不同。

因此近似它们的：

$$
q(z|x_1),\quad q(z|x_2),\quad q(z|x_3)
$$

也应该不同。

如果对每一个样本 $x_i$ 都单独优化一组参数：

$$
\phi_i
$$

那么：

```text
x₁ → 优化很多步 → 得到 φ₁
x₂ → 优化很多步 → 得到 φ₂
x₃ → 优化很多步 → 得到 φ₃
...
```

数据量一大，计算成本非常高。

---

## 1.2 Amortized Inference：摊销推断

解决方法：

> 不要为每个样本单独做优化，而是训练一个共享的神经网络。

这个网络：

$$
\boxed{q_\phi(z|x)}
$$

输入任意 $x$，直接输出对应的近似后验。

```text
              ┌─────────────┐
x₁ ─────────→ │             │ → q(z|x₁)
x₂ ─────────→ │   Encoder   │ → q(z|x₂)
x₃ ─────────→ │             │ → q(z|x₃)
              └─────────────┘
```

这就是：

$$
\boxed{\text{Amortized Inference}}
$$

### 通俗理解

“摊销”的意思是：

> 与其每次遇到一道题都重新从头计算答案，
> 不如训练一个学生，以后看到题目就直接给答案。

而这个“学生”就是 Encoder。

---

# 2. VAE 的基本结构

VAE 可以表示成：

```text
          Encoder
         qφ(z|x)
x ─────────────────→ z
                      │
                      │
                      ↓
                   Decoder
                   pθ(x|z)
                      │
                      ↓
                     x'
```

因此：

$$
\boxed{
x
\xrightarrow{q_\phi(z|x)}
z
\xrightarrow{p_\theta(x|z)}
x'
}
$$

---

# 3. Encoder 和 Decoder 分别是什么？

## 3.1 Encoder

Encoder 对应：

$$
\boxed{q_\phi(z|x)}
$$

方向：

$$
x\rightarrow z
$$

含义：

> 已经观察到数据 $x$，推断它背后的潜变量 $z$。

例如：

```text
一张手写数字 7
       ↓
    Encoder
       ↓
潜在表示 z
```

Encoder 的参数是：

$$
\phi
$$

---

## 3.2 Decoder

Decoder 对应：

$$
\boxed{p_\theta(x|z)}
$$

方向：

$$
z\rightarrow x
$$

含义：

> 给定潜变量 $z$，生成数据 $x$。

例如：

```text
潜变量 z
   ↓
Decoder
   ↓
一张数字 7
```

Decoder 的参数是：

$$
\theta
$$

> [!note]
> 可以简单记：
>
> - Encoder：$x\rightarrow z$
> - Decoder：$z\rightarrow x$

---

# 4. VAE 与普通 Autoencoder 最大的区别

普通 Autoencoder 和 VAE 表面都长这样：

```text
x → Encoder → z → Decoder → x'
```

但二者最重要的区别在 Encoder。

---

## 4.1 普通 Autoencoder

普通 AE 的 Encoder 输出：

$$
\boxed{\text{一个确定的 }z}
$$

例如：

$$
z=(2.3,-1.7)
$$

也就是：

```text
x
↓
Encoder
↓
一个确定的点 z
```

---

## 4.2 Variational Autoencoder

VAE 的 Encoder 输出：

$$
\boxed{\text{一个概率分布}}
$$

通常假设：

$$
q_\phi(z|x)
=
\mathcal N(\mu,\sigma^2)
$$

因此 Encoder 实际输出的是：

$$
\boxed{\mu,\sigma}
$$

然后再从这个分布中采样：

$$
z\sim q_\phi(z|x)
$$

流程：

```text
             μ
x → Encoder ─┤
             σ
             ↓
      得到一个 Gaussian
             ↓
         随机采样 z
```

---

# 5. 为什么 VAE 不直接输出一个确定的 z？

这是理解 VAE 最关键的问题之一。

假设普通 AE 学到：

```text
数字 0 → z = -10
数字 1 → z = -5
数字 2 → z = 7
数字 3 → z = 20
```

Decoder 在训练时只见过：

```text
-10, -5, 7, 20
```

那么如果生成时随机给：

$$
z=2
$$

Decoder 可能从来没有见过这个位置。

于是可能生成毫无意义的结果。

---

## 5.1 普通 AE 的潜空间可能有“洞”

例如：

```text
●●●


             ●●


      ●


                         ●●●
```

只有少数位置被训练过。

潜空间大量位置：

> Decoder 根本不知道对应什么。

因此：

$$
\boxed{\text{普通 AE 擅长重构，但不一定擅长随机生成}}
$$

---

# 6. VAE 的核心思想：整理潜空间

VAE 不仅希望：

> 原始数据能够被重构。

还希望：

> 潜变量 $z$ 的分布比较规整。

于是人为规定一个简单的先验：

$$
\boxed{
p(z)=\mathcal N(0,I)
}
$$

也就是标准正态分布。

然后要求 Encoder 输出的：

$$
q_\phi(z|x)
$$

不要和：

$$
p(z)
$$

差得太远。

---

# 7. VAE 最重要的公式：ELBO

VAE 的 ELBO 可以写成：

$$
\boxed{
\mathcal L(x;\theta,\phi)
=
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{KL}
\left(
q_\phi(z|x)\|p(z)
\right)
}
$$

它其实就是两个目标：

$$
\boxed{
\text{Reconstruction}
-
\text{KL Regularization}
}
$$

---

# 8. 第一部分：Reconstruction

第一项：

$$
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
$$

作用：

> 希望经过 Encoder → Decoder 以后，可以把原始输入恢复出来。

流程：

```text
原始图片 x
     ↓
 Encoder
     ↓
     z
     ↓
 Decoder
     ↓
重构图片 x'
```

希望：

$$
x'\approx x
$$

例如：

```text
输入：7

Encoder → z → Decoder

输出：7
```

而不是：

```text
输入：7
输出：2
```

---

## 8.1 Reconstruction 的直觉

Reconstruction 在告诉模型：

> “你的潜变量 $z$ 必须保存足够多关于 $x$ 的信息。”

否则 Decoder 无法恢复原图。

---

# 9. 第二部分：KL Regularization

第二项：

$$
D_{KL}
\left(
q_\phi(z|x)\|p(z)
\right)
$$

其中：

$$
p(z)=\mathcal N(0,I)
$$

KL 的作用：

> 约束 Encoder 输出的潜变量分布，让它不要到处乱跑。

可以把它理解成：

$$
\boxed{\text{潜空间整理费}}
$$

---

# 10. Reconstruction 与 KL 在互相“拉扯”

可以把 VAE 想象成两个老师。

## 老师 A：Reconstruction

说：

> 你把数据怎么编码都可以，只要最后能够恢复原图。

目标：

$$
\text{重构越准确越好}
$$

---

## 老师 B：KL

说：

> 等一下，潜变量不能随便乱放，要尽量服从统一的标准正态分布。

目标：

$$
q_\phi(z|x)\approx p(z)
$$

---

所以 VAE 在平衡：

$$
\boxed{
\text{保存数据的信息}
+
\text{保持潜空间规整}
}
$$

---

# 11. 为什么两项缺一不可？

## 11.1 只有 Reconstruction

模型可能：

```text
样本 A → z 在很左边
样本 B → z 在很右边
样本 C → z 在很上面
```

只要能重构就行。

潜空间：

```text
A


                B


       C


                          D
```

问题：

> 中间大量区域没有训练数据。

因此随机采样可能得到垃圾结果。

---

## 11.2 只有 KL

如果只强调：

$$
q_\phi(z|x)\approx N(0,I)
$$

模型可能把所有数据全部挤在一起。

例如：

```text
0 1 2 3 4 5 6 7 8 9
全部混在同一团
```

这样潜变量 $z$ 没有保存足够的信息。

Decoder 也无法知道应该重构哪个数字。

---

## 11.3 Reconstruction + KL

理想情况：

```text
      0 0 0
    0 0

       1 1
     1 1

          2 2
        2 2
```

潜空间：

- 有结构
- 连续
- 比较平滑
- 又保留语义信息

因此才能进行随机生成。

---

# 12. VAE 的完整训练过程

对于一个训练样本：

$$
x^i
$$

---

## Step 1：输入数据

$$
x^i
$$

---

## Step 2：Encoder 得到分布

$$
q_\phi(z|x^i)
$$

通常输出：

$$
\mu(x^i),\sigma(x^i)
$$

---

## Step 3：从分布中采样

$$
z\sim q_\phi(z|x^i)
$$

---

## Step 4：Decoder 重构

$$
z
\rightarrow
p_\theta(x|z)
\rightarrow
x'
$$

并评价：

$$
\log p_\theta(x^i|z)
$$

---

## Step 5：计算 KL

$$
D_{KL}
(q_\phi(z|x)\|p(z))
$$

---

## Step 6：计算 ELBO

$$
\mathcal L
=
\text{Reconstruction}
-
\text{KL}
$$

---

## Step 7：更新参数

同时更新：

$$
\theta,\phi
$$

其中：

- $\phi$：Encoder 参数
- $\theta$：Decoder 参数

---

# 13. 一个新问题：随机采样怎么反向传播？

VAE 中间存在：

```text
x
↓
Encoder
↓
μ, σ
↓
随机采样 z
↓
Decoder
↓
Loss
```

问题：

> 神经网络依赖梯度下降训练，但是“随机采样”操作本身没法像普通函数一样直接求导。

因此：

$$
\boxed{\text{如何让梯度通过 sampling？}}
$$

这就引出了：

$$
\boxed{\text{Reparameterization Trick}}
$$

即重参数化技巧。

---

# 14. Reparameterization Trick

原来我们直接：

$$
z\sim\mathcal N(\mu,\sigma^2)
$$

现在改成：

$$
\epsilon\sim\mathcal N(0,I)
$$

然后：

$$
\boxed{
z=\mu+\sigma\epsilon
}
$$

这两种方法产生的 $z$ 的分布相同。

---

# 15. 为什么这样就能反向传播？

原来：

```text
μ, σ
 ↓
随机采样 z
 ↓
Decoder
```

随机操作直接挡在参数和 $z$ 中间。

---

重参数化以后：

```text
          ε ~ N(0,I)
              ↓
μ ───────→ μ + σε ←────── σ
              ↓
              z
              ↓
           Decoder
```

随机性全部集中到了：

$$
\epsilon
$$

而：

$$
z=\mu+\sigma\epsilon
$$

对于一次固定的 $\epsilon$ 来说，就是一个普通函数。

因此：

$$
\frac{\partial z}{\partial\mu}=1
$$

$$
\frac{\partial z}{\partial\sigma}=\epsilon
$$

梯度可以继续传回 Encoder。

---

## 15.1 重参数化一句话记忆

$$
\boxed{
z=\mu+\sigma\epsilon,\qquad
\epsilon\sim N(0,I)
}
$$

作用：

> **把随机性从网络参数的计算路径中搬出去，从而让梯度可以反向传播。**

---

# 16. VAE 如何生成一个全新的样本？

训练阶段：

```text
x
↓
Encoder
↓
qφ(z|x)
↓
z
↓
Decoder
↓
x'
```

但真正生成新图片时：

> 我们没有输入图片 $x$。

这时候直接：

$$
z\sim p(z)
$$

因为：

$$
p(z)=N(0,I)
$$

然后：

```text
z ~ N(0,I)
      ↓
   Decoder
      ↓
   新样本 x
```

即：

$$
\boxed{
z\sim N(0,I)
\rightarrow
p_\theta(x|z)
\rightarrow
x
}
$$

---

# 17. 为什么 KL 对“生成”特别重要？

训练时 Decoder 接收到的 $z$ 来自：

$$
q_\phi(z|x)
$$

生成时 Decoder 接收到的 $z$ 来自：

$$
p(z)
$$

如果两者差别巨大：

$$
q_\phi(z|x)\not\approx p(z)
$$

那么训练和生成就会错位。

例如：

```text
训练时 z 都在 100~200

但生成时：
z ~ N(0,1)

抽出来：
-0.5
0.8
1.3
```

Decoder：

> 这些位置我训练的时候根本没见过。

因此 KL 要让：

$$
\boxed{
q_\phi(z|x)\approx p(z)
}
$$

这样生成阶段随机从 $p(z)$ 采样才有意义。

---

# 18. 普通 AE 与 VAE 对比

| 对比项 | Autoencoder | VAE |
|---|---|---|
| Encoder 输出 | 一个确定的 $z$ | 一个分布 $q(z|x)$ |
| 常见输出 | $z$ | $\mu,\sigma$ |
| 中间是否采样 | 否 | 是 |
| Loss | Reconstruction | Reconstruction + KL |
| 潜空间 | 可能有大量空洞 | 更连续、更规整 |
| 从随机 $z$ 生成 | 不可靠 | 可以 |
| 主要目标 | 压缩、重构 | 重构 + 生成 |

---

# 19. 从概率模型角度看 VAE

VAE 实际有三个非常重要的分布。

## Prior

$$
\boxed{p(z)}
$$

一般：

$$
p(z)=N(0,I)
$$

含义：

> 在不知道具体数据之前，潜变量通常是什么样。

---

## Approximate Posterior

$$
\boxed{q_\phi(z|x)}
$$

含义：

> 观察到 $x$ 后，它背后的 $z$ 可能是什么。

由 Encoder 实现。

---

## Likelihood / Decoder

$$
\boxed{p_\theta(x|z)}
$$

含义：

> 给定 $z$ 后，如何生成 $x$。

由 Decoder 实现。

---

# 20. 三个分布一定不要混淆

| 分布 | 作用 | 神经网络 |
|---|---|---|
| $p(z)$ | 潜变量先验 | 一般人为规定 |
| $q_\phi(z|x)$ | 从 $x$ 推断 $z$ | Encoder |
| $p_\theta(x|z)$ | 从 $z$ 生成 $x$ | Decoder |

可以记成：

```text
            qφ(z|x)
       x ───────────→ z
                       │
                       │ pθ(x|z)
                       ↓
                       x
```

---

# 21. 为什么 VAE 是生成模型？

普通 AE 更像：

> 压缩 → 解压。

而 VAE 通过 KL 把潜空间整理成接近：

$$
N(0,I)
$$

因此可以主动：

$$
z\sim N(0,I)
$$

然后：

$$
z\rightarrow Decoder\rightarrow x
$$

得到从来没有出现在训练集中的新样本。

所以：

$$
\boxed{
\text{VAE = Autoencoder + 概率潜空间}
}
$$

---

# 22. Generative Latent Space

VAE 希望潜空间不仅“能编码”，还应该：

- 连续
- 平滑
- 比较规整
- 邻近的 $z$ 对应相似的数据
- 从大部分合理的 $z$ 都能生成有意义的样本

例如二维潜空间：

```text
          8 8 8

      3 3     9 9

   2 2           7 7

      1 1     4 4

          0 0
```

如果从：

```text
数字 1 所在区域
```

慢慢移动到：

```text
数字 7 所在区域
```

生成的图像也应该逐渐变化。

这就是 VAE 潜空间“连续性”的直觉。

---

# 23. Disentangled Representation

理想情况下，希望潜变量 $z$ 的不同维度分别控制不同因素。

例如：

$$
z=(z_1,z_2,z_3,\dots)
$$

其中：

```text
z₁ → 年龄
z₂ → 是否微笑
z₃ → 眼镜
z₄ → 头部姿态
```

这样只修改：

$$
z_2
$$

应该只改变“微笑”，而不会同时改变：

- 年龄
- 发色
- 身份
- 姿态

这叫：

$$
\boxed{\text{Disentangled Representation}}
$$

即：

> 解耦表示。

---

# 24. β-VAE

普通 VAE：

$$
\mathcal L
=
\text{Reconstruction}
-
D_{KL}
$$

β-VAE 修改为：

$$
\boxed{
\mathcal L
=
\text{Reconstruction}
-
\beta D_{KL}
}
$$

其中：

$$
\beta>1
$$

时，相当于加强 KL 的作用。

---

## 24.1 β-VAE 的直觉

普通 VAE：

```text
重构老师     KL老师
   ↓           ↓
差不多同等重要
```

β-VAE：

```text
重构老师        KL老师
   ↓             ↓↓↓↓↓
             管得更严格
```

因此潜空间往往更加规整，更可能出现解耦表示。

但代价可能是：

> 对 $z$ 的限制越强，可用于保存重构信息的自由度越少。

---

# 25. Posterior Collapse

VAE 原本希望：

```text
x
↓
Encoder
↓
z
↓
Decoder
↓
x'
```

其中 $z$ 很重要。

但是如果 Decoder 太强，例如使用强大的 Autoregressive Decoder：

它可能发现：

> 我不使用 $z$，也可以把 $x$ 建模得很好。

于是：

```text
x → Encoder → z ──X──→ Decoder
                       ↑
                     忽略 z
```

这叫：

$$
\boxed{\text{Posterior Collapse}}
$$

---

## 25.1 Posterior Collapse 时发生什么？

可能出现：

$$
q_\phi(z|x)=p(z)
$$

于是：

$$
D_{KL}(q_\phi(z|x)\|p(z))=0
$$

看起来 KL 非常完美。

但其实：

> $z$ 已经完全不包含关于 $x$ 的信息。

例如：

```text
猫   → q(z|猫)
狗   → q(z|狗)
汽车 → q(z|汽车)
```

得到的分布几乎一样。

那么 $z$ 就没有语义意义了。

---

## 25.2 Posterior Collapse 的问题

会导致：

- 潜变量无法学习有意义的语义
- Representation Learning 失效
- 条件生成困难
- Encoder 变得近似没用

---

## 25.3 常见解决思路

课件提到：

- Weakening Decoder
- KL Annealing
- InfoVAE
- $\delta$-VAE
- Skip-VAE

核心思想都是：

> 想办法逼迫模型真正使用 $z$。

---

# 26. VQ-VAE

普通 VAE 的潜变量是连续的：

$$
z=(0.37,-1.26,0.55,\dots)
$$

但某些任务的表示天然更像离散符号，例如：

- 语言
- 语音
- 高层语义
- 推理
- 图像局部结构

因此 VQ-VAE 使用：

$$
\boxed{\text{Discrete Latent Code}}
$$

---

# 27. VQ-VAE 的直觉：一本“潜变量字典”

假设有一个 Codebook：

```text
e₁ → 某种特征
e₂ → 某种特征
e₃ → 某种特征
...
eₖ → 某种特征
```

Encoder 首先产生：

$$
E(x)
$$

然后寻找距离它最近的 embedding：

$$
e_k
$$

即：

$$
k
=
\arg\min_i
\|E(x)-e_i\|_2
$$

最终潜变量不是任意连续向量，而是：

$$
\boxed{\text{Codebook 中某个离散向量}}
$$

可以简单理解成：

> 把连续表示变成“查字典”。

---

# 28. VAE 整节课最终流程图

```text
训练样本 x
    │
    ▼
┌──────────────┐
│   Encoder    │
│ qφ(z | x)    │
└──────────────┘
    │
    │ 输出
    ▼
   μ, σ
    │
    │
ε ~ N(0,I)
    │
    ▼
z = μ + σε
    │
    ▼
┌──────────────┐
│   Decoder    │
│ pθ(x | z)    │
└──────────────┘
    │
    ▼
重构 x'
```

训练目标：

```text
         ELBO
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
Reconstruction   KL
     │           │
保证能恢复 x   整理潜空间
```

---

# 29. 生成阶段

训练完成后：

```text
z ~ N(0,I)
     │
     ▼
 Decoder
     │
     ▼
新样本 x
```

不再需要 Encoder。

因此：

> Encoder 主要负责训练时进行推断。

> Decoder 才是真正负责生成数据的生成模型。

---

# 30. 与上一节课完整连接

```text
【Latent Variable Models I】

观察数据 x
    ↓
假设存在潜变量 z
    ↓
p(x,z)=p(x|z)p(z)
    ↓
需要计算 p(x)
    ↓
p(x)=∫p(x,z)dz
    ↓
积分太难
    ↓
引入 q(z)
    ↓
Importance Sampling
    ↓
Jensen Inequality
    ↓
ELBO


===============================


【Variational Autoencoder II】

每个 x 都需要不同的 q
    ↓
不能每个样本单独优化
    ↓
Amortized Inference
    ↓
Encoder qφ(z|x)
    ↓
得到 VAE
    ↓
ELBO
    ↓
Reconstruction + KL
    ↓
需要从 qφ(z|x) 采样
    ↓
随机采样阻碍反向传播
    ↓
Reparameterization Trick
    ↓
z = μ + σε
    ↓
可以正常训练
    ↓
KL 整理潜空间
    ↓
z ~ N(0,I)
    ↓
Decoder
    ↓
生成新样本
```

---

# 31. 最需要掌握的六个公式

## ① Encoder

$$
\boxed{
q_\phi(z|x)
}
$$

作用：

> 从 $x$ 推断 $z$。

---

## ② Decoder

$$
\boxed{
p_\theta(x|z)
}
$$

作用：

> 从 $z$ 生成 $x$。

---

## ③ Prior

$$
\boxed{
p(z)=N(0,I)
}
$$

---

## ④ VAE ELBO

$$
\boxed{
\mathcal L
=
\mathbb E_{q_\phi(z|x)}
[\log p_\theta(x|z)]
-
D_{KL}(q_\phi(z|x)\|p(z))
}
$$

---

## ⑤ Reparameterization Trick

$$
\boxed{
\epsilon\sim N(0,I)
}
$$

$$
\boxed{
z=\mu+\sigma\epsilon
}
$$

---

## ⑥ Generation

$$
\boxed{
z\sim p(z)
\rightarrow
p_\theta(x|z)
\rightarrow
x
}
$$

---

# 32. 一句话区分几个核心概念

| 概念 | 一句话理解 |
|---|---|
| Latent Variable | 数据背后没有直接观察到的隐藏因素 |
| $p(z)$ | 潜变量的先验分布 |
| $q_\phi(z|x)$ | Encoder 根据 $x$ 猜 $z$ |
| $p_\theta(x|z)$ | Decoder 根据 $z$ 生成 $x$ |
| Reconstruction | 保证 $z$ 保留足够的信息 |
| KL | 保证潜空间规整 |
| ELBO | 原始 likelihood 太难算时使用的可优化下界 |
| Amortized Inference | 用一个共享 Encoder 代替每个样本单独优化 |
| Reparameterization | 把随机性搬到 $\epsilon$，从而允许反向传播 |
| β-VAE | 加强 KL，鼓励更加解耦的潜空间 |
| Posterior Collapse | Decoder 太强，导致模型不再使用 $z$ |
| VQ-VAE | 使用离散而非连续的潜变量 |

---

# 33. 考前速记版

> [!summary] VAE 核心
>
> **1.** 潜变量模型：
>
> $$
> p(x,z)=p(x|z)p(z)
> $$
>
> **2.** Encoder：
>
> $$
> q_\phi(z|x)
> $$
>
> **3.** Decoder：
>
> $$
> p_\theta(x|z)
> $$
>
> **4.** VAE Encoder 输出的是概率分布，一般输出 $\mu,\sigma$。
>
> **5.** ELBO：
>
> $$
> \text{Reconstruction}-KL
> $$
>
> **6.** Reconstruction 保证能恢复数据。
>
> **7.** KL 保证潜空间接近：
>
> $$
> N(0,I)
> $$
>
> **8.** 重参数化：
>
> $$
> z=\mu+\sigma\epsilon
> $$
>
> 解决随机采样无法方便反向传播的问题。
>
> **9.** 生成阶段：
>
> $$
> z\sim N(0,I)
> $$
>
> 然后直接送进 Decoder。
>
> **10.** Posterior Collapse：
>
> Decoder 太强，完全忽略 $z$。

---

# 34. 最重要的直觉

> [!tip]
> 不要先把 VAE 看成一大堆概率公式。
>
> 可以把它理解成：
>
> **一个“会整理潜空间的 Autoencoder”。**

普通 Autoencoder：

> 只管“存进去以后能不能找回来”。

VAE：

> 不仅要求能够找回来，
> 还要求整个潜空间按照统一规则整理好。

因此 VAE 的潜空间通常更适合：

- 插值
- 随机采样
- 表示学习
- 生成新数据

最终可以概括为：

$$
\boxed{
\text{VAE}
=
\text{Autoencoder 的重构能力}
+
\text{概率潜空间的生成能力}
}
$$