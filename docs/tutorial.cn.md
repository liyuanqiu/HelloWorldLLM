# 从零理解一个最小语言模型：Hello World LLM 教学文档

> **目标读者**：会写 CRUD 代码（前端/后端/全栈），但对机器学习/神经网络/LLM 完全没有概念的程序员。
>
> **我们要做什么**：逐行拆解一个用纯 TypeScript 手写的、约 500 行的字符级语言模型（Character-Level Language Model）。不依赖任何 ML 框架，所有数学都是 `for` 循环 + 加减乘除。

---

## 目录

1. [语言模型到底在干什么？](#1-语言模型到底在干什么)
2. [字符级 Tokenizer：把文字变成数字](#2-字符级-tokenizer把文字变成数字)
3. [Embedding：给每个字符一个「身份证」](#3-embedding给每个字符一个身份证)
4. [前向传播：从输入到预测](#4-前向传播从输入到预测)
   - 4.1 [矩阵乘法：一堆乘法加法的打包](#41-矩阵乘法一堆乘法加法的打包)
   - 4.2 [激活函数 tanh：让模型能学「弯曲」的规律](#42-激活函数-tanh让模型能学弯曲的规律)
   - 4.3 [Softmax：把分数变成概率](#43-softmax把分数变成概率)
5. [Loss 函数：模型猜得有多差？](#5-loss-函数模型猜得有多差)
6. [反向传播：告诉每个参数该怎么调](#6-反向传播告诉每个参数该怎么调)
   - 6.1 [导数/梯度：函数的「坡度」](#61-导数梯度函数的坡度)
   - 6.2 [链式法则：一环扣一环](#62-链式法则一环扣一环)
   - 6.3 [代码中的反向传播逐层拆解](#63-代码中的反向传播逐层拆解)
7. [优化器 SGD：顺着坡往下走](#7-优化器-sgd顺着坡往下走)
8. [训练循环：反复练习](#8-训练循环反复练习)
9. [文本生成：让模型说话](#9-文本生成让模型说话)
10. [从这个玩具到真正的 LLM：差在哪里？](#10-从这个玩具到真正的-llm差在哪里)

---

## 1. 语言模型到底在干什么？

你可以把语言模型想象成一个**超级自动补全**。

手机输入法打了「今天天」，它会建议「气」——这就是一个最简单的语言模型：**给定前面的文字，预测下一个字（或字符）的概率**。

```
输入：  "To be, or not to b"
模型输出：
  'e' → 92%    ← 最可能
  'a' →  3%
  'o' →  1%
  ... 其余字符分享剩下的 4%
```

ChatGPT、Claude 做的事情本质上也是这个——只不过它们的「字符」换成了更大的 token（词片段），模型规模大了几百万倍，并且经过了对话微调。但核心逻辑是一样的：**看前文，猜下文**。

我们这个 Hello World LLM 做的事情：
- 训练数据：两段莎士比亚独白（约 1900 个字符）
- 学习目标：看前 16 个字符，预测第 17 个字符
- 训练完成后：给它一个开头，它能「续写」出莎士比亚风格的文本（虽然会有不少错别词，毕竟模型很小）

---

## 2. 字符级 Tokenizer：把文字变成数字

计算机做数学运算只能处理数字，不能直接处理 'a'、'b' 这样的字符。所以第一步是建一个**字符与数字的映射表**。

```typescript
const chars = Array.from(new Set(TRAINING_TEXT)).sort();
// chars = ['\n', ' ', "'", ',', '.', ':', ';', '?', 'A', 'B', ... 'z']

const stoi: Record<string, number> = {}; // string to integer
const itos: Record<number, string> = {}; // integer to string
chars.forEach((ch, i) => { stoi[ch] = i; itos[i] = ch; });
```

这跟数据库里建一张 `char_id → char` 的查找表没什么区别：

| char_id | char |
|---------|------|
| 0       | `\n` |
| 1       | ` `  |
| 2       | `'`  |
| 3       | `,`  |
| ...     | ...  |
| 42      | `z`  |

`encode` 把字符串变成数字数组，`decode` 反过来：

```typescript
encode("To be") → [20, 31, 1, 18, 21]
decode([20, 31, 1, 18, 21]) → "To be"
```

**词表大小（VOCAB_SIZE）**= 训练文本中不重复字符的数量，这里大约 43 个。

> **和真正的 LLM 的区别**：GPT 用的不是单个字符，而是 BPE（Byte Pair Encoding）算法把常见的字符组合合并成 token。比如 "the" 可能是一个 token，"tion" 可能是一个 token。这样词表大小通常在 32K-100K 之间，平均一个 token 代表几个字符，效率更高。

---

## 3. Embedding：给每个字符一个「身份证」

现在每个字符有了一个编号（0, 1, 2, ...），但一个整数包含的信息太少了。我们希望用一个**多维向量**来表示每个字符，让模型有更多的「维度」来学习字符之间的关系。

### 什么是向量？

如果你写过前端 CSS，`rgb(255, 128, 0)` 就是一个 3 维向量，用 3 个数字描述一种颜色。

Embedding 的想法一样：用 `N_EMBD = 32` 个数字来描述一个字符。刚开始这 32 个数字是随机的，但训练过程中，它们会**自动调整**，使得：
- 相似的字符（比如元音 a, e, i, o, u）在 32 维空间中靠得更近
- 不同的字符离得更远

```typescript
// Token Embedding 表：VOCAB_SIZE × N_EMBD 的大矩阵
// 每一行是一个字符的 32 维「身份证」
const tokEmb = randn(VOCAB_SIZE * N_EMBD, 0.02);

// Position Embedding 表：BLOCK_SIZE × N_EMBD
// 每一行代表「这是上下文窗口中第几个位置」的信息
const posEmb = randn(BLOCK_SIZE * N_EMBD, 0.02);
```

**为什么还需要 Position Embedding？**

想象你在读 "ab" 和 "ba"——同样的两个字符，但顺序不同，意思完全不同。如果我们只用字符的 embedding，模型无法区分顺序。Position Embedding 给每个位置一个独特的向量，加到字符 embedding 上，模型就知道「这个 'a' 出现在第 1 个位置」和「这个 'a' 出现在第 5 个位置」是不同的。

代码中是这样做的：

```
位置 t 上字符 c 的最终表示 = tokEmb[c] + posEmb[t]
```

然后把 16 个位置的向量拼接成一个 16 × 32 = 512 维的大向量，这就是输入给下一层的数据。

> **类比**：想象一个学生信息系统。`tokEmb` 像是「学生的个人特征」（身高、体重、年龄...），`posEmb` 像是「座位号附加信息」。同一个学生坐在不同座位上，最终的描述是「个人特征 + 座位特征」。

---

## 4. 前向传播：从输入到预测

前向传播就是**从输入计算到输出的过程**——数据从左到右流过模型的每一层。可以类比为一个多步骤的数据处理管道（pipeline）：

```
[16个字符] → Embedding → 拼接成512维向量 → 隐藏层(512→128) → tanh → 输出层(128→43) → Softmax → 概率
```

### 4.1 矩阵乘法：一堆乘法加法的打包

这是神经网络最核心的运算。别被「矩阵」吓到，它就是一种批量做乘法和加法的方式。

**从一个具体例子开始**：假设你做一个「菜品推荐系统」，有 3 个特征（辣度、甜度、咸度），想计算 2 个人的偏好分数：

```
用户特征：  [辣度=0.8, 甜度=0.3, 咸度=0.5]

人物A的口味权重：辣×0.9 + 甜×0.1 + 咸×0.4
人物B的口味权重：辣×0.2 + 甜×0.8 + 咸×0.3
```

计算过程：
```
人物A的分数 = 0.8×0.9 + 0.3×0.1 + 0.5×0.4 = 0.72 + 0.03 + 0.20 = 0.95
人物B的分数 = 0.8×0.2 + 0.3×0.8 + 0.5×0.3 = 0.16 + 0.24 + 0.15 = 0.55
```

这就是一次矩阵乘法！「输入向量 × 权重矩阵 = 输出向量」——每个输出元素都是输入的所有元素分别乘以对应权重再加起来。

在我们的模型中：

```typescript
// 隐藏层：512 维输入 → 128 维输出
// W1 是一个 512×128 的权重矩阵（512×128 = 65536 个数字）
// b1 是一个 128 维的偏置向量
for (let h = 0; h < HIDDEN_DIM; h++) {
  let sum = b1[h];  // 偏置：一个基础值
  for (let i = 0; i < INPUT_DIM; i++) {
    sum += flat[b * INPUT_DIM + i] * W1[i * HIDDEN_DIM + h];
    //     ^^^^^^^^输入第i维^^^^^   ^^^^^^^^^^权重^^^^^^^^^^
  }
  hidden[b * HIDDEN_DIM + h] = Math.tanh(sum);
}
```

**偏置（bias）b1** 就是加了一个常数项，类比线性方程 `y = kx + b` 里的 `b`。没有它，所有直线都必须过原点，表达能力受限。

### 4.2 激活函数 tanh：让模型能学「弯曲」的规律

如果只有矩阵乘法（本质是线性变换），无论你叠多少层，最终效果等价于一个单层——因为**线性函数的组合还是线性函数**。

类比：`y = 3 × (2 × x)` 等价于 `y = 6x`，叠了两层但没有用。

`tanh`（双曲正切）函数把任何数字压到 -1 到 1 之间：

```
tanh(-100) ≈ -1
tanh(0)    = 0
tanh(100)  ≈ 1
tanh(0.5)  ≈ 0.46
```

它的形状像一个 S 型曲线。加了这个「弯曲」之后，模型就能学习非线性的模式了——比如「在此字符后，某些字符的概率突然升高」这种突变模式。

> **类比**：想象你在调音响的均衡器。没有 tanh 的线性模型就像只能统一调高调低所有频段；加了 tanh 后，就可以做出「低频增强、高频衰减」这种非线性的曲线。

### 4.3 Softmax：把分数变成概率

经过隐藏层和输出层后，我们得到 VOCAB_SIZE（43）个原始分数（叫做 **logits**）。比如：

```
logits = [2.1, -0.5, 0.3, 1.8, -1.2, ...]  （43个数字）
```

但这些分数有正有负，加起来也不等于 1，不能直接当概率。Softmax 函数把它们转换为**合法的概率分布**（全部为正，加起来等于 1）：

$$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

如果你没见过 $e$ —— 它是一个数学常数，约等于 2.718。$e^x$ 就是「2.718 的 x 次方」。它有个好性质：永远大于 0，而且 x 越大，$e^x$ 增长越快。

**用具体数字走一遍**：

```
logits = [2.0, 1.0, 0.1]

步骤 1：取 e 的各次方
  e^2.0 = 7.389
  e^1.0 = 2.718
  e^0.1 = 1.105

步骤 2：加起来
  总和 = 7.389 + 2.718 + 1.105 = 11.212

步骤 3：各自除以总和
  概率 = [7.389/11.212, 2.718/11.212, 1.105/11.212]
       = [0.659, 0.242, 0.099]
       （加起来 = 1.0 ✓）
```

**数值稳定性技巧**：代码中先减去了最大值 `maxVal`，这是因为 $e^{1000}$ 会溢出（变成 Infinity），但 $e^0$ 不会。数学上可以证明 `softmax(x - max)` 和 `softmax(x)` 结果完全一样。

```typescript
// 先找最大值
let maxVal = -Infinity;
for (let v = 0; v < VOCAB_SIZE; v++)
  maxVal = Math.max(maxVal, logits[b * VOCAB_SIZE + v]);

// 减去最大值再取 exp —— 防止溢出
let sumExp = 0;
for (let v = 0; v < VOCAB_SIZE; v++) {
  probs[b * VOCAB_SIZE + v] = Math.exp(logits[b * VOCAB_SIZE + v] - maxVal);
  sumExp += probs[b * VOCAB_SIZE + v];
}
// 归一化
for (let v = 0; v < VOCAB_SIZE; v++)
  probs[b * VOCAB_SIZE + v] /= sumExp;
```

---

## 5. Loss 函数：模型猜得有多差？

模型给出了 43 个字符的概率分布，但真正的下一个字符只有一个。**Loss（损失）**就是衡量模型的预测有多「差」的指标。

我们用的是**交叉熵损失（Cross-Entropy Loss）**，它非常简单：

$$\text{Loss} = -\log(\text{模型给正确答案的概率})$$

举例子：

| 情况 | 正确答案的概率 | Loss |
|------|--------------|------|
| 模型很确定，给了 90% | -log(0.9) = 0.105 | 很低（好） |
| 模型不确定，给了 50% | -log(0.5) = 0.693 | 中等 |
| 模型猜错，只给了 1% | -log(0.01) = 4.605 | 很高（差） |

**为什么用 log？**

- 如果概率接近 1（模型猜对了），-log(1) = 0，惩罚为零
- 如果概率接近 0（模型完全猜错了），-log(0) → ∞，惩罚无穷大
- 这比简单的「正确概率的相反数」(1 - p) 惩罚力度更大，能更有效地推动模型学习

```typescript
function computeLoss(cache: ForwardCache): number {
  let totalLoss = 0;
  for (let b = 0; b < cache.B; b++) {
    // 找到正确答案对应的概率，取 -log
    totalLoss -= Math.log(cache.probs[b * VOCAB_SIZE + cache.targets[b]] + 1e-10);
    //                                                                    ^^^^^^
    //                                              加一个极小值防止 log(0) = -Infinity
  }
  return totalLoss / cache.B;  // 取平均
}
```

训练的目标就是让这个 Loss 越来越小。

---

## 6. 反向传播：告诉每个参数该怎么调

这是整个模型最核心，也是最难理解的部分。但别怕，我们从最基础的概念开始。

### 6.1 导数/梯度：函数的「坡度」

想象你站在一座山上，想走到山谷最低点。你不需要看到全局地图，只需要感受**脚下的坡度**：哪个方向是往下的，就往那边走。

**导数就是坡度**。对于一个函数 $f(x)$，导数 $f'(x)$ 告诉你：

> 如果 x 增加一点点，f(x) 会增加多少？

举例：$f(x) = x^2$
- 在 $x = 3$ 时，导数 = $2 \times 3 = 6$，意思是 x 每增加 1，f(x) 大约增加 6
- 在 $x = -2$ 时，导数 = $2 \times (-2) = -4$，意思是 x 每增加 1，f(x) 反而减少 4

**梯度（Gradient）**就是多维版本的导数。如果函数有很多个输入参数，梯度就是对每个参数分别求导数，得到一个向量，告诉你**每个参数分别朝哪个方向调整能让函数减小最快**。

我们的模型有 ~44,000 个参数（所有 embedding、W1、b1、W2、b2 加起来）。我们需要知道：**如果微调每一个参数，Loss 会怎么变化？** 这就是反向传播要计算的东西。

### 6.2 链式法则：一环扣一环

我们的模型是一个多层的计算链：

```
输入 → embedding → 隐藏层 → tanh → 输出层 → softmax → loss
```

**链式法则（Chain Rule）**说的是：如果 $y = f(g(x))$，那么 $y$ 对 $x$ 的导数 = $f'$ 对 $g$ 的导数 × $g$ 对 $x$ 的导数。

用生活例子理解：

> 汇率：1 美元 = 7 人民币 = 0.9 欧元
>
> 问：1 欧元 = 多少人民币？
>
> 答：先「欧元→美元」乘以 1/0.9，再「美元→人民币」乘以 7 = 7/0.9 ≈ 7.78
>
> 这就是链式法则：**把每一步的变化率乘起来**。

在反向传播中：
1. 先算 Loss 对 softmax 输出的导数（最后一步的坡度）
2. 再乘以 softmax 对 logits 的导数（倒数第二步的坡度）
3. 再乘以 logits 对 W2 的导数（继续往回推）
4. ...一直推到最开始的 embedding

这就是「反向」传播——从最后一层往第一层反推。

### 6.3 代码中的反向传播逐层拆解

#### 第一步：Softmax + Cross-Entropy 的梯度

这里有一个著名的简化结果——softmax + 交叉熵组合之后，梯度的公式极其优美：

$$\frac{\partial \text{Loss}}{\partial \text{logits}_i} = \text{probs}_i - \mathbb{1}[i = \text{target}]$$

翻译成大白话：**梯度 = 模型给出的概率 - 正确答案**。

如果正确答案是字符 5：
```
probs   = [0.1, 0.05, 0.02, 0.03, 0.1, 0.6, ...]  ← 模型的预测
one_hot = [0,   0,    0,    0,    0,   1,   ...]  ← 正确答案（只有位置5是1）
梯度     = [0.1, 0.05, 0.02, 0.03, 0.1, -0.4, ...]  ← 差值
```

对应代码：

```typescript
const dLogits = new Float64Array(B * VOCAB_SIZE);
for (let b = 0; b < B; b++) {
  for (let v = 0; v < VOCAB_SIZE; v++)
    dLogits[b * VOCAB_SIZE + v] = c.probs[b * VOCAB_SIZE + v] / B;
  dLogits[b * VOCAB_SIZE + c.targets[b]] -= 1.0 / B;
  // 除以 B 是因为 loss 取了 batch 的平均值
}
```

#### 第二步：输出层（W2, b2）的梯度

数学上，`logits = hidden × W2 + b2`，所以：

$$\frac{\partial \text{Loss}}{\partial W2} = \text{hidden}^T \times d\text{Logits}$$

$$\frac{\partial \text{Loss}}{\partial b2} = \text{sum}(d\text{Logits}, \text{按 batch 维度})$$

直觉理解：哪个 hidden neuron 的值大，它对应的权重就需要调整更多（因为它的影响力大）。

```typescript
// dW2[h][v] = 所有 batch 上 hidden[h] × dLogits[v] 的求和
const dW2 = new Float64Array(HIDDEN_DIM * VOCAB_SIZE);
for (let h = 0; h < HIDDEN_DIM; h++) {
  for (let v = 0; v < VOCAB_SIZE; v++) {
    let s = 0;
    for (let b = 0; b < B; b++)
      s += c.hidden[b * HIDDEN_DIM + h] * dLogits[b * VOCAB_SIZE + v];
    dW2[h * VOCAB_SIZE + v] = s;
  }
}
```

#### 第三步：把梯度传播过 tanh

tanh 的导数有一个很方便的性质：

$$\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$$

也就是说，如果你已经知道 `tanh(x)` 的值（我们在前向传播中已经算出来存在 `hidden` 里了），直接用 `1 - hidden²` 就能得到导数，不需要重新计算。

```typescript
const dHiddenPre = new Float64Array(B * HIDDEN_DIM);
for (let i = 0; i < B * HIDDEN_DIM; i++) {
  const t = c.hidden[i];         // 已经是 tanh 之后的值
  dHiddenPre[i] = dHidden[i] * (1 - t * t);  // 乘以 tanh 的导数
}
```

#### 第四步：隐藏层（W1, b1）的梯度

跟第二步几乎一样的结构，只是维度不同：

```typescript
// dW1 = flat^T × dHiddenPre
const dW1 = new Float64Array(INPUT_DIM * HIDDEN_DIM);
for (let i = 0; i < INPUT_DIM; i++) {
  for (let h = 0; h < HIDDEN_DIM; h++) {
    let s = 0;
    for (let b = 0; b < B; b++)
      s += c.flat[b * INPUT_DIM + i] * dHiddenPre[b * HIDDEN_DIM + h];
    dW1[i * HIDDEN_DIM + h] = s;
  }
}
```

#### 第五步：Embedding 的梯度

Embedding 不是矩阵乘法，而是查表（lookup）。所以梯度传播不是矩阵运算，而是**把梯度累加回对应的位置**：

```typescript
for (let b = 0; b < B; b++) {
  for (let t = 0; t < BLOCK_SIZE; t++) {
    const tokIdx = c.inputs[b][t];
    for (let d = 0; d < N_EMBD; d++) {
      const g = dFlat[b * INPUT_DIM + t * N_EMBD + d];
      dTokEmb[tokIdx * N_EMBD + d] += g;  // 累加！同一个字符出现多次，梯度要加起来
      dPosEmb[t * N_EMBD + d] += g;
    }
  }
}
```

---

## 7. 优化器 SGD：顺着坡往下走

有了梯度（每个参数的坡度），接下来就是**往坡度相反的方向走一小步**——因为我们要让 Loss 减小，而梯度指向 Loss 增大的方向。

$$\text{参数}_{\text{new}} = \text{参数}_{\text{old}} - \text{学习率} \times \text{梯度}$$

这就是 **SGD（Stochastic Gradient Descent，随机梯度下降）**，「随机」是因为每次只用一小批数据（batch）而不是全部数据来估算梯度。

```typescript
function sgdUpdate(g: Grads, lr: number): void {
  for (let i = 0; i < tokEmb.length; i++) tokEmb[i] -= lr * g.dTokEmb[i];
  for (let i = 0; i < W1.length; i++)     W1[i]     -= lr * g.dW1[i];
  // ... 对每组参数都这样做
}
```

**学习率（Learning Rate）** 就是步长。太大会「跨过」山谷来回震荡，太小会走得极慢。代码中用了 **Cosine Schedule**：

```
学习率随训练进度呈余弦曲线变化：
- 前 200 步：从 0 线性增加到最大值 0.05（Warmup，慢启动）
- 之后：余弦衰减到最小值 0.0005
```

直觉：一开始参数还是随机的，大步走没关系；后期模型已经学得差不多了，需要小步微调。

### 梯度裁剪（Gradient Clipping）

有时候梯度会突然变得很大（「梯度爆炸」），一步走太远把模型搞崩了。梯度裁剪就是设一个上限：如果梯度的总长度超过 `GRAD_CLIP = 5.0`，就把整个梯度等比例缩小到长度 5.0。

```typescript
function clipGrads(g: Grads, maxNorm: number): void {
  // 计算梯度的总长度（L2 范数）
  let normSq = 0;
  for (const arr of arrays)
    for (let i = 0; i < arr.length; i++) normSq += arr[i] * arr[i];
  const norm = Math.sqrt(normSq);

  // 超过上限就等比缩小
  if (norm > maxNorm) {
    const scale = maxNorm / norm;
    for (const arr of arrays)
      for (let i = 0; i < arr.length; i++) arr[i] *= scale;
  }
}
```

---

## 8. 训练循环：反复练习

整个训练过程就像学生反复做练习题：

```
重复 5000 次：
  1. 随机抽一批训练数据（32 个「16字符→1字符」的例子）
  2. 前向传播：让模型猜答案 → 得到概率分布
  3. 计算 Loss：衡量猜得多差
  4. 反向传播：算出每个参数的梯度（该往哪个方向调）
  5. SGD 更新：微调所有参数
```

```typescript
for (let step = 0; step < NUM_STEPS; step++) {
  const lr = ...;  // 计算当前学习率
  const { inputs, targets } = sampleBatch();    // 1. 抽题
  const cache = forward(inputs, targets);        // 2. 做题
  const loss = computeLoss(cache);               // 3. 打分
  const grads = backward(cache);                 // 4. 分析错在哪
  clipGrads(grads, GRAD_CLIP);                   // 4.5 防止梯度爆炸
  sgdUpdate(grads, lr);                          // 5. 改进
}
```

随着训练进行，Loss 会逐渐下降：
```
Step     0 │ Loss: 3.7612   ← 一开始瞎猜，loss 很高
Step  1000 │ Loss: 2.1543   ← 学到了一些模式
Step  3000 │ Loss: 1.5234   ← 效果越来越好
Step  4999 │ Loss: 1.2876   ← 基本收敛
```

> **理论最低 Loss**：如果模型完美学会了训练数据中的字符分布，Loss 大约在 1.0-1.5 左右（因为自然语言本身就有不确定性——同一个前缀后面可以接多种合理的字符）。

---

## 9. 文本生成：让模型说话

训练完成后，我们可以用模型自回归（autoregressive）地生成文本。「自回归」的意思是：**模型生成一个字符，然后把这个字符加入输入，再预测下一个字符，循环往复**。

```
上下文窗口 (16字符)          → 模型预测 → 采样 → 新字符
["T","o"," ","b","e",...]  → probs    → "," → 加入上下文
["o"," ","b","e",",",...]  → probs    → " " → 加入上下文
[" ","b","e",","," ",...]  → probs    → "o" → 加入上下文
...
```

### Temperature（温度）采样

模型输出的是概率分布，我们需要从中**随机采样**一个字符。Temperature 控制了随机性：

```
原始 logits = [2.0, 1.0, 0.1]

Temperature = 1.0 → softmax([2.0, 1.0, 0.1])   = [0.659, 0.242, 0.099]  （正常）
Temperature = 0.5 → softmax([4.0, 2.0, 0.2])   = [0.843, 0.114, 0.043]  （更集中，更保守）
Temperature = 2.0 → softmax([1.0, 0.5, 0.05])  = [0.414, 0.251, 0.335]  （更均匀，更随机）
```

原理：在 softmax 之前把 logits 除以 temperature：
- **Temperature < 1**：等效于放大差距，让高概率的字符更突出（生成更「安全」但无聊的文本）
- **Temperature > 1**：等效于缩小差距，让所有字符的概率更接近（生成更「创意」但可能乱码的文本）
- **Temperature → 0**：极端情况，永远选概率最高的字符（贪心解码）

代码中用 Temperature = 0.8，偏保守一点：

```typescript
// 除以 temperature
for (let v = 0; v < VOCAB_SIZE; v++)
  scaled[v] = cache.logits[v] / temperature;

// softmax
// ...

// 从概率分布中随机采样
const r = Math.random();  // [0, 1) 之间的随机数
let cum = 0;
let next = 0;
for (let v = 0; v < VOCAB_SIZE; v++) {
  cum += scaled[v];       // 累积概率
  if (cum >= r) { next = v; break; }  // 落在哪个区间就选哪个字符
}
```

这个采样方法就像一个转盘抽奖——概率越高的字符占的扇形面积越大，被抽中的机会就越大。

---

## 10. 从这个玩具到真正的 LLM：差在哪里？

我们这个模型 ~44K 参数，GPT-4 据估计有上万亿参数。但核心思想是一样的。主要差距：

| 方面 | 我们的 Hello World | 真正的 LLM（如 GPT-4） |
|------|-------------------|----------------------|
| **核心结构** | 单隐藏层 MLP | Transformer（多层 Self-Attention） |
| **关键创新** | 无 | Self-Attention：每个位置能「看到」所有其他位置 |
| **Tokenizer** | 单字符（~43 tokens） | BPE/SentencePiece（32K-100K tokens） |
| **参数量** | ~44,000 | 数十亿到数万亿 |
| **训练数据** | 1,900 字符的莎士比亚 | 数万亿 token 的互联网文本 |
| **上下文窗口** | 16 字符 | 数千到数十万 token |
| **优化器** | SGD | AdamW（带动量和自适应学习率） |
| **硬件** | CPU，几秒搞定 | 数千 GPU，训练数周到数月 |
| **激活函数** | tanh | GeLU / SwiGLU |
| **额外技术** | 无 | Layer Norm、Dropout、KV Cache、RLHF/DPO... |

**最关键的区别是 Self-Attention**：

我们的模型把 16 个字符的 embedding 暴力拼接成一个向量，每个位置的信息是固定的。而 Transformer 的 Self-Attention 让每个位置动态地「关注」其他位置，决定从哪里获取信息。这就是 "Attention Is All You Need" 那篇论文的核心贡献。

但即使是 Transformer，它训练的本质过程跟我们一模一样：

1. 看一段文本，遮住最后一个 token
2. 前向传播，预测被遮住的 token
3. 计算 loss
4. 反向传播、更新参数
5. 重复，直到 loss 足够低

你现在理解了这个 500 行的 Hello World，就理解了 LLM 的骨架。剩下的只是把每个组件换成更强的版本、堆更多的层、喂更多的数据。

---

## 附录：关键数学符号速查

| 符号 | 含义 | 类比 |
|------|------|------|
| $x$ | 输入 | 函数的参数 |
| $W$ | 权重矩阵 | 每条连接的「强度系数」 |
| $b$ | 偏置 | `y = kx + b` 中的 b |
| $\tanh(x)$ | 激活函数 | 把数字压到 [-1, 1] |
| $e^x$ | 指数函数 | 2.718 的 x 次方 |
| $\log(x)$ | 自然对数 | $e^x$ 的逆运算，$\log(e^x) = x$ |
| $\frac{\partial L}{\partial W}$ | 偏导数/梯度 | Loss 对 W 的坡度 |
| $\sum$ | 求和 | for 循环 + 累加 |
| $\prod$ | 连乘 | for 循环 + 累乘 |
| $\mathbb{1}[\text{条件}]$ | 指示函数 | 条件为真返回 1，否则返回 0 |
