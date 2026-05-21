# Transformer 深入讲解

> 理解 Transformer 是理解 GPT、Claude 等 LLM 的基础，也是 Agent 开发面试的高频考点。

## 一、为什么需要 Transformer？

### RNN 的问题

```
RNN 处理序列：逐个 token 顺序处理
"The cat sat on the" → 每一步依赖上一步的输出

问题：
1. 长距离依赖丢失（开头的词对结尾的影响越来越弱）
2. 无法并行（必须等前一个 token 处理完才能处理下一个）
3. 训练慢（GPU 并行能力浪费）
```

### Transformer 的核心思想

```
Transformer：Self-Attention 一次看到所有 token

"The cat sat on the mat"
     ↓
每个 token 同时计算与所有其他 token 的关系
     ↓
并行处理，长距离依赖不丢失
```

## 二、Self-Attention（自注意力）

### 直觉理解

```
句子："The cat sat on the mat because it was tired"

Self-Attention 解决的问题： "it" 指的是谁？

计算 "it" 与每个词的注意力分数：
  it → The:  0.02
  it → cat:  0.45  ← 最高！所以 "it" 指的是 "cat"
  it → sat:  0.05
  it → on:   0.01
  it → the:  0.02
  it → mat:  0.15
  it → because: 0.03
  it → was:  0.12
  it → tired: 0.15
```

### 数学过程

```python
# 每个 token 生成三个向量：Query, Key, Value
# Q = "我在找什么"
# K = "我有什么特征"
# V = "我的实际内容"

# 对于句子 "The cat sat":
# 每个 token 的 embedding → 通过三个权重矩阵 → Q, K, V

import numpy as np

# 假设 embedding 维度 = 4，简化演示
embeddings = {
    "The": np.array([1, 0, 1, 0]),
    "cat": np.array([0, 1, 0, 1]),
    "sat": np.array([1, 1, 0, 0]),
}

# 权重矩阵（训练得到）
W_Q = np.random.randn(4, 4) * 0.1
W_K = np.random.randn(4, 4) * 0.1
W_V = np.random.randn(4, 4) * 0.1

# 计算 Q, K, V
Q = {word: emb @ W_Q for word, emb in embeddings.items()}
K = {word: emb @ W_K for word, emb in embeddings.items()}
V = {word: emb @ W_V for word, emb in embeddings.items()}

# 注意力分数 = Q 和 K 的点积
# Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
#                       ↑                    ↑
#                       计算相关性            缩放（防止梯度消失）
```

### 缩放因子 sqrt(d_k) 的作用

```
为什么要除以 sqrt(d_k)？

假设 d_k = 64，Q 和 K 的点积值可能很大（如 20, -15, 30...）
softmax([20, -15, 30]) ≈ [0.0001, 0.0000, 0.9999]  → 几乎 one-hot，梯度消失

除以 sqrt(64)=8 后：[2.5, -1.875, 3.75]
softmax([2.5, -1.875, 3.75]) ≈ [0.15, 0.02, 0.83]  → 分布更平滑，梯度正常
```

## 三、Multi-Head Attention（多头注意力）

### 为什么需要多个头？

```
一个 attention head 只能关注一种关系：
- Head 1 可能关注语法关系（主语-谓语）
- Head 2 可能关注指代关系（it → cat）
- Head 3 可能关注位置关系（相邻词）

多头 = 同时关注多种关系
```

### 数学过程

```python
# 假设 d_model = 512，h = 8 个头
# 每个头的维度 = d_model / h = 64

# 多头注意力：
# 1. 把 Q, K, V 各分成 8 份（每份 64 维）
# 2. 每份独立做 attention
# 3. 拼接 8 个结果
# 4. 通过线性层投影回 d_model 维度

def multi_head_attention(Q, K, V, num_heads=8):
    d_model = Q.shape[-1]
    d_k = d_model // num_heads

    # 分头
    Q_heads = Q.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    K_heads = K.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)
    V_heads = V.reshape(batch, seq_len, num_heads, d_k).transpose(1, 2)

    # 每个头独立计算 attention
    attn_output = scaled_dot_product_attention(Q_heads, K_heads, V_heads)

    # 拼接 + 线性投影
    output = attn_output.transpose(1, 2).reshape(batch, seq_len, d_model)
    output = output @ W_O    # W_O: (d_model, d_model)
    return output
```

## 四、位置编码（Positional Encoding）

### 为什么需要位置编码？

```
Self-Attention 是"无序"的：
  "猫 吃 鱼" 和 "鱼 吃 猫" 在 attention 计算中完全一样！

但语义完全不同 → 需要注入位置信息
```

### 正弦位置编码

```python
import math

def positional_encoding(max_len, d_model):
    """原始 Transformer 的正弦位置编码"""
    PE = np.zeros((max_len, d_model))

    for pos in range(max_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            PE[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))

    return PE

# PE[pos, i] = sin(pos / 10000^(2i/d_model))
# PE[pos, i+1] = cos(pos / 10000^(2i/d_model))

# 为什么用 sin/cos？
# - 相对位置可以通过线性变换表示
# - PE(pos+k) 可以用 PE(pos) 的线性组合表示
# - 模型可以学到"距离"的概念
```

### 位置编码的叠加

```python
# 输入 = token embedding + position encoding
input_vector = token_embedding + positional_encoding[position]

# 不是拼接，是相加！
# 原因：拼接会增加维度，相加保持维度不变
# 实验表明相加效果和拼接差不多，但更节省参数
```

### RoPE（旋转位置编码）

```python
# GPT、Claude 等现代 LLM 用 RoPE（Rotary Position Embedding）
# 核心思想：通过旋转矩阵编码位置，让注意力分数自然包含相对位置信息

# RoPE 的优势：
# 1. 相对位置信息直接体现在 attention score 中
# 2. 外推性好（训练 1024 长度，推理 2048 也能用）
# 3. 不需要额外的 position embedding 参数

# 简化理解：
# 把 Q 和 K 的每两个维度看作一个二维向量
# 按位置旋转不同角度
# 两个向量的点积自然包含它们的角度差（即相对位置）
```

## 五、Encoder-Decoder 架构

### 原始 Transformer（Encoder-Decoder）

```
┌─────────────────────────────────────────┐
│              Transformer                │
│                                         │
│  ┌──────────────┐    ┌──────────────┐   │
│  │   Encoder     │    │   Decoder     │   │
│  │  (理解输入)    │───→│  (生成输出)    │   │
│  │               │    │               │   │
│  │ Self-Attention│    │ Masked Attn   │   │
│  │      × N      │    │ Cross-Attn    │   │
│  │               │    │      × N      │   │
│  └──────────────┘    └──────────────┘   │
│        ↑                    ↓            │
│     输入序列            输出序列          │
│  "I love AI"         "我 爱 人工智能"      │
└─────────────────────────────────────────┘
```

### 三种架构变体

```
1. Encoder-Decoder（原始 Transformer, T5, BART）
   - 适合：翻译、摘要等 seq2seq 任务
   - Encoder 理解输入，Decoder 生成输出

2. Encoder-only（BERT, RoBERTa）
   - 适合：分类、NER、问答等理解任务
   - 双向注意力（每个词能看到所有其他词）
   - 不能直接生成文本

3. Decoder-only（GPT, Claude, LLaMA）
   - 适合：文本生成、对话、Agent
   - 单向注意力（每个词只能看到前面的词）
   - 现代 LLM 的主流架构
```

### 为什么 Decoder-only 成为主流？

```
1. 统一范式：所有任务都可以转化为"给定上文，预测下文"
2. 涌现能力：规模够大时出现 in-context learning、推理等能力
3. 训练简单：只需要预测下一个 token，不需要复杂的预训练目标
4. Scaling Law：参数量和数据量增加时，性能稳定提升
```

## 六、Decoder 的 Masked Attention

### 为什么需要 Mask？

```
训练时，输入是完整句子："I love AI"
但 Decoder 生成时只能看到前面的词：

位置 1: 预测 "I"     → 看到: [_]
位置 2: 预测 "love"  → 看到: [I, _]
位置 3: 预测 "AI"    → 看到: [I, love, _]

Mask = 下三角矩阵，遮住未来的词
```

```python
# Causal Mask（因果掩码）
import torch

def create_causal_mask(seq_len):
    """下三角掩码，防止看到未来的 token"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# seq_len = 4 时：
# tensor([[0, -inf, -inf, -inf],
#         [0,    0, -inf, -inf],
#         [0,    0,    0, -inf],
#         [0,    0,    0,    0]])

# softmax 后 -inf 变成 0，未来的 token 不参与注意力计算
```

## 七、Feed-Forward Network（前馈网络）

```python
# 每个 Transformer 层除了 Attention，还有一个 FFN
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)    # 扩展
        self.linear2 = nn.Linear(d_ff, d_model)    # 压缩回去
        self.gelu = nn.GELU()                      # 激活函数

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))

# d_ff 通常是 d_model 的 4 倍
# d_model=768 → d_ff=3072

# FFN 的作用：
# Attention 负责"token 之间的关系"
# FFN 负责"单个 token 的特征变换"
# 两者互补
```

### GLU 变体（现代 LLM 常用）

```python
# GPT-3、LLaMA 等用 SwiGLU 替代 GELU
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

# SwiGLU 比 GELU 效果更好，但参数更多
# 实际 d_ff 会调小（如 d_ff = 8/3 * d_model）来保持总参数量
```

## 八、Layer Normalization

```python
# LayerNorm = 对每个 token 的特征维度做归一化
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

# 为什么用 LayerNorm 而不是 BatchNorm？
# BatchNorm 对 batch 维度归一化 → 依赖 batch 大小
# LayerNorm 对 feature 维度归一化 → 不依赖 batch，适合变长序列
```

### Pre-Norm vs Post-Norm

```
Post-Norm（原始 Transformer）：
  x → Attention → Add → LayerNorm → FFN → Add → LayerNorm

Pre-Norm（GPT-2、LLaMA 等现代模型）：
  x → LayerNorm → Attention → Add → LayerNorm → FFN → Add

Pre-Norm 优势：
- 训练更稳定（梯度流更顺畅）
- 不需要 warmup
- 可以用更大的学习率
```

## 九、完整 Transformer 层

```python
class TransformerBlock(nn.Module):
    """一个完整的 Transformer 层（Pre-Norm 风格）"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-Attention + 残差连接
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        # FFN + 残差连接
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x

# 残差连接（Residual Connection）的作用：
# 解决深层网络的梯度消失问题
# 梯度可以"跳过"复杂的层直接传回去
# x + f(x) → 梯度 = 1 + f'(x)，不会消失
```

## 十、LLM 的推理过程

### 自回归生成

```python
def generate(model, prompt_tokens, max_new_tokens):
    """自回归生成：每次只生成一个 token"""
    tokens = prompt_tokens.copy()

    for _ in range(max_new_tokens):
        # 1. 前向传播（所有 token 一起输入）
        logits = model(tokens)  # (seq_len, vocab_size)

        # 2. 只取最后一个位置的 logits
        next_token_logits = logits[-1]  # (vocab_size,)

        # 3. 采样（greedy / top-k / top-p / temperature）
        next_token = sample(next_token_logits)

        # 4. 追加到序列
        tokens.append(next_token)

        # 5. 遇到结束符就停
        if next_token == EOS_TOKEN:
            break

    return tokens

# 为什么叫"自回归"？
# 每一步的输入包含之前所有生成的 token
# 第 5 个 token 的生成依赖前 4 个 token
# 这就是"回归"——用自身的历史预测未来
```

### KV Cache（推理优化）

```python
# 问题：生成第 100 个 token 时，需要把前 99 个 token 都输入模型
# 但前 99 个 token 的 Key 和 Value 已经算过了，重复计算浪费

# KV Cache：缓存之前 token 的 K 和 V，只计算新 token 的 Q, K, V

# 无 KV Cache：每步计算量 = O(n^2)，n 是当前序列长度
# 有 KV Cache：每步计算量 = O(n)，只算新 token 的 attention

# 内存占用：
# KV Cache 大小 = 2 × num_layers × num_heads × seq_len × d_head × batch_size
# 对于 70B 模型、4096 长度：约 2.5GB（fp16）
```

## 十一、Transformer 与 Agent 开发的关系

### LLM 是 Agent 的大脑

```
Agent 架构：
┌─────────────────────────────────────┐
│              Agent                  │
│                                     │
│  ┌─────────────────────────────┐   │
│  │     LLM（Transformer）       │   │
│  │  - 理解用户意图              │   │
│  │  - 决定调用哪个工具          │   │
│  │  - 整合工具结果生成回复      │   │
│  └─────────────────────────────┘   │
│         ↑↓ Tool Calls              │
│  ┌─────────────────────────────┐   │
│  │     Tools（外部能力）         │   │
│  │  - 搜索、数据库、API         │   │
│  └─────────────────────────────┘   │
└─────────────────────────────────────┘
```

### 理解 Token 对 Agent 的影响

```python
# Agent 的成本 = token 数 × 单价
# 理解 token 计费对成本控制很重要

# GPT-4o 定价（示例）：
# 输入：$2.50 / 1M tokens
# 输出：$10.00 / 1M tokens

# Agent 的 token 消耗：
# 1. System Prompt：~500 tokens（固定开销）
# 2. 对话历史：累积增长
# 3. 工具描述：每个工具 ~100 tokens
# 4. 工具调用结果：可变
# 5. LLM 输出：包含思考过程

# 优化策略：
# - 压缩对话历史（只保留最近 N 轮）
# - 精简工具描述
# - 用小模型处理简单任务
# - 缓存常见查询的结果
```

### Streaming 与 Transformer

```python
# Transformer 的自回归特性天然支持流式输出
# 每生成一个 token 就可以立即返回，不用等全部生成完

async def stream_response(prompt):
    """流式输出——利用 Transformer 的自回归特性"""
    tokens = encode(prompt)
    while True:
        next_token = model.generate_one(tokens)  # 生成一个 token
        yield decode(next_token)                  # 立即返回
        tokens.append(next_token)
        if next_token == EOS:
            break

# 用户体验：逐字显示，而不是等 10 秒后一次性显示
# Agent 场景：边生成边调用工具，不用等整个回复生成完
```

## 十二、现代注意力变体（2023-2026 前沿）

> 参考：Sebastian Raschka [A Visual Guide to Attention Variants in Modern LLMs](https://magazine.sebastianraschka.com/p/visual-attention-variants)（2026.03）

### 为什么需要注意力变体？

```
标准 MHA 的问题：
- 每个 head 都有自己的 K/V → KV Cache 内存爆炸
- 长序列时 O(n^2) 计算量太大
- 128K/256K/1M 上下文时，推理成本不可接受

解决方向：
1. 减少 K/V 数量（GQA）
2. 压缩 K/V 存储（MLA）
3. 限制注意力范围（SWA、Sparse Attention）
4. 用更便宜的机制替代部分注意力层（Hybrid）
```

### GQA（Grouped-Query Attention）

```
核心思想：多个 Query Head 共享同一组 K/V

MHA（标准）：8 个 Q Head，8 个 K/V Head → 每个 Q 有独立 K/V
GQA（分组）：8 个 Q Head，2 个 K/V Head → 4 个 Q 共享 1 组 K/V
MQA（极端）：8 个 Q Head，1 个 K/V Head → 所有 Q 共享 1 组 K/V

KV Cache 节省：
  MHA: 8 × seq_len × d_head × 2(K+V)
  GQA: 2 × seq_len × d_head × 2(K+V)  → 节省 75%
  MQA: 1 × seq_len × d_head × 2(K+V)  → 节省 87.5%

使用 GQA 的模型：
  Llama 3、Qwen3、Gemma 3、Mistral Small 3.1

为什么 GQA 在 2026 仍是主流？
- 实现简单，训练稳定
- 超参调优少
- 小模型（<100B）效果好
```

### MLA（Multi-Head Latent Attention）

```
核心思想：不共享 K/V，而是压缩 K/V 的存储

MHA/GQA：缓存完整的 K 和 V 矩阵
MLA：缓存一个低维的"潜在表示"（latent），用时再还原为 K/V

类比：
  GQA = 多人共用一个硬盘（共享）
  MLA = 把文件压缩后再存（压缩）

优势：
- 相同内存下，建模质量比 GQA 更好
- DeepSeek-V2 消融实验：MLA 甚至略优于 MHA

劣势：
- 实现更复杂
- 需要额外的压缩/解压计算
- 小模型（<100B）不如 GQA 好调

使用 MLA 的模型：
  DeepSeek V3/R1、Kimi K2、GLM-5、Ling 2.5、Mistral Large 3
```

### SWA（Sliding Window Attention）

```
核心思想：每个 token 只关注附近的固定窗口，而非全部历史

全局注意力：token 可以看到所有之前的 token → O(n^2)
滑动窗口：token 只看最近 W 个 token → O(n × W)

Gemma 3 的方案：
- 5:1 的 local:global 比例（5 层 SWA + 1 层全局注意力）
- 窗口大小 1024 tokens
- 消融实验：对性能影响很小

组合使用：SWA + GQA
  SWA 减少每层的注意力范围
  GQA 减少每 token 的 KV 缓存大小
  两者叠加 → 长上下文推理成本大幅降低
```

### DeepSeek Sparse Attention（DSA）

```
核心思想：不是固定窗口，而是让模型自己决定关注哪些历史 token

SWA = 固定关注最近 W 个（硬编码局部性）
DSA = 学习关注最相关的 K 个（动态选择）

机制：
1. Lightning Indexer：对历史 token 打相关性分数
2. Token Selector：只保留 top-K 个高分 token
3. 对选中的 token 做标准 attention

使用 DSA 的模型：DeepSeek V3.2、GLM-5
通常与 MLA 配合使用：MLA 压缩缓存 + DSA 减少注意力范围
```

### 现代 LLM 架构对比表

```
┌──────────────────┬──────────┬──────────┬──────────┬──────────────┐
│ 模型              │ Attention│ 位置编码  │ FFN 变体  │ 其他特点      │
├──────────────────┼──────────┼──────────┼──────────┼──────────────┤
│ GPT-2            │ MHA      │ 可学习    │ GELU     │ 原始 Decoder │
│ Llama 3          │ GQA      │ RoPE     │ SwiGLU   │ 现代标准      │
│ DeepSeek V3      │ MLA      │ RoPE     │ SwiGLU+MoE│ 专家混合     │
│ Gemma 3          │ GQA+SWA  │ RoPE     │ GeGLU    │ 滑动窗口      │
│ Qwen3.5          │ Hybrid   │ RoPE     │ SwiGLU   │ 线性注意力混合 │
│ Claude (推测)     │ GQA/MLA  │ RoPE     │ SwiGLU   │ 未公开细节     │
└──────────────────┴──────────┴──────────┴──────────┴──────────────┘

共同趋势（2025-2026）：
- RoPE 成为位置编码标准
- SwiGLU 成为 FFN 标准
- GQA 或 MLA 替代原始 MHA
- 长上下文推动 SWA/Sparse/Hybrid 架构
```

### Hybrid Attention（混合注意力）— 未来趋势

```
核心思想：大部分层用便宜的线性注意力/状态空间模型，少数层保留标准注意力

Qwen3.5 方案（3:1）：
  3 层 Gated DeltaNet（线性注意力，O(n) 复杂度）
  + 1 层 Gated Attention（标准注意力，O(n^2) 但更精确）

为什么 Hybrid 是趋势？
- 128K+ 上下文时，全注意力太贵
- 线性注意力做"大致理解"，标准注意力做"精确检索"
- 混合后内存增长接近线性，而非二次方

其他 Hybrid 方案：
- Kimi Linear：Gated DeltaNet + Gated MLA
- Ling 2.5：Lightning Attention + MLA
- Nemotron：Mamba-2（状态空间模型）+ 稀疏注意力
```

## 十三、面试高频问题

**Q: Transformer 的 Self-Attention 计算复杂度是多少？**
A: O(n^2 * d)，n 是序列长度，d 是维度。因为每个 token 都要和所有其他 token 计算注意力。这是长文本处理的主要瓶颈。

**Q: 为什么用 Multi-Head 而不是单个大的 Attention？**
A: 多头可以同时关注不同类型的关系（语法、语义、位置等）。实验表明多头效果更好，且计算量相同（总维度不变）。

**Q: 位置编码的作用是什么？有哪些方案？**
A: Transformer 的 Self-Attention 是排列不变的，需要位置编码注入顺序信息。方案：正弦编码（原始）、可学习编码（BERT）、RoPE（GPT-NeoX、LLaMA、Claude）。

**Q: Decoder-only 为什么成为主流？**
A: 统一范式（所有任务转化为文本生成）、涌现能力（规模够大时出现推理等能力）、训练简单（只需预测下一个 token）、Scaling Law 表现好。

**Q: KV Cache 是什么？为什么重要？**
A: 缓存已计算的 Key 和 Value 矩阵，避免重复计算。没有 KV Cache，生成第 n 个 token 需要重新计算前 n-1 个 token 的 attention。KV Cache 是 LLM 推理性能的关键。

**Q: Pre-Norm 和 Post-Norm 的区别？**
A: Pre-Norm 先归一化再做 attention/FFN，训练更稳定。Post-Norm 先做再归一化，原始 Transformer 用这个。现代 LLM 普遍用 Pre-Norm。

**Q: Transformer 和 RNN 的核心区别？**
A: RNN 顺序处理，长距离依赖丢失，无法并行。Transformer 用 Self-Attention 并行处理所有 token，直接建模任意距离的依赖关系。

**Q: 为什么 Transformer 适合做 Agent 的大脑？**
A: 理解能力强（Self-Attention 捕捉复杂语义）、支持工具调用（Function Calling 是训练出来的能力）、流式输出（自回归天然支持）、上下文窗口大（可以处理长对话和工具结果）。

**Q: GQA 和 MLA 的区别？**
A: GQA 通过多个 Query Head 共享同一组 K/V 来减少 KV Cache（共享策略）。MLA 通过压缩 K/V 的存储为低维潜在表示来减少缓存（压缩策略）。GQA 实现简单，小模型效果好；MLA 实现复杂，但大模型（>100B）建模质量更好。Llama 3 用 GQA，DeepSeek V3 用 MLA。

**Q: 什么是 Sliding Window Attention？**
A: 每个 token 只关注最近的固定窗口（如 1024 tokens），而非全部历史。将 O(n^2) 降为 O(n×W)。Gemma 3 用 5:1 的 local:global 比例，消融实验显示对性能影响很小。通常与 GQA 组合使用。

**Q: 2025-2026 LLM 架构的共同趋势是什么？**
A: RoPE 成为位置编码标准；SwiGLU 成为 FFN 标准；GQA/MLA 替代原始 MHA；长上下文推动 SWA/Sparse/Hybrid 架构；Hybrid（线性注意力+标准注意力）是未来方向。

**Q: Hybrid Attention 是什么？**
A: 大部分层用便宜的线性注意力（如 Gated DeltaNet、Mamba-2），少数层保留标准注意力做精确检索。Qwen3.5 用 3:1 比例。优势是长上下文时内存增长接近线性。适合 Agent 场景（需要处理大量工具结果和长对话）。
