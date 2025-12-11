import math
import torch
import torch.nn as nn
from typing import Optional

from positional_embedding import apply_rope_emb, compute_freqs_cis
from normalization import RMSNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能整除 n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert self.head_dim % 2 == 0, "head_dim 必须为偶数以支持 RoPE"
        # nn.Linear 类似于 nn.Parameter(torch.empty(d_model, n_head * head_dim))
        # 然后 nn.init.kaiming_uniform_ 初始化
        self.wq = nn.Linear(d_model, n_head * head_dim, bias=False)
        self.wk = nn.Linear(d_model, n_head * head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_head * head_dim, bias=False)
        self.wo = nn.Linear(n_head * head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        支持掩码的多头注意力层
        """
        # x shape: [b, t, d_model]
        q = self.wq(x)  # [b, t, n_head * head_dim]
        k = self.wk(x)
        v = self.wv(x)
        # 分头，需先 reshape 再 transpose -> [b, n_head, t, head_dim]
        q = q.reshape(q.shape[0], q.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(k.shape[0], k.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        v = v.reshape(v.shape[0], v.shape[1], self.n_head, self.head_dim).transpose(1, 2)
        # RoPE 嵌入
        q, k = apply_rope_emb(q, k, freq_cis)
        # 多头注意力
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        attn = torch.softmax(scores, dim=-1)
        o = attn @ v
        # 合并头 [b, n_head, t, head_dim] -> [b, t, n_head * head_dim]
        o = o.transpose(1, 2).reshape(o.shape[0], -1, self.n_head * self.head_dim)
        return self.wo(o)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        # [b, t, d_model] -> [b, t, d_hidden] -> [b, t, d_model]
        return self.w2(torch.nn.functional.relu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_hidden):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.ffn = FeedForward(d_model, d_hidden)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, freq_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        支持掩码的 Transformer 层
        """
        # 自注意力
        attn = self.attn(self.norm1(x), freq_cis, mask)
        # 残差连接
        x = x + attn
        # 前馈网络
        ffn = self.ffn(self.norm2(x))
        # 残差连接
        x = x + ffn
        return x


class TransformerModel(nn.Module):
    def __init__(self, d_model, n_head, d_hidden, n_layer, vocab_size, max_seq_len=8192, rope_theta=10000.0):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_head, d_hidden) for _ in range(n_layer)])
        self.norm = RMSNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        # RoPE 频率按 head_dim 计算，而不是 d_model
        self.freq_cis = compute_freqs_cis(max_seq_len, d_model // n_head, rope_theta)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        支持掩码的 Transformer 模型
        """
        # 词嵌入
        x = self.embeddings(tokens)
        # 因果三角掩码（加性）：未来位置设为 -inf，其余为 0
        # torch.triu 生成上三角矩阵，diagonal=1 表示主对角线以下为 0
        t = tokens.shape[1]
        mask = torch.triu(torch.full((t, t), float('-inf'), device=tokens.device), diagonal=1).view(1, 1, t, t)
        # 层循环
        # 将 RoPE 频率裁剪到当前序列长度并移动到正确设备
        freqs_cis = self.freq_cis[:t].to(x.device)
        for layer in self.layers:
            x = layer(x, freqs_cis, mask)
        # 输出层
        return self.output(self.norm(x))


