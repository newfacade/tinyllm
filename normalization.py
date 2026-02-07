import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    简化版 LayerNorm：在张量的最后一维上进行归一化，并应用可学习的仿射变换。
    等价目标：与 `torch.nn.LayerNorm(normalized_shape=dim, eps=eps)` 的数值行为一致。

    为什么在 LLM 中使用 LayerNorm：
    - LayerNorm 按每个 token 的特征维度（d_model）归一化，和 batch 大小、序列长度无关，
      训练与推理阶段数值行为一致，适合自回归生成与可变长度序列。
      LayerNorm normalize each example and each position independently.
    """
    def __init__(self, dim, eps=1e-5):
        # 在 LLM 中， dim 通常就是隐藏维度 d_model
        super().__init__()
        self.eps = eps
        # 可学习的缩放和偏置参数，形状为 [dim]，按最后一维进行广播
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x 一般是三维张量，形状是 (batch_size, seq_len, d_model) ，LayerNorm在最后一维（d_model）上归一化
        mean = x.mean(dim=-1, keepdim=True)
        # 使用总体方差（unbiased=False）以匹配 PyTorch 官方 LayerNorm 的实现
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 将 eps 加到方差上再开方，提升数值稳定性，符合标准定义：
        # y = (x - mean) / sqrt(var + eps)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        # 应用可学习的仿射变换
        # 每个样本每个位置的仿射变换都一样
        return self.weight * x_hat + self.bias


class RMSNorm(nn.Module):
    """
    RMSNorm：仅通过均方根（Root Mean Square）进行归一化，不减均值。

    为什么很多 LLM 使用 RMSNorm 而不是 LayerNorm：
    - 数值稳定性更好：RMSNorm不做均值扣除，减少减法造成的数值抵消，
      在半精度（fp16/bf16）训练与推理中更稳，梯度更平滑。
    - 计算与内存更省：只需计算均方值（x^2 的均值），省去均值与方差的两次统计，
      访存/通信负担更小，在张量并行、流水线并行下更友好。
    - 与批大小和序列长度解耦：与LayerNorm一样独立于batch/seq，但在小批量、
      长序列和较大学习率的设定下，实证更易收敛、更稳（如 T5、LLaMA 的报告）。
    - 简化实现与参数：常见实现仅使用缩放参数（无偏置），与预归一化（Pre-Norm）残差结构搭配稳定。

    归一化方式：
        y = x / sqrt(mean(x^2) + eps) * weight

    形状约定：
        输入 x 形状为 (..., dim)，在最后一维 dim 上归一化；weight 的形状为 [dim]，按最后一维广播。
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # root mean square
        ms = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(ms + self.eps)

    def forward(self, x):
        # x shape: (batch_size, seq_len, dim) 
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
