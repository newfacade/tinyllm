import torch


def compute_freqs_cis(seq_len: int, dim: int, base=10000.0):
    """
    计算固定频率 cis 用于 RoPE 嵌入和固定位置编码。
    返回形状为 [seq_len, dim//2] 的复数张量，每个元素为 cis(θ) = cos(θ) + i·sin(θ) = e^{iθ}。
    """
    t = torch.arange(seq_len)
    # theta_j
    assert dim % 2 == 0, "dim 需为偶数以匹配频率对"
    freqs = base ** (-torch.arange(0, dim, 2) / dim)
    # m * theta_j
    freqs = torch.outer(t, freqs)  # [seq_len, dim//2]
    # exp ** (i*m*theta_j)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def sinusoidal_positional_encoding_new(seq_len: int, d_model: int, base=10000.0):
    """
    返回形状为 [seq_len, d_model] 的固定正弦位置编码（偶数 cos，奇数 sin）。
    """
    freq_cis = compute_freqs_cis(seq_len, d_model, base)
    return torch.view_as_real(freq_cis).flatten(1)


def apply_rope_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    """
    应用 RoPE 嵌入到查询 q 和键 k 上
    """
    # view_as_complex: [[q_0, q_1], [q_2, q_3],...] -> [q_0 + i*q_1, q_2 + i*q_3, ...]
    # (b, t, n_head, head_dim) -> (b, t, n_head, head_dim//2)
    q = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.reshape(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
    # (b, t, n_head, head_dim//2) -> (b, t, n_head, head_dim//2, 2) -> (b, t, n_head, head_dim)
    q = torch.view_as_real(q * freqs_cis).flatten(3)
    k = torch.view_as_real(k * freqs_cis).flatten(3)
    return q, k

