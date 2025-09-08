# src/models/components/mask_utils.py
# 统一的注意力掩码工具：可复用在所有 Transformer 系模型中

import torch

__all__ = [
    "build_causal_mask",
    "build_key_padding_mask",
]

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    严格上三角因果掩码，禁止看未来。
    使用 -1e9 代替 -inf，提升数值稳定性。
    形状: [seq_len, seq_len]，用于 MultiheadAttention(attn_mask=...)
    """
    mask = torch.full((seq_len, seq_len), -1e9, device=device)
    return torch.triu(mask, diagonal=1)

def build_key_padding_mask(sequences: torch.Tensor) -> torch.Tensor:
    """
    Key padding mask：True 表示被 mask（即 padding=0 的位置）。
    形状: [batch, seq_len]
    """
    return sequences.eq(0)