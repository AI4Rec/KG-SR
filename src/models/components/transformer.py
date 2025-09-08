# src/models/components/transformer.py
# 通用 TransformerBlock 与掩码构造工具，供 SASRec / BERT4Rec 复用

import torch
import torch.nn as nn
from src.models.components.init_utils import (
    init_linear_xavier_uniform,
    init_embedding_xavier_uniform,
    init_layernorm_default,
)

_ACT_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}

def get_activation(name: str):
    name = (name or "relu").lower()
    return _ACT_MAP.get(name, nn.ReLU)()

def build_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), -1e9, device=device)
    return torch.triu(mask, diagonal=1)

class TransformerBlock(nn.Module):
    """
    通用 Transformer Block（自注意力 + FFN + 两处残差 LN）
    """
    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str = "relu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=attn_dropout_prob,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.do1 = nn.Dropout(hidden_dropout_prob)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            get_activation(hidden_act),
            nn.Linear(inner_size, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.do2 = nn.Dropout(hidden_dropout_prob)

        # 统一初始化
        self.apply(init_linear_xavier_uniform)
        self.apply(init_embedding_xavier_uniform)
        self.apply(init_layernorm_default)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        attn_out, _ = self.attention(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = self.ln1(x + self.do1(attn_out))
        ff = self.ffn(x)
        x = self.ln2(x + self.do2(ff))
        return x