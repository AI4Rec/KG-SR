# 文件路径：src/models/bert4rec.py
import torch
import torch.nn as nn
from src.models.components.transformer import TransformerBlock
from src.models.components.mask_utils import build_key_padding_mask
from src.models.components.init_utils import (
    init_linear_xavier_uniform,
    init_embedding_xavier_uniform,
    init_layernorm_default,
)

class BERT4Rec(nn.Module):
    """
    适配当前项目训练协议（next-item 预测）的 BERT4Rec 变体：
    - 使用双向 Transformer（无因果掩码），只对 padding 做 key padding mask；
    - 用户表示采用最后一个非 PAD 位置的 token 表示（last pooling）；
    - 输出对所有 items 的打分 logits（与 Trainer 的 CE loss 接口一致）。
    """
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
        max_len: int,
        num_blocks: int,
        num_attention_heads: int,
        dropout_prob: float,
        pooling: str = "last",   # 可选: last / mean
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        assert embedding_dim % num_attention_heads == 0, \
            "embedding_dim 必须能被 num_attention_heads 整除"

        self.num_items = int(num_items)
        self.hidden_size = int(embedding_dim)
        self.inner_size = self.hidden_size * 4
        self.max_len = int(max_len)
        self.dropout_prob = float(dropout_prob)
        self.layer_norm_eps = float(layer_norm_eps)
        self.num_blocks = int(num_blocks)
        self.num_heads = int(num_attention_heads)
        self.pooling = str(pooling).lower()

        # 与现有模型保持一致：物品 ID 从 1..num_items；索引时需减 1；0 保留为 PAD
        self.item_embedding = nn.Embedding(self.num_items, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                n_heads=self.num_heads,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.dropout_prob,
                attn_dropout_prob=self.dropout_prob,
                hidden_act='gelu',            # BERT 常用 GELU
                layer_norm_eps=self.layer_norm_eps,
            ) for _ in range(self.num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        # 初始化策略与本仓库其他模型保持一致
        self.apply(init_linear_xavier_uniform)
        self.apply(init_embedding_xavier_uniform)
        self.apply(init_layernorm_default)

    @staticmethod
    def _key_padding_mask(sequences: torch.Tensor) -> torch.Tensor:
        # True 表示需要 mask 的位置（即 PAD=0）
        return build_key_padding_mask(sequences)

    @staticmethod
    def _last_hidden(x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        # 取每条序列最后一个非 PAD 的位置表示
        seq_len = (sequences != 0).sum(dim=1).clamp(min=1)
        b_idx = torch.arange(sequences.size(0), device=sequences.device)
        return x[b_idx, seq_len - 1, :]

    @staticmethod
    def _mean_hidden(x: torch.Tensor, sequences: torch.Tensor) -> torch.Tensor:
        # 对非 PAD 位置取平均
        mask = (sequences != 0).float().unsqueeze(-1)  # (B, L, 1)
        summed = (x * mask).sum(dim=1)                # (B, D)
        denom = mask.sum(dim=1).clamp(min=1.0)        # (B, 1)
        return summed / denom

    def encode(self, sequences: torch.Tensor, override_item_emb: torch.Tensor = None) -> torch.Tensor:
        """
        双向编码，无因果 mask，仅做 padding mask。
        """
        # (B, L) -> (B, L, D)
        nonpad = (sequences > 0).unsqueeze(-1).float()
        seq_idx = torch.clamp(sequences - 1, min=0)

        if override_item_emb is None:
            item_emb = self.item_embedding(seq_idx)   # (B, L, D)
        else:
            # 兼容外部传入的预编码 item embedding（与 SASRec 接口一致）
            item_emb = override_item_emb

        item_emb = item_emb * nonpad

        # 位置嵌入
        pos_ids = torch.arange(sequences.size(1), device=sequences.device).unsqueeze(0).expand_as(sequences)
        pos_emb = self.position_embedding(pos_ids)

        x = item_emb + pos_emb
        x = self.layer_norm(x)
        x = self.dropout(x)

        # 仅使用 key padding mask（无因果掩码 -> 双向）
        key_padding_mask = self._key_padding_mask(sequences)

        for block in self.transformer_blocks:
            x = block(x, attn_mask=None, key_padding_mask=key_padding_mask)
            # 对 PAD 位置清零，数值稳定
            x[sequences == 0] = 0.0

        # 池化为用户表示
        if self.pooling == "mean":
            final_user_repr = self._mean_hidden(x, sequences)
        else:
            final_user_repr = self._last_hidden(x, sequences)

        return final_user_repr

    def forward(self, sequences: torch.Tensor, override_item_emb: torch.Tensor = None) -> torch.Tensor:
        """
        输出对所有 items 的打分矩阵 (B, num_items)
        —— 与 Trainer 中的 CrossEntropyLoss(next_item-1) 完全对齐。
        """
        user_repr = self.encode(sequences, override_item_emb=override_item_emb)  # (B, D)
        logits = torch.matmul(user_repr, self.item_embedding.weight.transpose(0, 1))  # (B, N)
        return logits