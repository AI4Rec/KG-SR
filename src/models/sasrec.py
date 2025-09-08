import torch
import torch.nn as nn
from src.models.components.transformer import TransformerBlock
from src.models.components.mask_utils import build_causal_mask, build_key_padding_mask
from src.models.components.init_utils import (
    init_linear_xavier_uniform,
    init_embedding_xavier_uniform,
    init_layernorm_default,
)

class SASRec(nn.Module):
    def __init__(self, num_items, embedding_dim, max_len, num_blocks, num_attention_heads, dropout_prob):
        super().__init__()
        self.hidden_size = embedding_dim
        self.inner_size = self.hidden_size * 4
        self.max_len = max_len
        self.dropout_prob = dropout_prob
        self.layer_norm_eps = 1e-12

        # 不含 padding 行
        self.item_embedding = nn.Embedding(num_items, self.hidden_size)
        self.position_embedding = nn.Embedding(self.max_len, self.hidden_size)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                n_heads=num_attention_heads,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.dropout_prob,
                attn_dropout_prob=self.dropout_prob,
                hidden_act="relu",
                layer_norm_eps=self.layer_norm_eps,
            ) for _ in range(num_blocks)
        ])

        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.apply(init_linear_xavier_uniform)
        self.apply(init_embedding_xavier_uniform)
        self.apply(init_layernorm_default)

    @staticmethod
    def _generate_attention_mask(sequences):
        key_padding_mask = build_key_padding_mask(sequences)
        causal_mask = build_causal_mask(sequences.size(1), sequences.device)
        return causal_mask, key_padding_mask

    def encode(self, sequences, override_item_emb=None):
        # 将 1..N → 0..N-1；padding 0 单独 mask
        nonpad = (sequences > 0).unsqueeze(-1).float()
        seq_idx = torch.clamp(sequences - 1, min=0)

        if override_item_emb is None:
            item_emb = self.item_embedding(seq_idx)
        else:
            # override_item_emb 已经在外部按 0 基对齐
            item_emb = override_item_emb

        # 抹掉 padding 位置的 embedding 贡献
        item_emb = item_emb * nonpad

        position_ids = torch.arange(sequences.size(1), dtype=torch.long, device=sequences.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequences)
        position_emb = self.position_embedding(position_ids)

        x = item_emb + position_emb
        x = self.layer_norm(x)
        x = self.dropout(x)

        causal_mask, key_padding_mask = self._generate_attention_mask(sequences)
        for block in self.transformer_blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
            x[sequences == 0] = 0.0

        item_seq_len = (sequences != 0).sum(dim=1)
        b_idx = torch.arange(sequences.size(0), device=sequences.device)
        last_idx = torch.clamp(item_seq_len - 1, min=0)
        final_user_repr = x[b_idx, last_idx, :]
        return final_user_repr

    def forward(self, sequences, override_item_emb=None):
        final_user_repr = self.encode(sequences, override_item_emb=override_item_emb)
        # 物品词表是 0..N-1，与 logits 列对齐
        logits = torch.matmul(final_user_repr, self.item_embedding.weight.transpose(0, 1))
        return logits