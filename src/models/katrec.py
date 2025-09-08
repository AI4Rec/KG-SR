import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_
from src.models.components.transformer import TransformerBlock
from src.models.components.mask_utils import build_causal_mask, build_key_padding_mask
from src.models.components.init_utils import init_linear_xavier_uniform, init_embedding_xavier_uniform, init_layernorm_default

class KATRec(nn.Module):

    def __init__(self, num_items: int, config):
        super().__init__()
        self.num_items = int(num_items)
        self.embedding_dim = int(getattr(config, 'embedding_dim', 64))
        self.hidden_size = self.embedding_dim
        self.num_blocks = int(getattr(config, 'num_blocks', 2))
        self.num_heads = int(getattr(config, 'num_attention_heads', 2))
        assert self.embedding_dim % self.num_heads == 0, 'embedding_dim 必须能被 num_attention_heads 整除'
        self.head_dim = self.embedding_dim // self.num_heads
        self.dropout_prob = float(getattr(config, 'dropout_prob', 0.2))
        self.max_len = int(getattr(config, 'window_size', 50))

        # KG 相关
        self.kg_dim = int(getattr(config, 'kg_embedding_dim', getattr(config, 'kg_embedding_size', 64)))
        self.n_gcn_layers = int(getattr(config, 'n_gcn_layers', 2))
        self.mess_dropout_prob = float(getattr(config, 'mess_dropout_prob', 0.1))
        self.kg_margin = float(getattr(config, 'kg_margin', 1.0))
        self.kg_attn_alpha = float(getattr(config, 'kg_attn_alpha', 0.2))

        self.num_users = int(config.num_users)
        self.num_entities = int(config.num_entities)
        self.num_relations = int(config.num_relations)

        # 稀疏邻接（来自 dataloader 构建）
        self._raw_adj = config.adj_matrix.coalesce()
        self.register_buffer('adj_indices', self._raw_adj.indices())
        self.register_buffer('adj_values', self._raw_adj.values())
        self.adj_shape = self._raw_adj.shape

        # item -> entity 对齐
        item_to_ent = torch.full((self.num_items,), -1, dtype=torch.long)
        for (it_1, ent) in getattr(config, 'item_entity_map', {}).items():
            (it_i, ent_i) = (int(it_1) - 1, int(ent))
            if 0 <= it_i < self.num_items and 0 <= ent_i < self.num_entities:
                item_to_ent[it_i] = ent_i
        self.register_buffer('item_to_entity', item_to_ent)

        # 序列编码
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.position_embedding = nn.Embedding(self.max_len, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.embedding_dim, eps=1e-12)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.embedding_dim, n_heads=self.num_heads,
                inner_size=self.embedding_dim * 4, hidden_dropout_prob=self.dropout_prob,
                attn_dropout_prob=self.dropout_prob, hidden_act='relu', layer_norm_eps=1e-12
            ) for _ in range(self.num_blocks)
        ])

        # KG 编码
        self.user_embedding_kg = nn.Embedding(self.num_users, self.kg_dim)
        self.entity_embedding_kg = nn.Embedding(self.num_entities, self.kg_dim)
        self.relation_embedding = nn.Embedding(self.num_relations, self.kg_dim, padding_idx=0)
        self.msg_dropout = nn.Dropout(self.mess_dropout_prob)

        # 融合 & 注意力偏置构建
        self.kg2e_token = nn.Linear(self.kg_dim, self.embedding_dim)
        self.kg_q_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.kg_k_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.kg2e_item = nn.Linear(self.kg_dim, self.embedding_dim)
        self.item_fuse = nn.Linear(self.embedding_dim + self.embedding_dim, self.embedding_dim)

        self.apply(self._init_weights)
        self.apply(init_linear_xavier_uniform)
        self.apply(init_embedding_xavier_uniform)
        self.apply(init_layernorm_default)

        # 缓存（仅用于推理/评测阶段加速，不持久化）
        self.register_buffer('z_entity_cached', None, persistent=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _sparse_adj_on_device(self, device):
        return torch.sparse.FloatTensor(
            self.adj_indices.to(device), self.adj_values.to(device), torch.Size(self.adj_shape)
        ).coalesce()

    def _graph_propagate(self, device):
        adj = self._sparse_adj_on_device(device)
        x0_user = self.user_embedding_kg.weight
        x0_ent = self.entity_embedding_kg.weight
        x = torch.cat([x0_user, x0_ent], dim=0)

        outs = [x]
        for _ in range(self.n_gcn_layers):
            x = torch.sparse.mm(adj, x)
            x = self.msg_dropout(x)
            outs.append(x)

        x_final = torch.stack(outs, dim=0).mean(0)
        z_user = x_final[:self.num_users]
        z_entity = x_final[self.num_users:]
        return (z_user, z_entity)

    # ---------- 新增：KG 表征预计算与缓存（供评测/推理使用） ----------
    def precompute_kg(self, device):
        with torch.no_grad():
            _, z_entity = self._graph_propagate(device)
        # 缓存在 buffer 中（不持久化到 state_dict）
        self.z_entity_cached = z_entity

    def _build_kga_bias(self, sequences, z_entity):
        device = sequences.device
        (B, L) = sequences.size()

        nonpad = (sequences > 0).unsqueeze(-1)
        idx0 = torch.clamp(sequences - 1, min=0)
        ent_ids = self.item_to_entity[idx0]
        valid_mask = (ent_ids >= 0) & nonpad.squeeze(-1)

        ent_ids_clamped = torch.clamp(ent_ids, min=0)
        ent_k = z_entity[ent_ids_clamped]
        ent_k = ent_k * valid_mask.unsqueeze(-1).float()

        ent_e = self.kg2e_token(ent_k)

        q = self.kg_q_proj(ent_e).view(B, L, self.num_heads, self.head_dim)
        k = self.kg_k_proj(ent_e).view(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        bias = torch.matmul(q, k.transpose(-2, -1)) * scale

        key_pad = ~valid_mask
        key_pad_exp = key_pad.unsqueeze(1).unsqueeze(1)
        qry_pad_exp = key_pad.unsqueeze(1).unsqueeze(2)
        neg_inf = torch.tensor(-1e9, device=device, dtype=bias.dtype)
        bias = bias.masked_fill(key_pad_exp, neg_inf)
        bias = bias.masked_fill(qry_pad_exp, neg_inf)

        bias = bias.reshape(B * self.num_heads, L, L)
        return bias

    def forward(self, sequences: torch.Tensor):
        device = sequences.device

        # 使用缓存（若不存在则现算）——训练时通常不缓存，以便保持梯度/动态更新
        z_entity = self.z_entity_cached
        if z_entity is None:
            _, z_entity = self._graph_propagate(device)

        (B, L) = sequences.size()
        nonpad = (sequences > 0).unsqueeze(-1).float()
        seq_idx = torch.clamp(sequences - 1, min=0)

        item_emb = self.item_embedding(seq_idx) * nonpad
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.position_embedding(pos_ids)

        x = self.layer_norm(item_emb + pos_emb)
        x = self.dropout(x)

        causal_mask = build_causal_mask(L, device)
        key_padding_mask = build_key_padding_mask(sequences)

        kg_bias = self._build_kga_bias(sequences, z_entity)
        causal_mask_expanded = causal_mask.unsqueeze(0).expand(B * self.num_heads, -1, -1)
        attn_mask = causal_mask_expanded + self.kg_attn_alpha * kg_bias

        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
            x[sequences == 0] = 0.0

        seq_len = (sequences != 0).sum(dim=1).clamp(min=1)
        b_idx = torch.arange(B, device=device)
        user_vec = x[b_idx, seq_len - 1, :]

        all_item_e = self.item_embedding.weight
        all_item_ent = self.item_to_entity
        all_ent_ids = torch.clamp(all_item_ent, min=0)
        all_valid = (all_item_ent >= 0).float().unsqueeze(-1)

        all_ent_k = z_entity[all_ent_ids] * all_valid
        all_ent_e = self.kg2e_item(all_ent_k)

        all_item_vec = self.item_fuse(torch.cat([all_item_e, all_ent_e], dim=-1))
        logits = torch.matmul(user_vec, all_item_vec.t())
        return logits

    @staticmethod
    def _transE_score(h, r, t):
        return -torch.norm(h + r - t, p=2, dim=-1)

    def calculate_kg_loss(self, h_idx, r_idx, pos_t_idx, neg_t_idx):
        device = h_idx.device
        h = self.entity_embedding_kg(h_idx).to(device)
        r = self.relation_embedding(r_idx).to(device)
        t_pos = self.entity_embedding_kg(pos_t_idx).to(device)
        t_neg = self.entity_embedding_kg(neg_t_idx).to(device)

        pos_score = self._transE_score(h, r, t_pos)
        neg_score = self._transE_score(h, r, t_neg)

        target = torch.ones_like(pos_score, device=device)
        loss_fn = nn.MarginRankingLoss(margin=self.kg_margin)
        loss = loss_fn(pos_score, neg_score, target)
        return loss