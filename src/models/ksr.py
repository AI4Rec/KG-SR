import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_

class KSR(nn.Module):
    def __init__(self, num_items: int, config):
        super().__init__()
        self.n_items = int(num_items)
        self.embedding_size = int(getattr(config, "embedding_dim", 64))
        self.hidden_size = int(getattr(config, "hidden_size", self.embedding_size))
        self.num_layers = int(getattr(config, "num_layers", 1))
        self.dropout_prob = float(getattr(config, "dropout_prob", 0.2))

        self.kg_embedding_size = int(getattr(config, "kg_embedding_size",
                                             getattr(config, "kg_embedding_dim", 64)))
        self.gamma = float(getattr(config, "gamma", 0.5))
        self.freeze_kg = bool(getattr(config, "freeze_kg", False))

        assert hasattr(config, "num_entities"), "KSR needs config.num_entities"
        assert hasattr(config, "num_relations"), "KSR needs config.num_relations (with padding 0)"
        self.n_entities = int(config.num_entities)

        self.n_relations_total = int(config.num_relations)  # 包含 0（pad）
        self.n_relations = self.n_relations_total - 1
        assert self.n_relations > 0

        # 物品不含 padding 行
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)

        # KG embedding
        self.entity_embedding = nn.Embedding(self.n_entities, self.kg_embedding_size)
        if self.freeze_kg:
            self.entity_embedding.weight.requires_grad = False
        self.relation_embedding = nn.Embedding(self.n_relations_total, self.kg_embedding_size, padding_idx=0)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )

        self.dense_seq_to_k = nn.Linear(self.hidden_size, self.kg_embedding_size)
        self.dense_layer_u = nn.Linear(self.hidden_size + self.kg_embedding_size, self.embedding_size)
        self.dense_layer_i = nn.Linear(self.embedding_size + self.kg_embedding_size, self.embedding_size)

        # item -> entity（零基）
        item_to_entity = torch.full((self.n_items,), -1, dtype=torch.long)
        for it_1, ent in getattr(config, "item_entity_map", {}).items():  # it_1: 1..N
            it_i = int(it_1) - 1
            ent_i = int(ent)
            if 0 <= it_i < self.n_items and 0 <= ent_i < self.n_entities:
                item_to_entity[it_i] = ent_i
        self.register_buffer("item_to_entity", item_to_entity)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            try:
                nn.init.xavier_uniform_(module.weight_hh_l0)
                nn.init.xavier_uniform_(module.weight_ih_l0)
            except AttributeError:
                for name, param in module.named_parameters():
                    if "weight" in name:
                        xavier_uniform_(param.data)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _get_kg_embedding(self, head_items: torch.Tensor):
        # head_items: 原序列 token（0 表 pad，1..N 真实）
        nonpad = (head_items > 0).unsqueeze(-1).float()
        idx0 = torch.clamp(head_items - 1, min=0)
        ent_ids = self.item_to_entity[idx0]                    # 零基 item -> 实体
        valid_mask = (ent_ids >= 0).float().unsqueeze(-1)
        ent_ids_clamped = torch.clamp(ent_ids, min=0)

        head_e = self.entity_embedding(ent_ids_clamped) * valid_mask * nonpad

        # tail_M: 针对所有关系（去掉 0）构造
        rel_mat = self.relation_embedding.weight[1:]
        rel_mat = rel_mat.unsqueeze(0).expand(head_e.size(0), -1, -1)
        head_M = head_e.unsqueeze(1).expand(-1, self.n_relations, -1)
        tail_M = head_M + rel_mat
        return head_e, tail_M

    def _memory_update_cell(self, user_memory: torch.Tensor, update_memory: torch.Tensor):
        z = torch.sigmoid((user_memory * update_memory).sum(-1)).unsqueeze(-1)
        updated_user_memory = (1.0 - z) * user_memory + z * update_memory
        return updated_user_memory

    def memory_update(self, item_seq: torch.Tensor, item_seq_len: torch.Tensor):
        B, L = item_seq.size()
        device = item_seq.device
        last_idx = torch.clamp(item_seq_len - 1, min=0)

        user_memory = torch.zeros(B, self.n_relations, self.kg_embedding_size, device=device)
        last_user_memory = torch.zeros_like(user_memory)

        for i in range(L):
            _, upd_mem = self._get_kg_embedding(item_seq[:, i])
            user_memory = self._memory_update_cell(user_memory, upd_mem)
            mask = (last_idx == i)
            if mask.any():
                last_user_memory[mask] = user_memory[mask]
        return last_user_memory

    def memory_read(self, seq_output_H: torch.Tensor, user_memory: torch.Tensor):
        q_k = self.dense_seq_to_k(seq_output_H)
        attrs = self.relation_embedding.weight[1:]
        logits = self.gamma * torch.matmul(q_k, attrs.t()).float()
        attn = torch.softmax(logits, dim=-1)
        u_m = (user_memory * attn.unsqueeze(-1)).sum(1)
        return u_m

    @staticmethod
    def _last_hidden_from_gru(gru_out: torch.Tensor, seq_len: torch.Tensor):
        B = gru_out.size(0)
        idx = torch.arange(B, device=gru_out.device)
        last = gru_out[idx, torch.clamp(seq_len - 1, min=0)]
        return last

    def _get_item_comb_embedding(self, items: torch.Tensor):
        # items: 零基物品索引（0..N-1）
        ent_ids = self.item_to_entity[items]
        valid = (ent_ids >= 0).float().unsqueeze(-1)
        ent_ids_clamped = torch.clamp(ent_ids, min=0)

        h_e = self.entity_embedding(ent_ids_clamped) * valid
        i_e = self.item_embedding(items)
        q_i = self.dense_layer_i(torch.cat((i_e, h_e), dim=-1))
        return q_i

    def forward(self, sequences: torch.Tensor):
        # 1..N -> 0..N-1；padding=0 用 mask
        nonpad = (sequences > 0).unsqueeze(-1).float()
        seq_idx = torch.clamp(sequences - 1, min=0)

        emb = self.item_embedding(seq_idx) * nonpad
        emb = self.emb_dropout(emb)
        gru_out, _ = self.gru_layers(emb)
        seq_len = (sequences != 0).sum(dim=1)
        seq_H = self._last_hidden_from_gru(gru_out, seq_len)

        user_memory = self.memory_update(sequences, seq_len)
        u_m = self.memory_read(seq_H, user_memory)

        p_u = self.dense_layer_u(torch.cat((seq_H, u_m), dim=-1))

        # 所有物品（零基 0..N-1）
        all_item_e = self.item_embedding.weight
        all_item_ent = self.item_to_entity
        all_ent_ids = torch.clamp(all_item_ent, min=0)
        all_valid = (all_item_ent >= 0).float().unsqueeze(-1)
        all_h_e = self.entity_embedding(all_ent_ids) * all_valid
        all_q_i = self.dense_layer_i(torch.cat((all_item_e, all_h_e), dim=-1))

        logits = torch.matmul(p_u, all_q_i.t())
        return logits