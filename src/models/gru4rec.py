import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_

class GRU4Rec(nn.Module):
    def __init__(self, num_items, embedding_dim, hidden_size, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout_prob = float(dropout_prob)

        # 不含 padding 行
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)

        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_dim)
        self.out_dropout = nn.Dropout(0.0)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    xavier_uniform_(param.data)
        elif isinstance(module, nn.Linear):
            xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _last_hidden_from_outputs(outputs: torch.Tensor, seq_len: torch.Tensor):
        B = outputs.size(0)
        idx = torch.arange(B, device=outputs.device)
        last = outputs[idx, torch.clamp(seq_len - 1, min=0)]
        return last

    def forward(self, sequences: torch.Tensor):
        # 1..N -> 0..N-1，padding=0 做 mask
        nonpad = (sequences > 0).unsqueeze(-1).float()
        seq_idx = torch.clamp(sequences - 1, min=0)

        emb = self.item_embedding(seq_idx)
        emb = emb * nonpad
        emb = self.emb_dropout(emb)

        gru_out, _ = self.gru_layers(emb)
        proj_out = self.dense(gru_out)

        seq_len = (sequences != 0).sum(dim=1)
        seq_output = self._last_hidden_from_outputs(proj_out, seq_len)
        seq_output = self.out_dropout(seq_output)

        logits = torch.matmul(seq_output, self.item_embedding.weight.transpose(0, 1))
        return logits