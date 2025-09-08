# src/models/components/init_utils.py
# 统一的权重初始化助手：便于跨模型保持一致且可配置

import torch.nn as nn
from torch.nn.init import xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, constant_

__all__ = [
    "init_linear_xavier_uniform",
    "init_embedding_xavier_uniform",
    "init_layernorm_default",
    "init_gru_xavier_uniform",
]

def init_linear_xavier_uniform(m: nn.Module):
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight)
        if m.bias is not None:
            constant_(m.bias, 0)

def init_embedding_xavier_uniform(m: nn.Module):
    if isinstance(m, nn.Embedding):
        xavier_uniform_(m.weight)

def init_layernorm_default(m: nn.Module):
    if isinstance(m, nn.LayerNorm):
        constant_(m.bias, 0)
        constant_(m.weight, 1.0)

def init_gru_xavier_uniform(m: nn.Module):
    if isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if "weight_ih" in name or "weight_hh" in name:
                xavier_uniform_(param.data)