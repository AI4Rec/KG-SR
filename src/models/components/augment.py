import torch

__all__ = [
    "safe_random_mask",
    "random_crop",
    "local_reorder",
    "random_substitute",
    "compose_augmentation",
]

def _nonzero_pos(row: torch.Tensor):
    return row.nonzero(as_tuple=False).flatten()

@torch.no_grad()
def safe_random_mask(seq: torch.Tensor, mask_ratio: float) -> torch.Tensor:
    """
    保证每个样本至少保留 1 个非零 token 的随机mask（替代原随机伯努利）。
    """
    if mask_ratio <= 0.0:
        return seq
    out = seq.clone()
    (B, L) = out.size()
    dev = out.device
    for i in range(B):
        pos = _nonzero_pos(out[i])
        n = int(pos.numel())
        if n <= 1:
            continue  # 不动，避免全零
        m = int(round(mask_ratio * n))
        m = max(0, min(m, n - 1))  # 至多mask n-1个
        if m == 0:
            continue
        idx = torch.randperm(n, device=dev)[:m]
        out[i, pos[idx]] = 0
    return out

@torch.no_grad()
def random_crop(seq: torch.Tensor, min_keep_ratio: float) -> torch.Tensor:
    if min_keep_ratio >= 1.0:
        return seq
    out = seq.clone()
    (B, L) = out.size()
    dev = out.device
    valid_len = (out != 0).sum(dim=1)
    r = torch.empty(B, device=dev).uniform_(min_keep_ratio, 1.0)
    keep = torch.clamp((valid_len.float() * r).round().long(), min=1)
    for i in range(B):
        k = int(keep[i].item())
        pos = _nonzero_pos(out[i])
        if pos.numel() == 0:
            continue
        pos = pos[-k:]  # 保留尾部k个
        new_row = torch.zeros(L, dtype=out.dtype, device=dev)
        new_row[-k:] = out[i, pos]
        out[i] = new_row
    return out

@torch.no_grad()
def local_reorder(seq: torch.Tensor, reorder_prob: float, max_window: int = 4) -> torch.Tensor:
    if reorder_prob <= 0.0 or max_window <= 1:
        return seq
    out = seq.clone()
    (B, L) = out.size()
    dev = out.device
    for i in range(B):
        if torch.rand(1, device=dev).item() > reorder_prob:
            continue
        pos = _nonzero_pos(out[i])
        if pos.numel() <= 2:
            continue
        w = int(torch.randint(2, min(max_window, pos.numel()) + 1, (1,), device=dev).item())
        start = int(torch.randint(0, pos.numel() - w + 1, (1,), device=dev).item())
        seg = pos[start:start + w]
        vals = out[i, seg].clone()
        out[i, seg] = vals[torch.randperm(w, device=dev)]
    return out

@torch.no_grad()
def random_substitute(seq: torch.Tensor, substitute_prob: float, num_items: int) -> torch.Tensor:
    if substitute_prob <= 0.0 or num_items <= 1:
        return seq
    out = seq.clone()
    (B, L) = out.size()
    dev = out.device
    prob = torch.full((B, L), substitute_prob, device=dev)
    take = (torch.bernoulli(prob).bool()) & out.ne(0)
    if take.any():
        out[take] = torch.randint(1, num_items + 1, (take.sum().item(),), device=dev, dtype=out.dtype)
    return out

@torch.no_grad()
def compose_augmentation(
    seq: torch.Tensor,
    *,
    num_items: int,
    mask_ratio: float = 0.0,
    crop_min_ratio: float = 1.0,
    reorder_prob: float = 0.0,
    substitute_prob: float = 0.0,
) -> torch.Tensor:
    """
    先裁剪再扰动，最后用“安全mask”，并做最终兜底：若某行被意外置零，则把“原序列最后一个非零token”放回末尾。
    """
    orig = seq
    out = seq
    # 顺序：crop -> reorder -> substitute -> safe_mask
    out = random_crop(out, crop_min_ratio)
    out = local_reorder(out, reorder_prob, max_window=4)
    out = random_substitute(out, substitute_prob, num_items)
    out = safe_random_mask(out, mask_ratio)

    # 兜底：防全零（极端情况下）
    (B, L) = out.size()
    dev = out.device
    for i in range(B):
        if out[i].nonzero(as_tuple=False).numel() == 0:
            pos = _nonzero_pos(orig[i])
            if pos.numel() > 0:
                last_val = int(orig[i, pos[-1]].item())
                new_row = torch.zeros(L, dtype=out.dtype, device=dev)
                new_row[-1] = last_val
                out[i] = new_row
    return out