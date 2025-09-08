# src/utils/metrics.py

import torch

class AllRankMetrics:
    """
    Calculates all-ranking metrics (Hit@K, NDCG@K, MRR) by accumulating
    statistics batch by batch, avoiding high memory usage.
    The evaluation protocol is to rank the ground truth item against all other items.
    """
    def __init__(self, top_k=10):
        self.top_k = top_k
        # 初始化统计累加值，而不是列表
        self.reset()

    def reset(self):
        """Resets the internal accumulators."""
        self._total_hits = 0.0
        self._total_ndcgs = 0.0
        self._total_mrrs = 0.0
        self._count = 0

    def __call__(self, logits, targets):
        """
        Updates metrics with a new batch of predictions and targets.
        It calculates metrics for the batch and adds them to the running totals.

        Args:
            logits (torch.Tensor): The model output scores. Shape: (batch_size, num_items)
            targets (torch.Tensor): The ground truth item ids. Shape: (batch_size,)
        """
        # 确保 targets 和 logits 在同一设备上进行比较
        targets = targets.to(logits.device)

        # 获取每个样本中目标物品的分数
        target_scores = logits.gather(1, targets.view(-1, 1))

        # 计算目标物品的排名 (有多少个物品的分数比目标高 + 1)
        ranks = (logits > target_scores).sum(dim=1) + 1
        
        # --- 逐批次计算指标并累加 ---
        
        # Hit Rate @ K: 排名是否在前 K
        hits_at_k = (ranks <= self.top_k).float()
        
        # NDCG @ K: DCG = 1 / log2(rank + 1) for ranks in top K. IDCG is 1.
        in_top_k_mask = (ranks <= self.top_k).float()
        ndcg_at_k = in_top_k_mask * (1.0 / torch.log2(ranks.float() + 1))

        # Mean Reciprocal Rank (MRR)
        mrr = 1.0 / ranks.float()

        # 累加当前批次的结果到总和中
        # 使用 .sum().item() 将批次总和转换为Python标量，避免张量累积
        self._total_hits += hits_at_k.sum().item()
        self._total_ndcgs += ndcg_at_k.sum().item()
        self._total_mrrs += mrr.sum().item()
        self._count += targets.size(0)

    def summary(self):
        """
        Calculates and returns the final average metrics based on accumulated stats.
        
        Returns:
            dict: A dictionary containing the average Hit@K, NDCG@K, and MRR.
        """
        if self._count == 0:
            return {f'Hit@{self.top_k}': 0, f'NDCG@{self.top_k}': 0, 'MRR': 0}

        # 计算最终的平均值
        avg_hit = self._total_hits / self._count
        avg_ndcg = self._total_ndcgs / self._count
        avg_mrr = self._total_mrrs / self._count
        
        return {
            f'Hit@{self.top_k}': avg_hit,
            f'NDCG@{self.top_k}': avg_ndcg,
            'MRR': avg_mrr
        }