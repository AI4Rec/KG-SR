from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm

from src.candidates.predictors import predict_topk_for_train_sequences
from src.candidates.successors import build_successor_topn_for_train


def _merge_dedup_lists(a: List[int], b: List[int]) -> List[int]:
    """按顺序去重合并（先保留 a 顺序，再补 b 中未出现的）"""
    seen = set()
    out: List[int] = []
    for x in a + b:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def build_candidates_for_train(
    dataset_name: str,
    window_size: int,
    model_name: Optional[str],  # 具体名或 'ALL'
    K: int,
    N: int,
    save_csv: bool = True,
):
    """
    同时构建两种候选（TopK 预测 & 后继 TopN），并按行合并去重：
      - model_name 支持 'ALL'：会分别输出每个模型的合并候选；
      - 返回 {模型名: DataFrame[user_id, sequence, target_id, topk_pred, succ_topN, merged_candidates]}
    """
    # 1) 预测候选
    pred_dict: Dict[str, pd.DataFrame] = predict_topk_for_train_sequences(
        dataset_name=dataset_name, window_size=window_size, model_name=model_name, K=K, save_csv=False
    )

    # 2) 后继候选（只算一次复用）
    succ_df = build_successor_topn_for_train(dataset_name=dataset_name, window_size=window_size, N=N, save_csv=False)
    succ_col = succ_df['succ_topN'].tolist()

    results: Dict[str, pd.DataFrame] = {}

    for mname, pdf in pred_dict.items():
        assert len(pdf) == len(succ_df), '两种候选的行数不一致，请检查数据'
        merged_col: List[str] = []

        for i in tqdm(range(len(pdf)), desc=f'Merge[{mname}]'):
            topk = list(map(int, str(pdf.iloc[i]['topk_pred']).split(',')))
            succ = list(map(int, str(succ_col[i]).split(',')))
            merged = _merge_dedup_lists(topk, succ)
            merged_col.append(','.join(map(str, merged)))

        out = pdf.copy()
        out['succ_topN'] = succ_df['succ_topN']
        out['merged_candidates'] = merged_col

        if save_csv:
            # 保存在各自模型目录下，便于追溯
            # NOTE: predictors.py 默认把预测保存到 saved/{dataset}/{Model}/L_{L}/train_topK_pred.csv
            # 这里保持同一目录，文件名标注 merged
            from pathlib import Path
            mdir = Path('./saved') / dataset_name / mname / f'L_{window_size}'
            mdir.mkdir(parents=True, exist_ok=True)
            out_path = mdir / f'train_candidates_top{K}_succ{N}_merged.csv'
            out.to_csv(out_path, index=False)
            print(f'已保存：{out_path}')

        results[mname] = out

    return results