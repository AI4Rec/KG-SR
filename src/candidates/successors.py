from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Set
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from src.utils.path_finder import PathFinder

def _load_user_sequences(processed_inter: Path) -> Dict[int, List[int]]:
    df = pd.read_csv(processed_inter, sep='\t', header=0)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df = df.sort_values(['user_id', 'timestamp'], kind='mergesort')
    seqs = df.groupby('user_id')['item_id'].apply(list).to_dict()
    return {int(u): list(map(int, items)) for (u, items) in seqs.items()}

def _build_training_transitions(user_seqs: Dict[int, List[int]]) -> Tuple[Dict[int, Dict[int, Counter]], Dict[int, Counter], Dict[int, Set[int]]]:
    user_to_succ: Dict[int, Dict[int, Counter]] = defaultdict(lambda : defaultdict(Counter))
    global_succ: Dict[int, Counter] = defaultdict(Counter)
    user_train_seen: Dict[int, Set[int]] = defaultdict(set)
    for (uid, seq) in user_seqs.items():
        if len(seq) < 3:
            continue
        train_seq = seq[:-2]
        if len(train_seq) < 2:
            user_train_seen[uid].update(train_seq)
            continue
        user_train_seen[uid].update(train_seq)
        for i in range(len(train_seq) - 1):
            t = int(train_seq[i])
            nxt = int(train_seq[i + 1])
            user_to_succ[uid][t][nxt] += 1
            global_succ[t][nxt] += 1
    return (user_to_succ, global_succ, user_train_seen)

def _rank_take(counter: Counter, limit: int, exclude: set) -> List[int]:
    out: List[int] = []
    for (item, _cnt) in counter.most_common():
        if item in exclude:
            continue
        out.append(item)
        if len(out) >= limit:
            break
    return out

def build_successor_topn_for_train(dataset_name: str, window_size: int, N: int, seed: int=42, save_csv: bool=True):
    rng = np.random.default_rng(seed)
    pf = PathFinder(dataset_name, window_size)
    processed_inter = pf.processed_dir() / f'{dataset_name}.inter'
    assert processed_inter.exists(), f'未找到：{processed_inter}（请先跑 preprocess.py）'

    user_seqs = _load_user_sequences(processed_inter)
    (user_to_succ, global_succ, user_train_seen) = _build_training_transitions(user_seqs)

    train_csv = pf.split_dir_path() / 'train.csv'
    df_train = pd.read_csv(train_csv)
    assert 'user_id' in df_train.columns and 'target_id' in df_train.columns, f'train.csv 需包含列：user_id, target_id；当前列={list(df_train.columns)}'

    # === 用 link 统计 num_items（替代 item_map.csv） ===
    link_path = pf.processed_dir() / f'{dataset_name}.link'
    link_df = pd.read_csv(link_path, sep='\t', dtype={'item_id:token': 'int32'})
    num_items = int(link_df['item_id:token'].max())
    all_items = set(range(1, num_items + 1))

    succ_lists: List[List[int]] = []
    for (_, row) in tqdm(df_train.iterrows(), total=len(df_train), desc='SuccessorTopN'):
        uid = int(row['user_id'])
        target = int(row['target_id'])
        chosen: List[int] = []
        used = {target}

        if uid in user_to_succ and target in user_to_succ[uid]:
            picked = _rank_take(user_to_succ[uid][target], N, used)
            chosen.extend(picked)
            used.update(picked)

        if len(chosen) < N and target in global_succ:
            fill = _rank_take(global_succ[target], N - len(chosen), used)
            chosen.extend(fill)
            used.update(fill)

        if len(chosen) < N:
            seen = user_train_seen.get(uid, set())
            cand = all_items - used - seen
            if not cand:
                cand = all_items - used
            cand_pool = list(cand)
            rng.shuffle(cand_pool)
            need = N - len(chosen)
            chosen.extend(cand_pool[:need])

        succ_lists.append(chosen[:N])

    out = df_train.copy()
    out['succ_topN'] = [','.join(map(str, xs)) for xs in succ_lists]

    if save_csv:
        save_dir = Path('./saved') / dataset_name / 'SuccessorTopN' / f'L_{window_size}'
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f'train_succ_top{N}.csv'
        out.to_csv(out_path, index=False)
        print(f'已保存：{out_path}')

    return out