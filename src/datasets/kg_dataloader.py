import pandas as pd
from pathlib import Path
import torch
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from src.utils.path_finder import PathFinder

def get_adj_matrix(dataset_name, num_users, num_entities, item_entity_map, kg_df):
    print('--- Building Adjacency Matrix for KATRec ---')
    t1 = pd.Timestamp.now()

    adj_mat = sp.lil_matrix((num_users + num_entities, num_users + num_entities), dtype=np.float32)

    inter_path = PathFinder(dataset_name, window_size=0).processed_dir() / f'{dataset_name}.inter'
    inter_df = pd.read_csv(inter_path, sep='\t', usecols=[0, 1], names=['user_id', 'item_id'], header=0)
    inter_df['entity_id'] = inter_df['item_id'].map(item_entity_map)
    inter_df.dropna(subset=['entity_id'], inplace=True)
    inter_df['entity_id'] = inter_df['entity_id'].astype(int)

    rows = inter_df['user_id'].values
    cols = inter_df['entity_id'].values + num_users
    adj_mat[rows, cols] = 1
    adj_mat[cols, rows] = 1
    print(f'Added {len(inter_df)} user-item interactions to matrix.')

    rows = kg_df['head'].values.astype(int) + num_users
    cols = kg_df['tail'].values.astype(int) + num_users
    adj_mat[rows, cols] = 1
    adj_mat[cols, rows] = 1
    print(f'Added {len(kg_df)} KG triples to matrix.')

    adj_mat = adj_mat.tocsr()
    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    norm_adj = adj_mat.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col))).long()
    values = torch.from_numpy(norm_adj.data)
    shape = torch.Size(norm_adj.shape)
    norm_adj_tensor = torch.sparse.FloatTensor(indices, values, shape)

    t2 = pd.Timestamp.now()
    print(f'Adjacency matrix created and normalized in {(t2 - t1).total_seconds():.2f}s. Shape: {norm_adj.shape}')
    return norm_adj_tensor

def get_kg_data(config):
    dataset_name = config.dataset_name
    pf = PathFinder(dataset_name, getattr(config, 'window_size', 50))
    processed_dir = pf.processed_dir()

    # 以 link / user 作为唯一真相（替代 item_map.csv / user_map.csv）
    link_path = processed_dir / f'{dataset_name}.link'
    user_path = processed_dir / f'{dataset_name}.user'
    kg_path = processed_dir / f'{dataset_name}.kg'

    if not link_path.exists() or not user_path.exists() or not kg_path.exists():
        raise FileNotFoundError(
            f'Required files not found. Please run preprocess.py first. '
            f'Missing one of: {link_path}, {user_path}, {kg_path}'
        )

    print('--- Loading KG and Link/User data ---')

    # 用户数
    user_df = pd.read_csv(user_path, sep='\t')
    # 兼容仅一列或两列
    if 'user_id:token' in user_df.columns:
        num_users = int(user_df.shape[0])  # 连续 0..N-1
    else:
        # 回退：若无列名，尝试第一列
        num_users = int(user_df.iloc[:, 0].nunique())

    # 物品-实体映射
    link_df = pd.read_csv(link_path, sep='\t', dtype={'item_id:token': 'int32', 'entity_id:token': 'string'})
    inferred_num_items = int(link_df['item_id:token'].max())
    num_items_local = int(getattr(config, 'num_items', inferred_num_items))
    if num_items_local < inferred_num_items:
        print(f'[KG] Warning: config.num_items ({num_items_local}) < inferred ({inferred_num_items}). Using inferred value.')
        num_items_local = inferred_num_items
        try:
            setattr(config, 'num_items', num_items_local)
        except Exception:
            pass

    # 读取 processed KG
    kg_df_raw = pd.read_csv(kg_path, sep='\t')
    # 统一列名到 head/relation/tail（原文件头：head_id:token / relation:token / tail_id:token）
    kg_df_raw.columns = ['head', 'relation', 'tail']

    # 建实体/关系词典
    all_entities_str = sorted(list(set(kg_df_raw['head']).union(set(kg_df_raw['tail']))))
    entity_map = {e: i for (i, e) in enumerate(all_entities_str)}
    all_relations_str = sorted(list(set(kg_df_raw['relation'])))
    relation_map = {r: i + 1 for (i, r) in enumerate(all_relations_str)}  # 0 预留 padding
    num_entities = len(entity_map)
    num_relations = len(relation_map) + 1

    # item_id(1-based) -> 实体id(int)，来自 link
    item_id_to_entity_str = pd.Series(
        link_df['entity_id:token'].values,
        index=link_df['item_id:token'].values
    ).to_dict()
    item_entity_map = {
        int(item_id): entity_map[ent_str]
        for (item_id, ent_str) in item_id_to_entity_str.items()
        if isinstance(ent_str, str) and ent_str in entity_map
    }

    # KG 映射为 index
    kg_df_mapped = kg_df_raw.copy()
    kg_df_mapped['head'] = kg_df_mapped['head'].map(entity_map)
    kg_df_mapped['relation'] = kg_df_mapped['relation'].map(relation_map)
    kg_df_mapped['tail'] = kg_df_mapped['tail'].map(entity_map)
    kg_df_mapped.dropna(inplace=True)
    kg_df_mapped[['head', 'relation', 'tail']] = kg_df_mapped[['head', 'relation', 'tail']].astype(int)

    # 为 KSR 构建 item->topR relations（可选）
    R_MAX = getattr(config, 'max_relations_per_item', 8)
    if R_MAX < 1:
        R_MAX = 1

    head_rel_counts = kg_df_mapped.groupby(['head', 'relation']).size().reset_index(name='cnt')
    rel_list_by_head = {}
    for (head, grp) in head_rel_counts.groupby('head'):
        grp_sorted = grp.sort_values('cnt', ascending=False)
        rel_list_by_head[head] = grp_sorted['relation'].tolist()

    item_relation_map_tensor = torch.zeros(num_items_local, R_MAX, dtype=torch.long)
    filled_items = 0
    for (item_id_1, ent_id) in item_entity_map.items():
        zero_idx = int(item_id_1) - 1
        if zero_idx < 0 or zero_idx >= num_items_local:
            continue
        if ent_id < 0 or ent_id >= num_entities:
            continue
        rels = rel_list_by_head.get(ent_id, [])
        if not rels:
            continue
        rels = rels[:R_MAX]
        item_relation_map_tensor[zero_idx, :len(rels)] = torch.tensor(rels, dtype=torch.long)
        filled_items += 1
    print(f'[KSR] Built item_relation_map_tensor: shape={tuple(item_relation_map_tensor.shape)}, filled_items={filled_items}, R_MAX={R_MAX}')

    # 打包不同模型需求
    if config.model_name.lower() in ['ksr']:
        return {
            'num_users': num_users,
            'num_entities': num_entities,
            'num_relations': num_relations,
            'item_entity_map': item_entity_map,
            'item_relation_map_tensor': item_relation_map_tensor,
            'kg_triples_df': kg_df_mapped
        }
    elif config.model_name.lower() == 'katrec':
        adj_matrix = get_adj_matrix(dataset_name, num_users, num_entities, item_entity_map, kg_df_mapped)
        return {
            'num_users': num_users,
            'num_entities': num_entities,
            'num_relations': num_relations,
            'adj_matrix': adj_matrix,
            'item_entity_map': item_entity_map,
            'kg_triples_df': kg_df_mapped
        }
    return {}