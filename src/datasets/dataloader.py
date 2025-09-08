import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import gc
import numpy as np
import random
from src.utils.path_finder import PathFinder

class RecDenoisingDataset(Dataset):

    def __init__(self, data_path):
        dtypes = {'user_id': 'int32', 'sequence': 'string', 'target_id': 'int32'}
        self.data_df = pd.read_csv(data_path, dtype=dtypes)
        print(f'Loaded {len(self.data_df)} samples from {data_path}')
        gc.collect()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        user_id = int(row['user_id'])
        sequence = list(map(int, row['sequence'].split(',')))
        target_id = int(row['target_id'])
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'sequence': torch.tensor(sequence, dtype=torch.long),
            'target_id': torch.tensor(target_id, dtype=torch.long)
        }

def get_dataloaders(config):
    pf = PathFinder(config.dataset_name, config.window_size)
    dp = pf.data_paths(
        use_corrected_train=bool(getattr(config, 'use_corrected_train', False)),
        corrected_train_filename=getattr(config, 'corrected_train_filename', 'corrected_train.csv'),
        ensure_dirs=True
    )
    (train_path, valid_path, test_path) = (dp.train_csv, dp.valid_csv, dp.test_csv)

    if not all([train_path.exists(), valid_path.exists(), test_path.exists()]):
        raise FileNotFoundError(
            f'One or more processed data files not found in {dp.split_dir}. '
            f'Please run preprocess.py first with --window_size {config.window_size}'
        )

    train_dataset = RecDenoisingDataset(train_path)
    valid_dataset = RecDenoisingDataset(valid_path)
    test_dataset = RecDenoisingDataset(test_path)

    # === 用 processed link 统计物品数（取代 item_map.csv） ===
    link_path = dp.processed_link
    if not link_path.exists():
        raise FileNotFoundError(f'Processed link file not found: {link_path}. Please run preprocess.py first.')

    link_df = pd.read_csv(link_path, sep='\t', dtype={'item_id:token': 'int32'})
    num_items = int(link_df['item_id:token'].max())  # 1..N，0 为 PAD
    del link_df
    gc.collect()

    optimized_num_workers = min(getattr(config, 'num_workers', 0), 2)
    optimized_batch_size = getattr(config, 'optimized_batch_size', getattr(config, 'batch_size', 256))
    seed = int(getattr(config, 'seed', 42))
    g = torch.Generator()
    g.manual_seed(seed)

    def _worker_init_fn(worker_id):
        s = seed + worker_id
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)

    common_loader_args = {
        'batch_size': optimized_batch_size,
        'num_workers': optimized_num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': optimized_num_workers > 0,
        'worker_init_fn': _worker_init_fn if optimized_num_workers > 0 else None,
        'generator': g
    }
    if optimized_num_workers > 0:
        common_loader_args['prefetch_factor'] = 2

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **common_loader_args)
    valid_loader = DataLoader(valid_dataset, shuffle=False, drop_last=False, **common_loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False, **common_loader_args)

    print(f'Number of items (EXCLUDING padding): {num_items}')
    print(f'Optimized batch size: {optimized_batch_size}')
    print(f'Optimized num workers: {optimized_num_workers}')
    print(f'Train loader: {len(train_loader.dataset)} samples, {len(train_loader)} batches.')
    print(f'Validation loader: {len(valid_loader.dataset)} samples, {len(valid_loader)} batches.')
    print(f'Test loader: {len(test_loader.dataset)} samples, {len(test_loader)} batches.')

    return (train_loader, valid_loader, test_loader, num_items)