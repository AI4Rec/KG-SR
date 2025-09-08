import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from src.utils.path_finder import PathFinder

class DataPreprocessor:

    def __init__(self, config):
        self.config = config
        self.dataset_name = config.dataset_name
        self.window_size = config.window_size
        self.min_core = config.min_core
        self.max_len = config.max_len

        self.pf = PathFinder(self.dataset_name, self.window_size)
        dp = self.pf.data_paths(ensure_dirs=True)

        self.data_dir = self.pf.data_dir()
        self.raw_data_dir = self.pf.raw_dir()
        self.base_processed_dir = dp.base_processed
        self.split_data_dir = dp.split_dir

        self.raw_file_path = dp.raw_inter
        self.link_file_path = dp.raw_link
        self.item_file_path = dp.raw_item

        print('#' * 20, ' Configuration ', '#' * 20)
        print(f'Dataset Name: {self.dataset_name}')
        print(f'Raw data file: {self.raw_file_path}')
        print(f'Entity link file: {self.link_file_path}')
        print(f'Raw item file: {self.item_file_path}')
        print(f'Base processed data path: {self.base_processed_dir}')
        print(f'L-dependent splits path: {self.split_data_dir}')
        print(f'Sliding Window Size (L): {self.window_size}')
        print(f'Min-Core for Filtering (K): {self.min_core}')
        print(f'Max History Length (for truncation): {self.max_len}')
        print('#' * 75)

    def _apply_k_core_filtering(self, df, min_core):
        print(f'\n--- Step 1a: Applying {min_core}-Core Filtering ---')
        while True:
            initial_rows = len(df)
            user_counts = df['user_id'].value_counts(dropna=False)
            valid_users = user_counts[user_counts >= min_core].index
            df = df[df['user_id'].isin(valid_users)]

            item_counts = df['item_id'].value_counts(dropna=False)
            valid_items = item_counts[item_counts >= min_core].index
            df = df[df['item_id'].isin(valid_items)]

            if len(df) == initial_rows:
                break
        print(f'Filtering complete. Final data has {len(df)} interactions.')
        return df

    def run(self):
        print('\n--- Step 1: Loading Data ---')
        df = pd.read_csv(self.raw_file_path, sep='\t', header=0, engine='python')
        df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        df['user_id'] = df['user_id'].astype(str)
        df['item_id'] = df['item_id'].astype(str)
        print(f'Original data loaded: {len(df)} interactions.')

        # k-core
        df = self._apply_k_core_filtering(df, self.min_core)
        print(f"User count after filtering: {df['user_id'].nunique()}")
        print(f"Item count after filtering: {df['item_id'].nunique()}")

        df = df.sort_values(by=['user_id', 'timestamp'], ascending=True)

        # --- User remap ---
        print('\n--- Step 2: Create and Apply User ID Mapping ---')
        unique_users = df['user_id'].unique()
        sorted_unique_users = sorted(unique_users, key=lambda x: int(x) if x.isdigit() else x)
        user_map = {original_id: new_id for (new_id, original_id) in enumerate(sorted_unique_users)}
        df['user_id'] = df['user_id'].map(user_map)
        print(f'Created and applied mapping for {len(user_map)} users.')

        # --- Item remap (顺序出现分配，保证紧凑 1..N；0 为 PAD) ---
        print('\n--- Step 3: Create and Apply Item ID Mapping ---')
        item_map = {}
        next_item_id = 1
        user_sequences_original_items = df.groupby('user_id')['item_id'].apply(list)
        sorted_mapped_user_ids = sorted(user_sequences_original_items.index.tolist())
        for mapped_user_id in tqdm(sorted_mapped_user_ids, desc='Creating Item Map'):
            sequence = user_sequences_original_items[mapped_user_id]
            for original_item_id in sequence:
                if original_item_id not in item_map:
                    item_map[original_item_id] = next_item_id
                    next_item_id += 1
        df['item_id'] = df['item_id'].map(item_map)
        print(f'Created and applied mapping for {len(item_map)} items (IDs from 1 to {next_item_id - 1}). Token 0 is reserved for padding.')

        # --- 存 remapped inter ---
        print('\n--- Step 4: Saving Remapped Interaction File ---')
        remapped_inter_path = self.base_processed_dir / f'{self.dataset_name}.inter'
        custom_header = ['user_id:token', 'item_id:token', 'rating:float', 'timestamp:float']
        df.to_csv(remapped_inter_path, sep='\t', index=False, header=custom_header)
        print(f'Remapped interaction data saved to {remapped_inter_path}')

        # --- LOO 样本 ---
        print('\n--- Step 5: Generating and Truncating User Interaction Sequences ---')
        df = df.sort_values(by=['user_id', 'timestamp'], ascending=True)
        user_sequences_mapped = df.groupby('user_id')['item_id'].apply(list)
        user_sequences_mapped = user_sequences_mapped.apply(lambda seq: seq[-self.max_len:])
        print(f'Generated and truncated sequences for {len(user_sequences_mapped)} users (max length = {self.max_len}).')

        print('\n--- Step 6: Applying Padding and Generating Samples (Leave-One-Out) ---')
        (train_data, valid_data, test_data) = ([], [], [])
        PAD_TOKEN = 0
        for (user_id, sequence) in tqdm(user_sequences_mapped.items(), desc='Processing sequences'):
            if len(sequence) < 3:
                continue
            # test
            test_hist = sequence[-self.window_size - 1:-1]
            test_padded_hist = [PAD_TOKEN] * (self.window_size - len(test_hist)) + test_hist
            test_data.append({
                'user_id': user_id,
                'sequence': ','.join(map(str, test_padded_hist)),
                'target_id': sequence[-1]
            })
            # valid
            valid_hist = sequence[-self.window_size - 2:-2]
            valid_padded_hist = [PAD_TOKEN] * (self.window_size - len(valid_hist)) + valid_hist
            valid_data.append({
                'user_id': user_id,
                'sequence': ','.join(map(str, valid_padded_hist)),
                'target_id': sequence[-2]
            })
            # train
            train_seq = sequence[:-2]
            if len(train_seq) < 2:
                continue
            for i in range(1, len(train_seq)):
                target_id = train_seq[i]
                history = train_seq[max(0, i - self.window_size):i]
                padded_history = [PAD_TOKEN] * (self.window_size - len(history)) + history
                train_data.append({
                    'user_id': user_id,
                    'sequence': ','.join(map(str, padded_history)),
                    'target_id': target_id
                })

        print(f'Generated {len(train_data)} training samples.')
        print(f'Generated {len(valid_data)} validation samples.')
        print(f'Generated {len(test_data)} testing samples.')

        # --- 保存 split & 新产物 ---
        print('\n--- Step 7: Saving Processed Data and Artifacts ---')
        train_path = self.split_data_dir / 'train.csv'
        valid_path = self.split_data_dir / 'valid.csv'
        test_path = self.split_data_dir / 'test.csv'
        df_train = pd.DataFrame(train_data)
        df_valid = pd.DataFrame(valid_data)
        df_test = pd.DataFrame(test_data)
        df_train.to_csv(train_path, index=False); print(f'Train data saved to {train_path}')
        df_valid.to_csv(valid_path, index=False); print(f'Validation data saved to {valid_path}')
        df_test.to_csv(test_path, index=False); print(f'Test data saved to {test_path}')

        # === 取代 user_map.csv：输出重映射后的 .user 文件 ===
        processed_user_path = self.base_processed_dir / f'{self.dataset_name}.user'
        df_user_out = pd.DataFrame({
            'user_id:token': [user_map[k] for k in sorted(user_map.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))],
            'original_id:token': sorted(user_map.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
        })
        df_user_out.to_csv(processed_user_path, sep='\t', index=False)
        print(f'User file (remapped) saved to {processed_user_path}')

        # === 继续生成 link（取代 item_map.csv 的功能） ===
        try:
            df_link_raw = pd.read_csv(
                self.link_file_path, sep='\t', header=0,
                names=['original_id', 'entity_id'],
                dtype={'original_id': str, 'entity_id': str}
            )
            print('Loaded raw link file.')
        except FileNotFoundError:
            print(f'Warning: Entity link file not found at {self.link_file_path}.')
            df_link_raw = pd.DataFrame(columns=['original_id', 'entity_id'])

        # merge 映射，构造 entity_id，占位补齐
        df_item_map = pd.DataFrame(item_map.items(), columns=['original_id', 'mapped_id'])
        df_item_map = pd.merge(df_item_map, df_link_raw, on='original_id', how='left')

        print('\n--- Step 7a: Generating Placeholder Entity IDs for Missing Items ---')
        missing_entity_mask = df_item_map['entity_id'].isnull()
        num_missing = int(missing_entity_mask.sum())
        if num_missing > 0:
            mapped_ids_to_impute = df_item_map.loc[missing_entity_mask, 'mapped_id']
            placeholders = mapped_ids_to_impute.apply(lambda x: f'n.00{x}')
            df_item_map.loc[missing_entity_mask, 'entity_id'] = placeholders
            print(f'Found and filled {num_missing} missing entity IDs with placeholders.')
        else:
            print('No missing entity IDs found.')

        link_export_path = self.base_processed_dir / f'{self.dataset_name}.link'
        link_export_df = df_item_map[['mapped_id', 'entity_id']].copy()
        link_export_df.dropna(subset=['entity_id'], inplace=True)
        link_export_df.rename(columns={'mapped_id': 'item_id:token', 'entity_id': 'entity_id:token'}, inplace=True)
        link_export_df = link_export_df.sort_values('item_id:token')
        link_export_df.to_csv(link_export_path, sep='\t', index=False)
        print(f'Generated new link file at: {link_export_path}')

        # === KG ===
        print('\n--- Step 7b: Loading Knowledge Graph and filtering by head entities from link ---')
        kg_dest_path = self.base_processed_dir / f'{self.dataset_name}.kg'
        kg_src_path = self.pf.raw_kg_path()
        default_kg_columns = ['head_id:token', 'relation:token', 'tail_id:token']

        try:
            df_kg_raw = pd.read_csv(kg_src_path, sep='\t', header=0)
            kg_columns = df_kg_raw.columns.tolist()
            print(f'Loaded raw KG with {len(df_kg_raw)} triples. Columns: {kg_columns}')
        except FileNotFoundError as e:
            print(f'Warning: Raw KG file not found: {e}. Will fallback to fake triples only.')
            kg_columns = default_kg_columns
            df_kg_raw = pd.DataFrame(columns=kg_columns)
        except Exception as e:
            print(f'An error occurred when reading raw KG: {e}. Will fallback to fake triples only.')
            kg_columns = default_kg_columns
            df_kg_raw = pd.DataFrame(columns=kg_columns)

        head_col = (df_kg_raw.columns[0] if not df_kg_raw.empty else default_kg_columns[0])
        allowed_heads = set(link_export_df['entity_id:token'].astype(str).unique())

        if not df_kg_raw.empty:
            df_kg_raw[head_col] = df_kg_raw[head_col].astype(str)
            before_cnt = len(df_kg_raw)
            df_kg_base = df_kg_raw[df_kg_raw[head_col].isin(allowed_heads)].copy()
            after_cnt = len(df_kg_base)
            print(f'Filtered KG by head entities: {before_cnt} -> {after_cnt} triples kept.')
        else:
            df_kg_base = pd.DataFrame(columns=df_kg_raw.columns if not df_kg_raw.empty else default_kg_columns)

        print('\n--- Step 7c: Updating Knowledge Graph with Fake Triples ---')
        all_triples = []
        if not df_kg_base.empty:
            all_triples.append(df_kg_base)

        # 为占位实体加 fake 边
        items_with_placeholders = link_export_df[link_export_df['entity_id:token'].astype(str).str.startswith('n.')]
        fake_triples = []
        if not items_with_placeholders.empty:
            kg_columns_use = (df_kg_base.columns.tolist() if not df_kg_base.empty else default_kg_columns)
            for _, row in items_with_placeholders.iterrows():
                head_entity = str(row['entity_id:token'])
                tail_entity = head_entity.replace('n.', 'f.', 1)
                fake_triples.append({
                    kg_columns_use[0]: head_entity,
                    kg_columns_use[1]: 'fake',
                    kg_columns_use[2]: tail_entity
                })
        if fake_triples:
            df_fake_triples = pd.DataFrame(fake_triples)
            all_triples.append(df_fake_triples)
            print(f'Generated {len(df_fake_triples)} fake triples to be added.')

        if all_triples:
            df_kg_updated = pd.concat(all_triples, ignore_index=True)
            # 规范列名与排序
            cols = df_kg_updated.columns.tolist()
            if len(cols) >= 3:
                rename_map = {}
                expect = default_kg_columns
                for i in range(3):
                    if cols[i] != expect[i]:
                        rename_map[cols[i]] = expect[i]
                if rename_map:
                    df_kg_updated.rename(columns=rename_map, inplace=True)
            sort_cols = default_kg_columns[:2]
            for c in sort_cols:
                df_kg_updated[c] = df_kg_updated[c].astype(str)
            df_kg_updated = df_kg_updated.sort_values(by=sort_cols, ascending=True, kind='mergesort')
            df_kg_updated.to_csv(kg_dest_path, sep='\t', index=False, header=True)
            print(f'Final knowledge graph with {len(df_kg_updated)} triples saved to: {kg_dest_path}')
        else:
            print('No KG triples to save. KG file not created.')

        # --- Remap .item (可选，沿用原逻辑) ---
        print('\n--- Step 7d: Remapping and Filtering Item Metadata (.item) ---')
        remapped_item_path = self.base_processed_dir / f'{self.dataset_name}.item'
        try:
            df_item_raw = pd.read_csv(self.item_file_path, sep='\t', header=0, engine='python')
            df_item_raw.columns = ['item_id:token', 'movie_title:token_seq', 'release_year:token', 'genre:token_seq']
            df_item_raw['item_id:token'] = df_item_raw['item_id:token'].astype(str)
            df_item_remapped = df_item_raw[df_item_raw['item_id:token'].isin(item_map.keys())].copy()
            df_item_remapped['item_id:token'] = df_item_remapped['item_id:token'].map(item_map)
            df_item_remapped = df_item_remapped.sort_values(by='item_id:token')
            df_item_remapped.to_csv(remapped_item_path, sep='\t', index=False)
            print(f'Remapped and filtered item metadata saved to {remapped_item_path}')
        except FileNotFoundError:
            print(f'Warning: Item metadata file not found at {self.item_file_path}.')

        print('\nPreprocessing finished successfully! 🎉')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Preprocessing for Sequential Recommendation')
    parser.add_argument('--dataset_name', type=str, default='ml-1m', help='Name of the dataset')
    parser.add_argument('--window_size', type=int, default=5, help='The size of the sliding window (L)')
    parser.add_argument('--min_core', type=int, default=5, help='The K-core filtering threshold')
    parser.add_argument('--max_len', type=int, default=200, help='The maximum length of user history')
    args = parser.parse_args()

    if args.max_len < args.window_size:
        print('Warning: max_len is less than window_size. Setting max_len to window_size for truncation.')
        args.max_len = args.window_size

    preprocessor = DataPreprocessor(args)
    preprocessor.run()