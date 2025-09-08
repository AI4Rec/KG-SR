from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

class DataPaths(NamedTuple):
    base_processed: Path
    split_dir: Path
    train_csv: Path
    valid_csv: Path
    test_csv: Path
    processed_link: Path
    processed_user: Path
    raw_inter: Path
    raw_link: Path
    raw_item: Path
    raw_kg: Path

@dataclass(frozen=True)
class PathFinder:
    dataset_name: str
    window_size: int
    project_root: Optional[Path] = None

    def root(self) -> Path:
        if self.project_root is not None:
            return self.project_root.resolve()
        return Path('.').resolve()

    def data_dir(self) -> Path:
        return self.root() / 'dataset'

    def raw_dir(self) -> Path:
        return self.data_dir() / 'raw' / self.dataset_name

    def processed_dir(self) -> Path:
        return self.data_dir() / 'processed' / self.dataset_name

    def split_dir_path(self) -> Path:
        return self.processed_dir() / f'L_{self.window_size}'

    def raw_inter_path(self) -> Path:
        return self.raw_dir() / f'{self.dataset_name}.inter'

    def raw_link_path(self) -> Path:
        return self.raw_dir() / f'{self.dataset_name}.link'

    def raw_item_path(self) -> Path:
        return self.raw_dir() / f'{self.dataset_name}.item'

    def raw_kg_path(self) -> Path:
        return self.raw_dir() / f'{self.dataset_name}.kg'

    # 新增：处理后产物路径
    def processed_link_path(self) -> Path:
        return self.processed_dir() / f'{self.dataset_name}.link'

    def processed_user_path(self) -> Path:
        return self.processed_dir() / f'{self.dataset_name}.user'

    def train_valid_test_paths(
        self,
        use_corrected_train: bool=False,
        corrected_train_filename: str='corrected_train.csv'
    ) -> Tuple[Path, Path, Path]:
        split = self.split_dir_path()
        corrected = split / corrected_train_filename
        train_default = split / 'train.csv'
        train = corrected if use_corrected_train and corrected.exists() else train_default
        return (train, split / 'valid.csv', split / 'test.csv')

    def data_paths(
        self,
        use_corrected_train: bool=False,
        corrected_train_filename: str='corrected_train.csv',
        ensure_dirs: bool=True
    ) -> DataPaths:
        base_processed = self.processed_dir()
        split_dir = self.split_dir_path()
        if ensure_dirs:
            base_processed.mkdir(parents=True, exist_ok=True)
            split_dir.mkdir(parents=True, exist_ok=True)
        (train_csv, valid_csv, test_csv) = self.train_valid_test_paths(
            use_corrected_train=use_corrected_train,
            corrected_train_filename=corrected_train_filename
        )
        return DataPaths(
            base_processed=base_processed,
            split_dir=split_dir,
            train_csv=train_csv,
            valid_csv=valid_csv,
            test_csv=test_csv,
            processed_link=self.processed_link_path(),
            processed_user=self.processed_user_path(),
            raw_inter=self.raw_inter_path(),
            raw_link=self.raw_link_path(),
            raw_item=self.raw_item_path(),
            raw_kg=self.raw_kg_path()
        )

    def save_dir(self, model_name: str) -> Path:
        out = Path('./saved') / self.dataset_name / model_name / f'L_{self.window_size}'
        out.mkdir(parents=True, exist_ok=True)
        return out