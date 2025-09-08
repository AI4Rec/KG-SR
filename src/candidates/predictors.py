from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Union
import yaml
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from src.utils.gpu_utils import select_device
from src.utils.path_finder import PathFinder
from src.datasets.dataloader import RecDenoisingDataset
from src.datasets.kg_dataloader import get_kg_data
from src.models.sasrec import SASRec
from src.models.gru4rec import GRU4Rec
from src.models.ksr import KSR
from src.models.katrec import KATRec

MODEL_BUILDERS = {
    'sasrec': lambda cfg, n_items: SASRec(
        num_items=n_items,
        embedding_dim=cfg.embedding_dim,
        max_len=cfg.window_size,
        num_attention_heads=cfg.num_attention_heads,
        num_blocks=cfg.num_blocks,
        dropout_prob=cfg.dropout_prob
    ),
    'gru4rec': lambda cfg, n_items: GRU4Rec(
        num_items=n_items,
        embedding_dim=cfg.embedding_dim,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout_prob=cfg.dropout_prob
    ),
    'ksr': lambda cfg, n_items: KSR(num_items=n_items, config=cfg),
    'katrec': lambda cfg, n_items: KATRec(num_items=n_items, config=cfg)
}

def _load_yaml(path: Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def _load_checkpoint(ckpt_path: Path, map_location: Union[str, torch.device]):
    state = torch.load(ckpt_path, map_location=map_location)
    return state

def _rebuild_config_from_yaml(cfg_dict: dict):
    class C:
        ...
    c = C()
    for (k, v) in cfg_dict.items():
        setattr(c, k, v)
    return c

def _find_model_dirs(dataset_name: str, window_size: int, model_name: Optional[str]) -> List[Path]:
    base = Path('./saved') / dataset_name
    if not base.exists():
        return []
    dirs: List[Path] = []
    if model_name and model_name.upper() != 'ALL':
        cand = base / model_name / f'L_{window_size}'
        if (cand / 'best_model.pth.tar').exists():
            dirs.append(cand)
        return dirs
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        d = sub / f'L_{window_size}'
        if (d / 'best_model.pth.tar').exists():
            dirs.append(d)
    return dirs

def _build_model_from_dir(model_dir: Path, device: torch.device):
    cfg_yaml = model_dir / 'config.yaml'
    ckpt = model_dir / 'best_model.pth.tar'
    assert cfg_yaml.exists() and ckpt.exists(), f'缺少文件：{cfg_yaml} 或 {ckpt}'
    cfg_dict = _load_yaml(cfg_yaml)
    cfg = _rebuild_config_from_yaml(cfg_dict)

    pf = PathFinder(cfg.dataset_name, cfg.window_size)

    # === 用 link 统计物品数（替代 item_map.csv） ===
    link_path = pf.processed_dir() / f'{cfg.dataset_name}.link'
    link_df = pd.read_csv(link_path, sep='\t', dtype={'item_id:token': 'int32'})
    num_items = int(link_df['item_id:token'].max())

    need_kg = str(cfg.model_name).lower() in ['ksr', 'katrec']
    if need_kg:
        kg_data = get_kg_data(cfg)
        cfg.num_entities = kg_data['num_entities']
        cfg.num_relations = kg_data['num_relations']
        if 'item_entity_map' in kg_data:
            cfg.item_entity_map = kg_data['item_entity_map']
        if str(cfg.model_name).lower() == 'katrec':
            cfg.num_users = kg_data['num_users']
            cfg.adj_matrix = kg_data['adj_matrix']
        elif str(cfg.model_name).lower() == 'ksr':
            cfg.item_relation_map_tensor = kg_data['item_relation_map_tensor']

    builder = MODEL_BUILDERS[str(cfg.model_name).lower()]
    model = builder(cfg, num_items)
    state = _load_checkpoint(ckpt, map_location=device)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model.to(device).eval()
    return (model, cfg, num_items)

@torch.no_grad()
def predict_topk_for_train_sequences(
    dataset_name: str,
    window_size: int,
    model_name: Optional[str],
    K: int,
    batch_size: int=512,
    num_workers: int=2,
    save_csv: bool=True
) -> Dict[str, pd.DataFrame]:
    device = select_device(prefer='auto', strategy='max_free', min_free_mem_gb=0.5, allow_cpu=True, verbose=True)
    pf = PathFinder(dataset_name, window_size)
    train_csv = pf.split_dir_path() / 'train.csv'
    assert train_csv.exists(), f'未找到训练集：{train_csv}（请先运行 preprocess.py）'

    dataset = RecDenoisingDataset(train_csv)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    model_dirs = _find_model_dirs(dataset_name, window_size, model_name)
    assert model_dirs, f'未发现已训练模型（saved/{dataset_name}/**/L_{window_size}/best_model.pth.tar）'

    outputs: Dict[str, pd.DataFrame] = {}
    for mdir in model_dirs:
        mname = mdir.parent.name
        print(f'==> 载入模型：{mname} from {mdir}')
        (model, cfg, num_items) = _build_model_from_dir(mdir, device)

        all_preds: List[List[int]] = []
        progress = tqdm(loader, desc=f'Predict[{mname}]')
        for batch in progress:
            seq: torch.Tensor = batch['sequence'].to(device, non_blocking=True)
            logits: torch.Tensor = model(seq)

            with torch.no_grad():
                seq_items = torch.clamp(seq, min=0)
                mask = torch.zeros((seq.size(0), num_items), dtype=torch.bool, device=device)
                nonzero = seq_items > 0
                if nonzero.any():
                    idx_0based = torch.clamp(seq_items[nonzero] - 1, min=0)
                    row_idx = nonzero.nonzero(as_tuple=False)[:, 0]
                    mask[row_idx, idx_0based] = True
                logits = logits.masked_fill(mask, float('-inf'))

                k = min(K, num_items)
                topk_idx = torch.topk(logits, k=k, dim=1, largest=True, sorted=True).indices
                topk_items = (topk_idx + 1).detach().cpu().tolist()
                all_preds.extend(topk_items)

        df_in = pd.read_csv(train_csv)
        assert len(df_in) == len(all_preds), '预测条数与训练集不一致'
        out = df_in.copy()
        out['topk_pred'] = [','.join(map(str, x)) for x in all_preds]

        if save_csv:
            out_path = mdir / f'train_top{K}_pred.csv'
            out.to_csv(out_path, index=False)
            print(f'已保存：{out_path}')

        outputs[mname] = out

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return outputs