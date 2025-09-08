import argparse
import torch
import random
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
from src.datasets.dataloader import get_dataloaders
from src.datasets.kg_dataloader import get_kg_data
from src.models.sasrec import SASRec
from src.models.gru4rec import GRU4Rec
from src.models.ksr import KSR
from src.models.katrec import KATRec
from src.models.bert4rec import BERT4Rec
from src.trainers.trainer import Trainer
from src.utils.logging_utils import setup_logging
from src.utils.memory_monitor import setup_memory_efficient_environment, MemoryMonitor
from src.utils.code_snapshot import write_code_snapshot
from src.utils.path_finder import PathFinder

def main(config):
    startup_config = {k: v for (k, v) in vars(config).items()}
    setattr(config, '_startup_config', startup_config)

    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    setup_memory_efficient_environment()
    _mm = MemoryMonitor(config)
    _mm.print_memory_stats('Before dataloader')
    _mm.auto_adjust_config()

    pf = PathFinder(config.dataset_name, config.window_size)
    config.save_dir = pf.save_dir(config.model_name)

    try:
        project_root = Path(__file__).resolve().parent
        write_code_snapshot(save_dir=config.save_dir, project_root=project_root)
    except Exception as e:
        print(f'[WARN] Code snapshot failed: {e}')

    # 仅控制台日志，不写 main.log 文件
    logger = setup_logging(level=getattr(config, 'log_level', 'INFO'),
                           file_path=None,  # <<< 不再写入 main.log
                           name='main')

    logger.info('===== Loading Data =====')
    (train_loader, valid_loader, test_loader, num_items) = get_dataloaders(config)
    config.num_items = num_items

    kg_data = None
    if config.model_name.lower() in ['ksr', 'katrec']:
        logger.info(f"Model '{config.model_name}' requires Knowledge Graph data. Loading...")
        kg_data = get_kg_data(config)
        config.num_entities = kg_data['num_entities']
        config.num_relations = kg_data['num_relations']
        if 'item_entity_map' in kg_data:
            config.item_entity_map = kg_data['item_entity_map']
        if config.model_name.lower() == 'katrec':
            config.num_users = kg_data['num_users']
            config.adj_matrix = kg_data['adj_matrix']
        elif config.model_name.lower() == 'ksr':
            config.item_relation_map_tensor = kg_data['item_relation_map_tensor']
    else:
        logger.info('Model does not require KG data.')

    logger.info('===== Initializing Model =====')
    model_name_lower = config.model_name.lower()

    
    if model_name_lower == 'sasrec':
        model = SASRec(num_items=num_items,
                       embedding_dim=config.embedding_dim,
                       max_len=config.window_size,
                       num_attention_heads=config.num_attention_heads,
                       num_blocks=config.num_blocks,
                       dropout_prob=config.dropout_prob)
    elif model_name_lower == 'gru4rec':
        model = GRU4Rec(num_items=num_items,
                        embedding_dim=config.embedding_dim,
                        hidden_size=config.hidden_size,
                        num_layers=config.num_layers,
                        dropout_prob=config.dropout_prob)
    elif model_name_lower == 'ksr':
        model = KSR(num_items=num_items, config=config)
    elif model_name_lower == 'katrec':
        model = KATRec(num_items=num_items, config=config)
    elif model_name_lower == 'bert4rec':
            model = BERT4Rec(
            num_items=num_items,
            embedding_dim=config.embedding_dim,
            max_len=config.window_size,
            num_blocks=config.num_blocks,
            num_attention_heads=config.num_attention_heads,
            dropout_prob=config.dropout_prob,
            pooling=getattr(config, 'pooling', 'last')
        )
    else:
        raise ValueError(f'Unknown model name: {config.model_name}')

    total_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    logger.info(f'Model: {config.model_name}, Total Parameters: {total_params:,}')

    logger.info('===== Initializing Trainer =====')
    trainer = Trainer(model, train_loader, valid_loader, test_loader, config)
    trainer.fit()

    logger.info('===== Process Finished =====')
    logger.info(f'Results and logs saved in: {config.save_dir}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main script for training sequential recommendation models.')
    parser.add_argument('--config', type=str, help='Path to a YAML config file.')
    parser.add_argument('--model_name', type=str, default='SASRec')
    parser.add_argument('--dataset_name', type=str, default='ml-1m')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_level', type=str, default='INFO', help='DEBUG/INFO/WARNING/ERROR')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--dropout_prob', type=float, default=0.2)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=2)
    parser.add_argument('--kg_embedding_dim', type=int, default=64)
    parser.add_argument('--freeze_kg', type=bool, default=False)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--n_gcn_layers', type=int, default=2)
    parser.add_argument('--mess_dropout_prob', type=float, default=0.1)
    parser.add_argument('--kg_loss_lambda', type=float, default=0.1)
    parser.add_argument('--regs', type=str, default='[1e-5, 1e-5]')

    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        parser.set_defaults(**config_dict)
        args = parser.parse_args()
    if isinstance(args.regs, str):
        args.regs = eval(args.regs)

    print('=' * 20 + ' Configuration ' + '=' * 20)
    for (k, v) in vars(args).items():
        print(f'{k}: {v}')
    print('=' * 55)
    main(args)