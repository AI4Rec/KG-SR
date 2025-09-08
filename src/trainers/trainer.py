import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import yaml
import gc
import time
import pandas as pd
import random
from collections import defaultdict
from src.utils.metrics import AllRankMetrics
from src.utils.memory_monitor import setup_memory_efficient_environment, MemoryMonitor
from src.utils.logging_utils import setup_logging
from src.utils.gpu_utils import select_device

class Trainer:

    def __init__(self, model, train_loader, valid_loader, test_loader, config):
        setup_memory_efficient_environment()
        self.memory_monitor = MemoryMonitor(config)
        self.memory_monitor.print_memory_stats('Trainer initialization')

        self.config = config
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.save_dir / 'train.log'
        self.logger = setup_logging(level=getattr(config, 'log_level', 'INFO'), file_path=self.log_file, name='train')

        self.use_amp = bool(getattr(config, 'use_amp', False) and torch.cuda.is_available())
        self.allow_fast_kernels = bool(getattr(config, 'allow_fast_kernels', False) and torch.cuda.is_available())
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = self.allow_fast_kernels
                torch.backends.cudnn.allow_tf32 = self.allow_fast_kernels
            except Exception:
                pass
            try:
                torch.backends.cuda.sdp_kernel(
                    enable_flash=self.allow_fast_kernels,
                    enable_mem_efficient=self.allow_fast_kernels,
                    enable_math=not self.allow_fast_kernels
                )
            except Exception:
                pass
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.device = select_device(
            prefer=getattr(config, 'device_prefer', 'auto'),
            strategy=getattr(config, 'gpu_select', 'max_free'),
            min_free_mem_gb=float(getattr(config, 'min_free_mem_gb', 0.5)),
            allow_cpu=True,
            explicit_id=getattr(config, 'gpu_id') if hasattr(config, 'gpu_id') else None,
            verbose=True
        )
        self.logger.info(f'Using device: {self.device}')

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

        top_k = int(getattr(config, 'top_k', 10))
        self.metrics_calculator = AllRankMetrics(top_k=top_k)
        self.grad_clip = float(getattr(config, 'grad_clip', 1.0))
        self.best_metric = -1
        self.epochs_no_improve = 0

        # === 仅用训练集构建评估掩码（防泄露） ===
        self._prepare_eval_protocol_masks()

        if self.config.model_name.lower() == 'katrec':
            self._setup_katrec_training()

    def _safe_config_dict(self):
        blacklist = {'adj_matrix', 'item_relation_map_tensor', 'item_entity_map'}
        cfg = {}
        for k, v in vars(self.config).items():
            if k in blacklist:
                continue
            try:
                yaml.safe_dump({k: v})
                cfg[k] = v
            except Exception:
                cfg[k] = f'<non-serializable:{type(v).__name__}>'
        return cfg

    # ========= 评估协议准备：只看 train.csv =========
    def _prepare_eval_protocol_masks(self):
        """
        构建两类掩码（仅使用训练集，绝不读取 valid/test 数据）：
          1) 全局非法候选：所有【未在训练中出现过】的物品（冷启动测试物品）
          2) 每用户已见集合：该用户在训练期间出现过的所有物品
        评估时：
          - 屏蔽(1)与(2)
          - 仅对当前样本的 ground-truth 解除屏蔽（允许其参与排名）
        """
        df = self.train_loader.dataset.data_df  # train.csv
        num_items = int(self.config.num_items)

        valid_items_set = set()
        user_seen = defaultdict(set)

        for _, row in df.iterrows():
            uid = int(row['user_id'])

            seq_str = row['sequence']
            if isinstance(seq_str, str):
                for tok in seq_str.split(','):
                    if not tok:
                        continue
                    it = int(tok)
                    if it > 0:
                        valid_items_set.add(it)
                        user_seen[uid].add(it)

            tgt = int(row['target_id'])
            if tgt > 0:
                valid_items_set.add(tgt)
                user_seen[uid].add(tgt)

        # 1D 全局掩码：True = 非法候选（未在训练集中出现过）
        valid_flag = torch.zeros(num_items, dtype=torch.bool)
        for it in valid_items_set:
            if 1 <= it <= num_items:
                valid_flag[it - 1] = True
        invalid_items_mask_1d = ~valid_flag
        self._invalid_items_mask_1d = invalid_items_mask_1d.to(self.device)

        # 每用户训练期已见（0-based 索引）
        max_uid = int(df['user_id'].max()) if len(df) > 0 else -1
        self._user_seen_idx = [torch.empty(0, dtype=torch.long, device=self.device) for _ in range(max_uid + 1)]
        for uid, items in user_seen.items():
            idxs = [it - 1 for it in items if 1 <= it <= num_items]
            if idxs:
                self._user_seen_idx[uid] = torch.as_tensor(sorted(set(idxs)), dtype=torch.long, device=self.device)

        self._log(f'[EvalProto] valid_items={len(valid_items_set)}, users={len(self._user_seen_idx)} ready.')

    # ========= KATRec KG 训练准备 =========
    def _setup_katrec_training(self):
        self.logger.info('Setting up data for KATRec KG training...')
        from src.datasets.kg_dataloader import get_kg_data
        kg_data = get_kg_data(self.config)
        kg_df = kg_data['kg_triples_df']
        self.kg_triples = kg_df[['head', 'relation', 'tail']].astype(int).values
        self.num_entities = kg_data['num_entities']
        self.kg_dict = defaultdict(list)
        for h, r, t in self.kg_triples:
            self.kg_dict[h, r].append(t)
        self.logger.info(f'Prepared {len(self.kg_triples)} triples for KG loss calculation.')

    def _sample_kg_batch(self, batch_size):
        indices = torch.randint(0, len(self.kg_triples), (batch_size,))
        batch_triples = self.kg_triples[indices]
        h, r, pos_t = (batch_triples[:, 0], batch_triples[:, 1], batch_triples[:, 2])
        neg_t = []
        for i in range(batch_size):
            head, rel = (h[i], r[i])
            tries = 0
            while True:
                neg_tail_candidate = random.randint(0, self.num_entities - 1)
                if neg_tail_candidate not in self.kg_dict[head, rel]:
                    neg_t.append(neg_tail_candidate)
                    break
                tries += 1
                if tries > 50:
                    neg_t.append(neg_tail_candidate)
                    break
        return (
            torch.LongTensor(h).to(self.device),
            torch.LongTensor(r).to(self.device),
            torch.LongTensor(pos_t).to(self.device),
            torch.LongTensor(neg_t).to(self.device)
        )

    def _log(self, message):
        print(message)
        self.logger.info(message)

    def _save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self._safe_config_dict()
        }
        if is_best:
            best_filename = self.save_dir / 'best_model.pth.tar'
            torch.save(state, best_filename)
            self._log(f'-> Found new best model on validation set, saving to {best_filename}')
        else:
            pass

    def _save_results_to_csv(self, final_metrics, best_epoch, duration_str):
        results_dir = Path('./results')
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / 'experiment_results.csv'

        results_data = {
            'timestamp': pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': self.config.model_name,
            'dataset_name': self.config.dataset_name,
            f'hit@{self.metrics_calculator.top_k}': final_metrics.get(f'Hit@{self.metrics_calculator.top_k}', -1),
            f'ndcg@{self.metrics_calculator.top_k}': final_metrics.get(f'NDCG@{self.metrics_calculator.top_k}', -1),
            'mrr': final_metrics.get('MRR', -1),
            'best_epoch': best_epoch,
            'total_epochs': self.config.epochs,
            'training_time': duration_str,
            'learning_rate': self.config.learning_rate,
            'batch_size': getattr(self.config, 'optimized_batch_size', self.config.batch_size),
            'embedding_dim': self.config.embedding_dim,
            'dropout_prob': self.config.dropout_prob,
            'window_size': self.config.window_size,
            'save_dir': str(self.save_dir)
        }
        df_results = pd.DataFrame([results_data])
        if not results_file.exists():
            df_results.to_csv(results_file, index=False, header=True)
        else:
            df_results.to_csv(results_file, mode='a', index=False, header=False)

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, total_rec_loss, total_kg_loss = (0.0, 0.0, 0.0)
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{self.config.epochs} [Training]')

        for batch in progress_bar:
            sequences = batch['sequence'].to(self.device, non_blocking=True)
            targets = batch['target_id'].to(self.device, non_blocking=True)
            targets_0 = torch.clamp_min(targets, 1) - 1

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    logits = self.model(sequences)
                    rec_loss = self.criterion(logits, targets_0)
                    if self.config.model_name.lower() == 'katrec':
                        h, r, pos_t, neg_t = self._sample_kg_batch(sequences.size(0))
                        kg_loss = self.model.calculate_kg_loss(h, r, pos_t, neg_t)
                        loss = rec_loss + self.config.kg_loss_lambda * kg_loss
                    else:
                        kg_loss = torch.tensor(0.0, device=self.device)
                        loss = rec_loss
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(sequences)
                rec_loss = self.criterion(logits, targets_0)
                if self.config.model_name.lower() == 'katrec':
                    h, r, pos_t, neg_t = self._sample_kg_batch(sequences.size(0))
                    kg_loss = self.model.calculate_kg_loss(h, r, pos_t, neg_t)
                    loss = rec_loss + self.config.kg_loss_lambda * kg_loss
                else:
                    kg_loss = torch.tensor(0.0, device=self.device)
                    loss = rec_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            total_rec_loss += rec_loss.item()
            total_kg_loss += kg_loss.item()

            postfix = {'loss': f'{loss.item():.4f}', 'rec': f'{rec_loss.item():.4f}'}
            if self.config.model_name.lower() == 'katrec':
                postfix['kg'] = f'{kg_loss.item():.4f}'
            progress_bar.set_postfix(postfix)

        avg_loss = total_loss / len(self.train_loader)
        avg_rec_loss = total_rec_loss / len(self.train_loader)
        avg_kg_loss = total_kg_loss / len(self.train_loader)
        log_msg = f'Epoch {epoch + 1} Training -> Avg Loss: {avg_loss:.4f} (Rec: {avg_rec_loss:.4f}'
        if self.config.model_name.lower() == 'katrec':
            log_msg += f', KG: {avg_kg_loss:.4f})'
        else:
            log_msg += ')'
        self._log(log_msg)

    def _evaluate(self, epoch, loader, phase='Validation'):
        if hasattr(self.model, 'precompute_kg'):
            try:
                self.model.precompute_kg(self.device)
            except Exception:
                pass

        self.model.eval()
        self.metrics_calculator.reset()
        progress_bar = tqdm(loader, desc=f'Epoch {epoch + 1}/{self.config.epochs} [{phase}]')

        with torch.inference_mode():
            for batch in progress_bar:
                sequences = batch['sequence'].to(self.device, non_blocking=True)
                targets = batch['target_id']  # CPU 张量
                targets_0 = torch.clamp_min(targets, 1) - 1

                logits = self.model(sequences)
                num_items = logits.size(1)

                # === “几乎全体物品”评估（严格防止泄露） ===
                # 1) 全局屏蔽：所有未在训练出现的物品（测试/验证专属冷启动物品）
                base_mask = self._invalid_items_mask_1d.view(1, -1).expand(sequences.size(0), -1).clone()

                # 2) 每用户再屏蔽：该用户训练期已见过的所有物品
                if hasattr(self, '_user_seen_idx') and self._user_seen_idx:
                    uids = batch['user_id'].tolist()  # 仍在 CPU，逐个索引即可
                    for i, uid in enumerate(uids):
                        if 0 <= uid < len(self._user_seen_idx):
                            seen_idx = self._user_seen_idx[uid]  # 已在 self.device
                            if seen_idx.numel() > 0:
                                base_mask[i, seen_idx] = True

                # 3) 仅解除当前样本的 ground-truth（允许其参与排序；除此之外不解除任何验证/测试物品）
                gt = targets_0.to(logits.device).long()
                valid = (gt >= 0) & (gt < num_items)
                if valid.any():
                    bs_idx = torch.arange(gt.size(0), device=logits.device)[valid]
                    base_mask[bs_idx, gt[valid]] = False

                # 4) 应用掩码
                logits = logits.masked_fill(base_mask, float('-inf')).detach().cpu()

                # 只统计 target > 0 的样本
                keep = targets > 0
                if keep.any():
                    self.metrics_calculator(logits[keep], (targets_0[keep]).to(torch.long))

        metrics = self.metrics_calculator.summary()
        k = self.metrics_calculator.top_k
        log_str = (
            f"Epoch {epoch + 1} {phase} Results -> "
            f"Hit@{k}: {metrics[f'Hit@{k}']:.4f}, "
            f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}, "
            f"MRR: {metrics['MRR']:.4f}"
        )
        self._log(log_str)
        return metrics

    def fit(self):
        self._log('=' * 20 + ' Starting Training ' + '=' * 20)

        startup_cfg = getattr(self.config, '_startup_config', None)
        to_dump = startup_cfg if isinstance(startup_cfg, dict) else self._safe_config_dict()
        cfg_path = self.save_dir / 'config.yaml'
        with open(cfg_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(to_dump, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        start_time = time.time()
        best_epoch_num = 0
        try:
            for epoch in range(self.config.epochs):
                self._train_epoch(epoch)
                metrics = self._evaluate(epoch, loader=self.valid_loader, phase='Validation')
                current_metric = metrics[f'NDCG@{self.metrics_calculator.top_k}']

                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.epochs_no_improve = 0
                    best_epoch_num = epoch + 1
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_no_improve += 1
                    self._log(f'-> No improvement in {self.epochs_no_improve} epochs.')
                    self._save_checkpoint(epoch, is_best=False)

                if self.epochs_no_improve >= self.config.patience:
                    self._log(f'Early stopping triggered after {self.config.patience} epochs.')
                    break
        finally:
            gc.collect()

        duration_str = time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))
        self._log(f'Total Training Time: {duration_str}')
        self._log('=' * 20 + ' Training Finished ' + '=' * 20)

        self._log('\n' + '=' * 20 + ' Final Evaluation on Test Set ' + '=' * 20)
        best_model_path = self.save_dir / 'best_model.pth.tar'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self._log(f'Loaded best model from epoch {best_epoch_num}')
            if hasattr(self.model, 'precompute_kg'):
                try:
                    self.model.precompute_kg(self.device)
                except Exception:
                    pass
            final_metrics = self._evaluate(checkpoint['epoch'], loader=self.test_loader, phase='Testing')
            self._save_results_to_csv(final_metrics, best_epoch_num, duration_str)
        else:
            self._log('No best model found to evaluate.')