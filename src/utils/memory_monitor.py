# src/utils/memory_monitor.py

import psutil
import torch
import os
import gc
from pathlib import Path

class MemoryMonitor:
    """轻量内存监控与建议/自适应调参（保持对外 API 不变）"""
    def __init__(self, config):
        self.config = config
        self.initial_memory = self.get_memory_usage()

    def get_memory_usage(self):
        mem = psutil.virtual_memory()
        info = {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent
        }
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory
            info.update({
                'gpu_total_gb': total / (1024**3),
                'gpu_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3),
                'gpu_cached_gb': torch.cuda.memory_reserved(0) / (1024**3),
                'gpu_free_gb': (total - torch.cuda.memory_reserved(0)) / (1024**3)
            })
        return info

    def print_memory_stats(self, stage=""):
        m = self.get_memory_usage()
        print(f"\n=== 内存使用 {stage} ===")
        print(f"系统: {m['used_gb']:.2f}/{m['total_gb']:.2f}GB ({m['percent']:.1f}%), 可用 {m['available_gb']:.2f}GB")
        if 'gpu_total_gb' in m:
            print(f"GPU: alloc {m['gpu_allocated_gb']:.2f}GB, cached {m['gpu_cached_gb']:.2f}GB, "
                  f"free {m['gpu_free_gb']:.2f}/{m['gpu_total_gb']:.2f}GB")

    def suggest_config_adjustments(self):
        m = self.get_memory_usage()
        tips = []
        if m['percent'] > 80:
            cur_bs = getattr(self.config, 'batch_size', 256)
            tips += [
                "系统内存紧张，建议：",
                f"- 减少 batch_size（当前 {cur_bs}）",
                "- 将 num_workers 调低为 0~1",
                "- 关闭 pin_memory",
                f"- 建议 batch_size ≈ {max(16, cur_bs // 2)}"
            ]
        if 'gpu_free_gb' in m and m['gpu_free_gb'] < 2.0:
            tips += [
                "GPU 内存紧张，建议：",
                "- 进一步减少 batch_size",
                "- 降低 embedding_dim",
                "- 使用 AMP 或梯度累积"
            ]
        return tips or ["内存使用正常"]

    def auto_adjust_config(self):
        m = self.get_memory_usage()
        # 系统内存
        if m['percent'] > 80 and hasattr(self.config, 'batch_size'):
            orig = self.config.batch_size
            self.config.optimized_batch_size = max(16, orig // 2)
            self.config.num_workers = min(getattr(self.config, 'num_workers', 0), 1)
            print(f"自动调整: batch_size {orig} -> {self.config.optimized_batch_size}, num_workers -> {self.config.num_workers}")
        # GPU 内存
        if 'gpu_free_gb' in m and m['gpu_free_gb'] < 2.0:
            cur = getattr(self.config, 'optimized_batch_size', getattr(self.config, 'batch_size', 256))
            self.config.optimized_batch_size = max(8, cur // 2)
            print(f"GPU 紧张: 进一步减少 batch_size -> {self.config.optimized_batch_size}")

def setup_memory_efficient_environment():
    """设置简洁的内存友好环境变量"""
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    if torch.cuda.is_available():
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        torch.cuda.empty_cache()
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    print("已应用内存优化环境变量")

def monitor_training_memory(model, train_loader, device, max_batches=5):
    """快速批次内显存观测，不影响训练逻辑"""
    print("\n=== 训练内存监控 ===")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= max_batches:
                break
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                before = torch.cuda.memory_allocated() / 1024**3
            seq = batch['sequence'].to(device, non_blocking=True)
            tgt = batch['target_id'].to(device, non_blocking=True)
            _ = model(seq)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                after = torch.cuda.memory_allocated() / 1024**3
                print(f"Batch {i+1}: ΔGPU {after - before:.3f}GB")
            del seq, tgt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print("内存监控完成")