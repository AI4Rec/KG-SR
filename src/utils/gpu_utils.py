# src/utils/gpu_utils.py
from __future__ import annotations
import os
import math
from typing import List, Dict, Optional, Tuple
import torch

__all__ = ["get_gpu_memory", "select_device", "describe_devices"]

def _try_init_nvml():
    """尝试初始化 NVML，成功则返回 pynvml 模块，否则返回 None。"""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        return pynvml
    except Exception:
        return None

def _shutdown_nvml(pynvml_mod) -> None:
    try:
        if pynvml_mod is not None:
            pynvml_mod.nvmlShutdown()
    except Exception:
        pass

def _parse_visible_device_indices() -> Optional[List[int]]:
    """
    解析 CUDA_VISIBLE_DEVICES，返回可见的物理 GPU 索引列表（整数）。
    若未设置或不可解析，返回 None。
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return None
    parts = [p.strip() for p in cvd.split(",") if p.strip() != ""]
    idxs: List[int] = []
    for p in parts:
        # 仅支持数字索引（不处理 UUID/MIG 的复杂场景）
        if p.isdigit():
            idxs.append(int(p))
        else:
            return None
    return idxs if idxs else None

def get_gpu_memory() -> List[Dict[str, float]]:
    """
    返回当前“可见” GPU 的资源/负载信息列表（按 PyTorch 可见索引编号）：
    [
      {
        'id': torch 可见索引（int）,
        'total': bytes,
        'used': bytes,
        'free': bytes,
        'util': used/total（显存占用比例，0~1，向后兼容）,
        'gpu_util_pct': Volatile GPU-Util（0~100，若 NVML 不可用则为 None）,
        'mem_ctrl_util_pct': 显存控制器利用率（0~100，若 NVML 不可用则为 None）
      }, ...
    ]
    若无 CUDA 或无可见 GPU，返回 []。
    """
    if not torch.cuda.is_available():
        return []

    infos: List[Dict[str, float]] = []
    pynvml = _try_init_nvml()
    try:
        visible_phys = _parse_visible_device_indices()  # 物理索引列表或 None
        if pynvml is not None:
            if visible_phys is None:
                # 未设置 CVD，则假定 NVML 顺序 == PyTorch 顺序
                phys_indices = list(range(pynvml.nvmlDeviceGetCount()))
                visible_map = list(range(len(phys_indices)))  # torch 可见索引
            else:
                phys_indices = visible_phys
                visible_map = list(range(len(phys_indices)))  # 0..N-1 映射为 torch 可见索引

            # 限制到 PyTorch 实际可见数量，避免 cuda:<id> 超界
            torch_visible_cnt = torch.cuda.device_count()
            phys_indices = phys_indices[:torch_visible_cnt]
            visible_map = visible_map[:torch_visible_cnt]

            for torch_idx, phys_idx in zip(visible_map, phys_indices):
                handle = pynvml.nvmlDeviceGetHandleByIndex(phys_idx)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = float(mem.total)
                used = float(mem.used)
                free = float(mem.free)
                util_ratio = (used / total) if total > 0 else 1.0

                # Volatile Utilization（过去一个采样窗口的平均）
                try:
                    ur = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util_pct = float(ur.gpu)           # SM/核心利用率
                    mem_ctrl_util_pct = float(ur.memory)   # 显存控制器
                except Exception:
                    gpu_util_pct = float("nan")
                    mem_ctrl_util_pct = float("nan")

                infos.append({
                    "id": torch_idx,
                    "total": total,
                    "used": used,
                    "free": free,
                    "util": util_ratio,
                    "gpu_util_pct": gpu_util_pct,
                    "mem_ctrl_util_pct": mem_ctrl_util_pct,
                })
        else:
            # 无 NVML：退化到“显存信息 + 显存占用比例”
            count = torch.cuda.device_count()
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                total = float(props.total_memory)
                try:
                    reserved = float(torch.cuda.memory_reserved(i))
                except Exception:
                    reserved = 0.0
                used = reserved
                free = max(0.0, total - reserved)
                util_ratio = (used / total) if total > 0 else 1.0
                infos.append({
                    "id": i,
                    "total": total,
                    "used": used,
                    "free": free,
                    "util": util_ratio,
                    "gpu_util_pct": float("nan"),
                    "mem_ctrl_util_pct": float("nan"),
                })
    finally:
        _shutdown_nvml(pynvml)
    return infos

def _gpu_util_for_sort(g: Dict[str, float]) -> float:
    """用于排序的 GPU 利用率值；NVML 不可用时退化为显存占用比例（*100）。"""
    v = g.get("gpu_util_pct", float("nan"))
    if isinstance(v, float) and math.isfinite(v):
        return v
    # 退化：用显存占用比例推测忙碌程度
    return float(g.get("util", 1.0)) * 100.0

def select_device(
    prefer: str = "auto",
    strategy: str = "min_gpu_util",  # 默认改为挑“最闲”的
    min_free_mem_gb: float = 1.0,
    allow_cpu: bool = True,
    explicit_id: Optional[int] = None,
    verbose: bool = True,
) -> torch.device:
    """
    选择最合适的计算设备：
    - prefer: 'auto' | 'cuda' | 'cpu'
    - strategy:
        'max_free'     -> 空闲显存最大的卡
        'min_util'     -> 显存占用比例最低（向后兼容）
        'min_gpu_util' -> Volatile GPU-Util 最低（NVML），失败则退化到 'min_util'
    - min_free_mem_gb: 至少需要的空闲显存（不足则放宽）
    - explicit_id: 显式指定 GPU id（torch 可见索引，优先级最高）
    - allow_cpu: 没有可用 GPU 时是否回退到 CPU
    返回 torch.device('cuda:{id}') 或 torch.device('cpu')
    """
    if prefer == "cpu" or not torch.cuda.is_available():
        if verbose:
            print("[GPU-Select] Using CPU (prefer=cpu or CUDA unavailable).")
        return torch.device("cpu")

    if explicit_id is not None:
        if verbose:
            print(f"[GPU-Select] Using explicit cuda:{explicit_id}.")
        return torch.device(f"cuda:{explicit_id}")

    gpus = get_gpu_memory()
    if not gpus:
        if allow_cpu:
            if verbose:
                print("[GPU-Select] No visible GPUs. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda:0")

    min_free_bytes = int(min_free_mem_gb * (1024 ** 3))
    candidates = [g for g in gpus if g["free"] >= min_free_bytes] or gpus

    strat = strategy.lower()
    if strat in ("max_free", "max_free_mem"):
        best = max(candidates, key=lambda g: g["free"])
        strat_name = "max_free"
    elif strat in ("min_gpu_util", "min_volatile", "least_busy"):
        best = min(candidates, key=_gpu_util_for_sort)
        strat_name = "min_gpu_util"
    else:
        best = min(candidates, key=lambda g: g.get("util", 1.0))
        strat_name = "min_util"

    device = torch.device(f"cuda:{int(best['id'])}")
    if verbose:
        tot = best["total"] / 1024**3
        fre = best["free"] / 1024**3
        usd = best["used"] / 1024**3
        gpuu = best.get("gpu_util_pct", float("nan"))
        if math.isfinite(gpuu):
            extra = f", gpu_util={gpuu:.0f}%"
        else:
            extra = ""
        print(f"[GPU-Select] Strategy={strat_name} -> pick cuda:{int(best['id'])} "
              f"(free={fre:.2f}GB, used={usd:.2f}GB, total={tot:.2f}GB{extra}).")
    return device

def describe_devices() -> str:
    """返回一段可打印字符串，描述当前可见 GPU 负载/显存状况。"""
    infos = get_gpu_memory()
    if not infos:
        return "No CUDA GPUs visible."
    lines = ["Visible GPUs:"]
    for g in infos:
        free_gb = g['free']/1024**3
        tot_gb = g['total']/1024**3
        used_gb = g['used']/1024**3
        util_pct = g.get("gpu_util_pct", float("nan"))
        memc_pct = g.get("mem_ctrl_util_pct", float("nan"))
        util_str = f"{util_pct:.0f}%" if math.isfinite(util_pct) else "N/A"
        memc_str = f"{memc_pct:.0f}%" if math.isfinite(memc_pct) else "N/A"
        lines.append(
            f"  cuda:{int(g['id'])}: free {free_gb:.2f}GB / total {tot_gb:.2f}GB "
            f"(used {used_gb:.2f}GB), GPU-Util {util_str}, MemCtrl {memc_str}"
        )
    return "\n".join(lines)