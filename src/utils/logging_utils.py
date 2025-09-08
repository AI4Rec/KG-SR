# src/utils/logging_utils.py

import logging
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO", file_path: Optional[Path] = None, name: str = "train"):
    """
    设置日志记录器，兼容 Python 3.9 及以下。
    - **修复**：避免重复添加 handlers 导致日志重复。
    """
    logger = logging.getLogger(name)

    # 清理旧的 handlers，防止重复输出
    if logger.handlers:
        for h in list(logger.handlers):
            logger.removeHandler(h)

    logger.propagate = False
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出（可选）
    if file_path is not None:
        fh = logging.FileHandler(file_path)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger