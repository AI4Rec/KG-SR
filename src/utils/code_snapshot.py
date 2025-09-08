# src/utils/code_snapshot.py
from __future__ import annotations
import os, sys, subprocess
from pathlib import Path
from datetime import datetime
from typing import Iterable

DEFAULT_INCLUDE_EXTS = {".py", ".yaml", ".yml", ".sh", ".txt", ".md", ".cfg", ".ini"}
DEFAULT_PARTIAL_TEXT_EXTS = {".csv", ".tsv", ".json", ".log", ".inter", ".kg", ".link"}
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache",
    ".mypy_cache", "dataset", "saved", "results", "wandb", "logs", ".DS_Store"
}
DEFAULT_BINARY_EXTS = {".pth", ".tar", ".pkl", ".npy", ".npz", ".pyc", ".so", ".dll", ".exe", ".bin", ".dat"}

def _is_binary_path(p: Path) -> bool:
    suf = "".join(p.suffixes).lower()
    if suf.endswith(".pth.tar") or suf.endswith(".pkl.gz"):
        return True
    return p.suffix.lower() in DEFAULT_BINARY_EXTS

def _try_read_bytes(p: Path):
    try: return p.read_bytes()
    except Exception: return None

def _decode_text(b: bytes) -> str:
    for enc in ("utf-8","utf-16","gbk","gb2312","latin-1"):
        try: return b.decode(enc)
        except Exception: pass
    return b.decode("utf-8", errors="replace")

def _iter_project_files(root: Path, include_exts, partial_text_exts, exclude_dirs) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted([d for d in dirnames if d not in exclude_dirs and not d.startswith(".") or d=="configs"], key=str.lower)
        for fname in sorted(filenames, key=str.lower):
            p = Path(dirpath)/fname
            if p.name=="main.py" or ("configs" in p.parts):
                yield p; continue
            suf = p.suffix.lower()
            if suf in include_exts or suf in partial_text_exts: yield p

def _build_tree(root: Path, exclude_dirs) -> str:
    lines = ["文件夹结构:", f"└── {root.name or '/'}"]
    def kids(path: Path):
        try:
            d=[c for c in path.iterdir() if c.is_dir() and c.name not in exclude_dirs and not c.name.startswith(".")]
            f=[c for c in path.iterdir() if c.is_file()]
            return sorted(d,key=lambda x:x.name.lower()), sorted(f,key=lambda x:x.name.lower())
        except PermissionError:
            return [],[]
    def walk(path: Path, prefix: str):
        d,f = kids(path); all_=d+f
        for i,ch in enumerate(all_):
            last = i==len(all_)-1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{ch.name}")
            if ch.is_dir():
                walk(ch, prefix + ("    " if last else "│   "))
    walk(root,"    ")
    return "\n".join(lines)

def _gather_env_info(project_root: Path) -> str:
    info=[ "环境信息:",
           f"- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
           f"- Python: {sys.version.split()[0]} on {sys.platform}" ]
    try:
        out = subprocess.check_output([sys.executable,"-m","pip","freeze"], stderr=subprocess.STDOUT, timeout=10)
        info += ["- pip freeze (partial):", _decode_text(out)[:5000].strip()]
    except Exception: info.append("- pip freeze: N/A")
    try:
        commit = subprocess.check_output(["git","rev-parse","HEAD"], cwd=project_root, stderr=subprocess.STDOUT, timeout=3)
        status = subprocess.check_output(["git","status","--porcelain"], cwd=project_root, stderr=subprocess.STDOUT, timeout=3)
        info.append(f"- git commit: {_decode_text(commit).strip()}")
        info.append(f"- git status: {'clean' if not _decode_text(status).strip() else 'dirty'}")
    except Exception: info.append("- git: N/A")
    try:
        out = subprocess.check_output(["nvidia-smi","-L"], stderr=subprocess.STDOUT, timeout=2)
        info.append("- GPU: " + _decode_text(out).strip())
    except Exception: info.append("- GPU: N/A")
    return "\n".join(info)

def write_code_snapshot(
    save_dir: Path,
    project_root: Path|None=None,
    include_exts=None,
    partial_text_exts=None,
    exclude_dirs=None,
    max_bytes_per_file: int = 1_000_000,
    partial_head_lines: int = 50,
) -> Path:
    save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir/"code.txt"
    project_root = Path(project_root) if project_root else Path.cwd()
    include_exts = include_exts or DEFAULT_INCLUDE_EXTS
    partial_text_exts = partial_text_exts or DEFAULT_PARTIAL_TEXT_EXTS
    exclude_dirs = exclude_dirs or DEFAULT_EXCLUDE_DIRS

    files = sorted(_iter_project_files(project_root, include_exts, partial_text_exts, exclude_dirs),
                   key=lambda p: "/".join(p.relative_to(project_root).parts).lower())
    tree_text = _build_tree(project_root, exclude_dirs)
    env_text = _gather_env_info(project_root)

    with out_path.open("w", encoding="utf-8", errors="replace") as f:
        f.write(f"分析文件夹: {project_root.resolve()}\n")
        f.write(f"输出文件: {out_path.name}\n")
        f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        f.write(env_text + "\n")
        f.write("\n" + "="*60 + "\n")
        f.write(tree_text + "\n")
        f.write("\n" + "="*60 + "\n")
        f.write("文件内容分析:\n")
        f.write("="*60 + "\n")

        for fp in files:
            if _is_binary_path(fp): continue
            rel = fp.relative_to(project_root)
            f.write("\n" + "="*60 + "\n")
            f.write(f"文件位置: {rel.as_posix()}\n")
            f.write("="*60 + "\n")

            raw = _try_read_bytes(fp)
            if raw is None:
                f.write("无法读取文件内容\n"); continue

            truncated = False
            if len(raw) > max_bytes_per_file:
                raw = raw[:max_bytes_per_file]; truncated = True

            text = _decode_text(raw)
            suf = fp.suffix.lower()
            if suf in include_exts:
                if truncated: f.write("[TRUNCATED]\n")
                f.write(text + "\n")
            else:
                lines = text.splitlines()
                for i, ln in enumerate(lines[:partial_head_lines], 1):
                    f.write(f"{i:3d}: {ln}\n")
                if len(lines) > partial_head_lines:
                    f.write(f"... (文件共 {len(lines)} 行，仅显示前{partial_head_lines}行)\n")

        f.write("\n" + "="*60 + "\n")
        f.write("分析完成!\n")
        f.write(f"结果已保存到: {out_path.name}\n")
    return out_path