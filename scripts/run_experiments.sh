#!/usr/bin/env bash
set -euo pipefail

# ==============================
# KG-Corrector 并行实验启动（tmux 稳健版）
# 版本说明（去掉 GPU 相关逻辑）
# ==============================

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ===== 可配置区域 =====
CONDA_ENV="recbole_env"      # 为空则不激活 conda
WL=5                      # <<< 手动设置：window_size 数值（示例为 128）
SESSION="mine${WL}"          # tmux 会话名，带上 WL 便于区分

# 固定顺序的模型与配置（按下标对齐）
MODELS=(SASRec GRU4Rec KSR KATRec BERT4Rec)
CONFIGS=(
  "configs/model/sasrec_ml1m.yaml"
  "configs/model/gru4rec_ml1m.yaml"
  "configs/model/ksr_ml1m.yaml"
  "configs/model/katrec_ml1m.yaml"
  "configs/model/bert4rec_ml1m.yaml"
)
# =====================

# 1) 长度一致性 + 配置文件存在性检查
if [[ "${#MODELS[@]}" -ne "${#CONFIGS[@]}" ]]; then
  echo "!! 配置错误：MODELS 数量(${#MODELS[@]}) 与 CONFIGS 数量(${#CONFIGS[@]}) 不一致。" >&2
  exit 1
fi
for i in "${!MODELS[@]}"; do
  [[ -n "${CONFIGS[$i]}" && -f "${CONFIGS[$i]}" ]] || {
    echo "!! Missing config for ${MODELS[$i]} -> ${CONFIGS[$i]}" >&2
    exit 1
  }
done

# 2) 解析 conda.sh 路径（如果安装了 conda）
CONDA_SH=""
if [[ -n "${CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
  if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  fi
fi

# 3) 必须有 tmux
if ! command -v tmux >/dev/null 2>&1; then
  echo "!! 未检测到 tmux，请安装或加入 PATH 后重试。" >&2
  exit 1
fi

# 4) 若会话已存在则清理（确保干净重启）
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SESSION}"
fi

# 5) 创建会话 & 第一个窗口
tmux new-session -d -s "${SESSION}" -n "${MODELS[0]}" -c "${PROJECT_ROOT}"

# 6) 自检会话存在
if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "!! tmux 会话创建失败（可能是 tmux socket 权限或环境问题）。" >&2
  exit 1
fi

# 7) 向某个窗口注入一套标准命令
send_job () {
  local win="$1"
  local cfg="$2"
  local wl="$3"
  tmux send-keys -t "${win}" "cd '${PROJECT_ROOT}'" C-m

  # Conda 激活（若需要）
  if [[ -n "${CONDA_ENV}" ]]; then
    if [[ -n "${CONDA_SH}" ]]; then
      tmux send-keys -t "${win}" "source '${CONDA_SH}'" C-m
      tmux send-keys -t "${win}" "conda activate '${CONDA_ENV}' || true" C-m
    else
      tmux send-keys -t "${win}" "[[ -f ~/.bashrc ]] && source ~/.bashrc" C-m
      tmux send-keys -t "${win}" "command -v conda >/dev/null 2>&1 && conda activate '${CONDA_ENV}' || true" C-m
    fi
  fi

  # 打印信息 & 执行训练；结束后保持交互
  tmux send-keys -t "${win}" 'echo "[info] PWD=$(pwd)"' C-m
  tmux send-keys -t "${win}" "echo \"[run] python main.py --config ${cfg} --window_size ${wl}\"" C-m
  tmux send-keys -t "${win}" "python main.py --config '${cfg}' --window_size ${wl}; ec=\$?; echo \"[exit] code=\$ec\"" C-m
  tmux send-keys -t "${win}" "exec bash" C-m
}

# 8) 注入第一个窗口
send_job "${SESSION}:0" "${CONFIGS[0]}" "${WL}"

# 9) 其余窗口
for i in "${!MODELS[@]}"; do
  [[ "$i" -eq 0 ]] && continue
  tmux new-window -t "${SESSION}" -n "${MODELS[$i]}" -c "${PROJECT_ROOT}"
  send_job "${SESSION}:$i" "${CONFIGS[$i]}" "${WL}"
done

# 10) 输出提示
echo "tmux 会话 '${SESSION}' 已启动（WL=${WL}），共 ${#MODELS[@]} 个窗口：${MODELS[*]}"
echo "进入会话：tmux attach -t ${SESSION}"