#!/bin/bash
# VyomRaksha — AWS EC2 Instance Setup
# Run ONCE after SSH into a fresh EC2 instance.
# Recommended AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)
# Recommended instance: g5.2xlarge (1x A10G 24GB, 8 vCPU, 32GB RAM)
#
# Usage:
#   export HF_TOKEN=hf_your_token_here
#   export GITHUB_TOKEN=ghp_your_token_here   # optional, for code pushes
#   bash training/aws/setup_instance.sh [ROLE]
#
# ROLE controls which model weights are pre-downloaded:
#   card1       — Qwen2.5-14B + Qwen2.5-7B  (Phase 1 Card 1: threat, power, fuel)
#   card2       — Qwen2.5-7B only            (Phase 1 Card 2: remaining 5 agents)
#   reward      — Qwen2.5-3B only            (Reward model training)
#   sarvadrishi — Qwen2.5-14B only           (Phase 2 SarvaDrishti)
#   emergency   — Qwen2.5-7B + Qwen2.5-14B  (Phase 3 emergency calibration)
#   all         — All models                 (if storage is not a concern)
#
# Default role: card1
#
# After this script finishes successfully, run:
#   bash training/aws/run_training.sh <ROLE>

set -euo pipefail

ROLE="${1:-card1}"
echo "=========================================="
echo " VyomRaksha AWS Instance Setup"
echo " Role: ${ROLE}"
echo " Started: $(date)"
echo " Host: $(hostname)"
echo "=========================================="

# ── 1. GPU check ─────────────────────────────────────────────────────────────
echo ""
echo "[1/9] GPU info"
nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version \
    --format=csv,noheader || { echo "ERROR: nvidia-smi failed. Is this a GPU instance?"; exit 1; }

# ── 2. Activate conda environment ────────────────────────────────────────────
echo ""
echo "[2/9] Activating conda environment"
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch
    echo "Activated: pytorch ($(python --version))"
else
    echo "WARNING: /opt/conda not found. Using system Python: $(python3 --version)"
    alias python=python3 2>/dev/null || true
fi

# ── 3. Install training dependencies ─────────────────────────────────────────
echo ""
echo "[3/9] Installing training dependencies (pinned versions)"
pip install --quiet --upgrade pip

# Core ML stack — pinned for reproducibility
pip install --quiet \
    "trl==0.16.0" \
    "transformers>=4.48.0,<5.0.0" \
    "accelerate>=1.0.0" \
    "peft>=0.14.0" \
    "bitsandbytes>=0.45.0" \
    "datasets>=3.0.0" \
    "huggingface_hub>=0.27.0"

# Unsloth — install after torch is confirmed present
pip install --quiet "unsloth>=2025.3" || {
    echo "WARNING: Unsloth install failed. Will use HF BitsAndBytes fallback."
    echo "Training will still work, just slightly slower."
}

# Project dependencies
pip install --quiet "openenv-core[core]>=0.2.3" "pydantic>=2.0" "python-dotenv>=1.0.0"

echo "Dependency install complete."
pip show trl transformers unsloth peft bitsandbytes | grep -E "^(Name|Version):"

# ── 4. Clone or update repository ────────────────────────────────────────────
echo ""
echo "[4/9] Repository setup"
REPO_DIR="${HOME}/Meta-Hack-VyomRaksha"

if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repo already exists at ${REPO_DIR} — pulling latest"
    git -C "${REPO_DIR}" pull origin main
else
    echo "Cloning repository to ${REPO_DIR}"
    if [ -n "${GITHUB_TOKEN:-}" ]; then
        git clone "https://Xx-D3V-xX:${GITHUB_TOKEN}@github.com/Xx-D3V-xX/Meta-Hack-VyomRaksha.git" "${REPO_DIR}"
    else
        git clone "https://github.com/Xx-D3V-xX/Meta-Hack-VyomRaksha.git" "${REPO_DIR}"
    fi
fi

cd "${REPO_DIR}"
echo "Repo: $(pwd)"
echo "Latest commit: $(git log --oneline -1)"

pip install --quiet -e ".[dev]"
echo "Project installed in editable mode."

# ── 5. HuggingFace credentials ───────────────────────────────────────────────
echo ""
echo "[5/9] HuggingFace credentials"
if [ -n "${HF_TOKEN:-}" ]; then
    python -c "from huggingface_hub import login; login(token='${HF_TOKEN}', add_to_git_credential=False)"
    echo "HF login successful."
else
    echo "WARNING: HF_TOKEN not set."
    echo "Set it before running this script: export HF_TOKEN=hf_..."
    echo "Model downloads from HuggingFace Hub may fail for gated models."
fi

export HF_HOME="${REPO_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"
echo "HF_HOME: ${HF_HOME}"

# ── 6. Pre-download model weights ────────────────────────────────────────────
echo ""
echo "[6/9] Pre-downloading model weights for role: ${ROLE}"
echo "This may take 20–60 minutes depending on network speed."

download_model() {
    local MODEL_ID="$1"
    echo "  Downloading ${MODEL_ID}..."
    python - <<PYEOF
import os, sys
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(
        repo_id="${MODEL_ID}",
        cache_dir=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        ignore_patterns=["*.msgpack", "*.h5", "flax_*", "tf_*"],
    )
    import subprocess
    size = subprocess.check_output(["du", "-sh", path]).decode().split()[0]
    print(f"  Downloaded to: {path} ({size})")
except Exception as e:
    print(f"  ERROR downloading ${MODEL_ID}: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

case "${ROLE}" in
    card1)
        download_model "Qwen/Qwen2.5-14B-Instruct"
        download_model "Qwen/Qwen2.5-7B-Instruct"
        ;;
    card2)
        download_model "Qwen/Qwen2.5-7B-Instruct"
        ;;
    reward)
        download_model "Qwen/Qwen2.5-3B-Instruct"
        ;;
    sarvadrishi)
        download_model "Qwen/Qwen2.5-14B-Instruct"
        ;;
    emergency)
        download_model "Qwen/Qwen2.5-7B-Instruct"
        download_model "Qwen/Qwen2.5-14B-Instruct"
        ;;
    all)
        download_model "Qwen/Qwen2.5-14B-Instruct"
        download_model "Qwen/Qwen2.5-7B-Instruct"
        download_model "Qwen/Qwen2.5-3B-Instruct"
        ;;
    *)
        echo "WARNING: Unknown role '${ROLE}'. Downloading Qwen2.5-7B only."
        download_model "Qwen/Qwen2.5-7B-Instruct"
        ;;
esac
echo "Model downloads complete."

# ── 7. GitHub credentials (for code pushes — NOT weights) ────────────────────
echo ""
echo "[7/9] GitHub credentials"
if [ -n "${GITHUB_TOKEN:-}" ]; then
    git config --global credential.helper store
    echo "https://Xx-D3V-xX:${GITHUB_TOKEN}@github.com" > "${HOME}/.git-credentials"
    chmod 600 "${HOME}/.git-credentials"
    echo "GitHub credentials configured."
else
    echo "GITHUB_TOKEN not set — code pushes after training will be skipped."
fi

# ── 8. Smoke tests ───────────────────────────────────────────────────────────
echo ""
echo "[8/9] Running smoke tests (no GPU needed — skip_model_load mode)"
cd "${REPO_DIR}"

echo "  [smoke 1/6] train_sub_agent power..."
python training/train_sub_agent.py --agent power --steps 3 --batch_size 1 --skip_model_load
echo "  PASS"

echo "  [smoke 2/6] train_sub_agent threat..."
python training/train_sub_agent.py --agent threat --steps 3 --batch_size 1 --skip_model_load
echo "  PASS"

echo "  [smoke 3/6] train_reward_model..."
python training/train_reward_model.py --steps 3 --batch_size 1 --model_size tiny
echo "  PASS"

echo "  [smoke 4/6] train_sarvadrishi..."
python training/train_sarvadrishi.py --steps 3 --batch_size 1 --model_size tiny
echo "  PASS"

echo "  [smoke 5/6] train_emergency..."
python training/train_emergency.py --steps 3
echo "  PASS"

echo "  [smoke 6/6] eval_pipeline..."
python training/eval_pipeline.py --n_eval_episodes 1 --output_dir /tmp/vyom_eval_test
echo "  PASS"

echo "All smoke tests passed."

# ── 9. Final summary ─────────────────────────────────────────────────────────
echo ""
echo "[9/9] Environment summary"
echo "  Conda env:        $(conda info --envs 2>/dev/null | grep '*' | awk '{print $1}' || echo 'N/A')"
echo "  Python:           $(python --version)"
echo "  PyTorch:          $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "  CUDA available:   $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'unknown')"
echo "  CUDA version:     $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'unknown')"
echo "  trl:              $(python -c 'import trl; print(trl.__version__)' 2>/dev/null || echo 'not found')"
echo "  transformers:     $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'not found')"
echo "  unsloth:          $(python -c 'import unsloth; print(unsloth.__version__)' 2>/dev/null || echo 'not installed (will use HF fallback)')"
echo "  peft:             $(python -c 'import peft; print(peft.__version__)' 2>/dev/null || echo 'not found')"
echo "  bitsandbytes:     $(python -c 'import bitsandbytes; print(bitsandbytes.__version__)' 2>/dev/null || echo 'not found')"
echo "  HF_HOME:          ${HF_HOME}"
echo "  Repo:             ${REPO_DIR}"
echo "  Commit:           $(git -C ${REPO_DIR} log --oneline -1)"
echo ""
echo "=========================================="
echo " Setup complete: $(date)"
echo " Next step: bash training/aws/run_training.sh ${ROLE}"
echo "=========================================="
