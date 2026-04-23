#!/bin/bash
# VyomRaksha — Phase 1 Training: Card 1
# GPU 1: threat (Qwen2.5-14B, 300 steps) → power (7B, 200 steps) → fuel (7B, 200 steps)
#
# Submit from project root:
#   sbatch training/cluster_jobs/phase1_card1.sh
#
#SBATCH --job-name=vy_phase1_c1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=divit.gupta23@spit.ac.in

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${PROJECT_DIR}/training/checkpoints"

mkdir -p "${CKPT_DIR}" "${PROJECT_DIR}/logs"

# Activate venv: try project venv first, then home venv, then conda
if [ -f "${PROJECT_DIR}/.venv/bin/activate" ]; then
    source "${PROJECT_DIR}/.venv/bin/activate"
elif [ -f "${HOME}/vyom_env/bin/activate" ]; then
    source "${HOME}/vyom_env/bin/activate"
elif command -v conda &>/dev/null; then
    source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
    conda activate pytorch 2>/dev/null || true
else
    echo "WARNING: No venv found — using system Python"
fi

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_DISABLED=true

# Push to HF Hub only if token is available
PUSH_FLAG=""
if [ -n "${HF_TOKEN:-}" ]; then
    PUSH_FLAG="--push_to_hub"
    echo "HF_TOKEN found — checkpoints will be pushed to Hub"
else
    echo "HF_TOKEN not set — checkpoints saved locally only"
fi

cd "${PROJECT_DIR}"

echo "=== VyomRaksha Phase 1 — Card 1 — Job ${SLURM_JOB_ID:-local_$$} ==="
echo "Node: $(hostname)  Started: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Checkpoints: ${CKPT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi unavailable"

# ── Step 1: Threat Sub-Agent (Qwen2.5-14B, 300 steps) ───────────────────────
# Threat is the most compute-intensive agent: 14B model, deepest CoT reasoning.
# 300 steps to ensure the 6-step CoT pipeline and cascade alert generation are
# reliably learned before SarvaDrishti depends on threat confidence scores.
echo ""
echo "$(date) --- [1/3] threat (Qwen2.5-14B, 300 steps) ---"
python training/train_sub_agent.py \
    --agent     threat \
    --model_size 14b \
    --steps     300 \
    --batch_size 2 \
    --output_dir "${CKPT_DIR}" \
    ${PUSH_FLAG}
echo "$(date) threat DONE — checkpoint: ${CKPT_DIR}/threat/"

# ── Step 2: Power Sub-Agent (Qwen2.5-7B, 200 steps) ─────────────────────────
echo ""
echo "$(date) --- [2/3] power (Qwen2.5-7B, 200 steps) ---"
python training/train_sub_agent.py \
    --agent     power \
    --model_size 7b \
    --steps     200 \
    --batch_size 4 \
    --output_dir "${CKPT_DIR}" \
    ${PUSH_FLAG}
echo "$(date) power DONE — checkpoint: ${CKPT_DIR}/power/"

# ── Step 3: Fuel Sub-Agent (Qwen2.5-7B, 200 steps) ──────────────────────────
echo ""
echo "$(date) --- [3/3] fuel (Qwen2.5-7B, 200 steps) ---"
python training/train_sub_agent.py \
    --agent     fuel \
    --model_size 7b \
    --steps     200 \
    --batch_size 4 \
    --output_dir "${CKPT_DIR}" \
    ${PUSH_FLAG}
echo "$(date) fuel DONE — checkpoint: ${CKPT_DIR}/fuel/"

echo ""
echo "=== Card 1 Phase 1 COMPLETE: $(date) ==="
echo "Agents trained: threat, power, fuel"
echo "Checkpoints at: ${CKPT_DIR}/{threat,power,fuel}/"
echo "Job ${SLURM_JOB_ID:-local_$$} complete on $(date)"
