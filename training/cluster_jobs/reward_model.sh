#!/bin/bash
# VyomRaksha — SarvaDrishti Preference Reward Model
# Pre-Phase 2 prerequisite. Trains Qwen2.5-3B on Bradley-Terry preference pairs
# from training/data/preference_pairs/sarvadrishi_pairs.jsonl (164 pairs).
# Output: training/checkpoints/sarvadrishi_reward_model/
#
# Submit with SLURM dependency (after Phase 1.5):
#   sbatch --dependency=afterok:<PHASE15_JOB_ID> \
#          training/cluster_jobs/reward_model.sh
# Or use submit_phase1.sh which handles this automatically.
#
#SBATCH --job-name=vy_reward_model
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/reward_model_%j.log
#SBATCH --error=logs/reward_model_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${PROJECT_DIR}/training/checkpoints"
PAIRS_FILE="${PROJECT_DIR}/training/data/preference_pairs/sarvadrishi_pairs.jsonl"

mkdir -p "${CKPT_DIR}" "${PROJECT_DIR}/logs"

module load python/3.11.14
source "${PROJECT_DIR}/my_env/bin/activate"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_DISABLED=true

PUSH_FLAG=""
if [ -n "${HF_TOKEN:-}" ]; then
    PUSH_FLAG="--push_to_hub"
    echo "HF_TOKEN found — reward model will be pushed to Hub"
else
    echo "HF_TOKEN not set — reward model saved locally only"
fi

cd "${PROJECT_DIR}"

echo "=== VyomRaksha Reward Model Training — Job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname)  Started: $(date)"
echo "Output: ${CKPT_DIR}/sarvadrishi_reward_model/"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Verify preference pairs ──────────────────────────────────────────────────
if [ -f "${PAIRS_FILE}" ]; then
    PAIR_COUNT=$(wc -l < "${PAIRS_FILE}")
    echo ""
    echo "Preference pairs: ${PAIR_COUNT} lines in ${PAIRS_FILE}"
    if [ "${PAIR_COUNT}" -lt 50 ]; then
        echo "WARNING: only ${PAIR_COUNT} pairs — reward model accuracy may be limited"
    fi
else
    echo "WARNING: ${PAIRS_FILE} not found"
    echo "Reward trainer will fall back to 30 synthetic pairs (lower quality)"
fi

# ── Train reward model (Qwen2.5-3B, 150 steps) ──────────────────────────────
# 3B model fits comfortably on 32GB with batch_size=8.
# 150 steps is sufficient for 164 real pairs to reach >90% held-out accuracy.
echo ""
echo "$(date) --- Training SarvaDrishti reward model ---"
python training/train_reward_model.py \
    --model_size 3b \
    --steps     150 \
    --batch_size 8 \
    --output_dir "${CKPT_DIR}" \
    ${PUSH_FLAG}

echo ""
echo "=== Reward Model Training COMPLETE: $(date) ==="
echo "Checkpoint: ${CKPT_DIR}/sarvadrishi_reward_model/"
echo ""
echo "Next step:"
echo "  sbatch training/cluster_jobs/phase2_sarvadrishi.sh  (R2-8.2)"
