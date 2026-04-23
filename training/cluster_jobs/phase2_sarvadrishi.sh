#!/bin/bash
# VyomRaksha — Phase 2: SarvaDrishti Orchestrator Training
# Both GPUs (gpu_max). Sub-agents must be FROZEN before this runs.
# Trains SarvaDrishti (Qwen2.5-14B) on the frozen sub-agent ensemble via GRPO.
# Run after reward_model.sh completes.
#
# Submit with SLURM dependency (after reward model):
#   sbatch --dependency=afterok:<REWARD_MODEL_JOB_ID> \
#          training/cluster_jobs/phase2_sarvadrishi.sh
# Or use submit_phase2.sh which handles this automatically.
#
#SBATCH --job-name=vy_phase2_sarva
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=divit.gupta23@spit.ac.in

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

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

cd "${PROJECT_DIR}"
CKPT_DIR="${PROJECT_DIR}/training/checkpoints"

mkdir -p "${CKPT_DIR}" "${PROJECT_DIR}/logs"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_DISABLED=true

PUSH_FLAG=""
if [ -n "${HF_TOKEN:-}" ]; then
    PUSH_FLAG="--push_to_hub"
    echo "HF_TOKEN found — checkpoint will be pushed to Hub"
else
    echo "HF_TOKEN not set — checkpoint saved locally only"
fi

echo "=== VyomRaksha Phase 2 — SarvaDrishti — Job ${SLURM_JOB_ID:-local_$$} ==="
echo "Node: $(hostname)  Started: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Checkpoints: ${CKPT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi unavailable"

# ── Verify sub-agent checkpoints exist ──────────────────────────────────────
ALL_AGENTS=("threat" "power" "fuel" "thermal" "computational" "structural" "communications" "probe_systems")
MISSING=()
for AGENT in "${ALL_AGENTS[@]}"; do
    if [ ! -d "${CKPT_DIR}/${AGENT}" ]; then
        MISSING+=("${AGENT}")
    fi
done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "ERROR: Missing sub-agent checkpoints: ${MISSING[*]}"
    echo "Run phase1_card1.sh + phase1_card2.sh + phase1_5.sh first."
    exit 1
fi

# ── Verify reward model checkpoint exists ────────────────────────────────────
if [ ! -d "${CKPT_DIR}/sarvadrishi_reward_model" ]; then
    echo "ERROR: Reward model checkpoint not found at ${CKPT_DIR}/sarvadrishi_reward_model"
    echo "Run reward_model.sh first."
    exit 1
fi

echo ""
echo "$(date) --- Training SarvaDrishti (Qwen2.5-14B, 1500 steps, batch_size=4) ---"
python training/train_sarvadrishi.py \
    --steps 1500 \
    --batch_size 4 \
    --sub_agent_checkpoints training/checkpoints/ \
    --reward_model_path training/checkpoints/sarvadrishi_reward_model/ \
    ${PUSH_FLAG}

echo ""
echo "=== Phase 2 SarvaDrishti COMPLETE: $(date) ==="
echo "Checkpoint: ${CKPT_DIR}/sarvadrishi/"
echo "Next step: submit_phase2.sh handles phase3_emergency.sh automatically"
echo "Job ${SLURM_JOB_ID:-local_$$} complete on $(date)"
