#!/bin/bash
# VyomRaksha — Phase 3: Emergency Authority Calibration
# GPU 1. SarvaDrishti is frozen; selected sub-agents partially unfrozen.
# Calibrates learned emergency authority triggers.
# Run after phase2_sarvadrishi.sh completes.
#
# Submit with SLURM dependency (after Phase 2):
#   sbatch --dependency=afterok:<PHASE2_JOB_ID> \
#          training/cluster_jobs/phase3_emergency.sh
# Or use submit_phase2.sh which handles this automatically.
#
#SBATCH --job-name=vy_phase3_emrg
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=divit.gupta23@spit.ac.in

set -euo pipefail

module load python/3.11.14 || { echo "FATAL: module load failed"; exit 1; }
source ~/vyom_env/bin/activate
cd ~/Meta-Hack-VyomRaksha

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(pwd)"
CKPT_DIR="${PROJECT_DIR}/training/checkpoints"

mkdir -p "${CKPT_DIR}" "${PROJECT_DIR}/logs"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_DISABLED=true

echo "=== VyomRaksha Phase 3 — Emergency Authority — Job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname)  Started: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Checkpoints: ${CKPT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi unavailable"

# ── Verify SarvaDrishti checkpoint exists ────────────────────────────────────
if [ ! -d "${CKPT_DIR}/sarvadrishi" ]; then
    echo "ERROR: SarvaDrishti checkpoint not found at ${CKPT_DIR}/sarvadrishi"
    echo "Run phase2_sarvadrishi.sh first."
    exit 1
fi

echo ""
echo "$(date) --- Emergency authority calibration (500 steps) ---"
python training/train_emergency.py \
    --steps 500 \
    --sarvadrishi_checkpoint training/checkpoints/sarvadrishi/ \
    --sub_agent_checkpoints training/checkpoints/

echo ""
echo "=== Phase 3 Emergency Calibration COMPLETE: $(date) ==="
echo "Emergency-authority sub-agents updated in: ${CKPT_DIR}/"
echo "Pipeline complete — all phases done."
echo "Job $SLURM_JOB_ID complete on $(date)"
