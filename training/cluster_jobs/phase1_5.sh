#!/bin/bash
# VyomRaksha — Phase 1.5: Brief Joint Exposure
# Runs AFTER both phase1_card1 and phase1_card2 complete.
# Each sub-agent trains 200 additional GRPO steps (--skip_sft) against varied
# multi-agent-flavored prompts. TRL GRPOTrainer resumes from the Phase 1
# checkpoint in training/checkpoints/{agent}/ if one is present.
# Purpose: robustness before SarvaDrishti trains on the frozen sub-agent ensemble.
#
# Submit with SLURM dependency:
#   sbatch --dependency=afterok:<CARD1_JOB_ID>:<CARD2_JOB_ID> \
#          training/cluster_jobs/phase1_5.sh
# Or use submit_phase1.sh which handles this automatically.
#
#SBATCH --job-name=vy_phase1_5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=divit.gupta23@spit.ac.in

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
CKPT_DIR="${PROJECT_DIR}/training/checkpoints"

mkdir -p "${CKPT_DIR}" "${PROJECT_DIR}/logs"

module load python/3.11.14 || { echo "FATAL: module load failed"; exit 1; }
source ~/vyom_env/bin/activate

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_DISABLED=true

PUSH_FLAG=""
if [ -n "${HF_TOKEN:-}" ]; then
    PUSH_FLAG="--push_to_hub"
fi

cd ~/Meta-Hack-VyomRaksha

echo "=== VyomRaksha Phase 1.5 — Joint Exposure — Job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname)  Started: $(date)"
echo "Checkpoints: ${CKPT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi unavailable"

# ── Verify Phase 1 checkpoints exist before starting ────────────────────────
ALL_AGENTS=("threat" "power" "fuel" "thermal" "computational" "structural" "communications" "probe_systems")
MISSING=()
for AGENT in "${ALL_AGENTS[@]}"; do
    if [ ! -d "${CKPT_DIR}/${AGENT}" ]; then
        MISSING+=("${AGENT}")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "WARNING: Phase 1 checkpoints missing for: ${MISSING[*]}"
    echo "Phase 1.5 will train from base model for these agents."
fi

# ── Step 1: Threat Sub-Agent (14B, 200 additional steps, skip SFT) ──────────
echo ""
echo "$(date) --- [1/8] threat — Phase 1.5 (14B, 200 steps, skip_sft) ---"
python training/train_sub_agent.py \
    --agent     threat \
    --model_size 14b \
    --steps     200 \
    --batch_size 2 \
    --skip_sft \
    --output_dir "${CKPT_DIR}" \
    ${PUSH_FLAG}
echo "$(date) threat phase 1.5 DONE"

# ── Steps 2–8: Remaining 7B agents ──────────────────────────────────────────
AGENTS_7B=("power" "fuel" "thermal" "computational" "structural" "communications" "probe_systems")
STEP=2
for AGENT in "${AGENTS_7B[@]}"; do
    echo ""
    echo "$(date) --- [${STEP}/8] ${AGENT} — Phase 1.5 (7B, 200 steps, skip_sft) ---"

    python training/train_sub_agent.py \
        --agent     "${AGENT}" \
        --model_size 7b \
        --steps     200 \
        --batch_size 4 \
        --skip_sft \
        --output_dir "${CKPT_DIR}" \
        ${PUSH_FLAG}

    echo "$(date) ${AGENT} phase 1.5 DONE"
    STEP=$((STEP + 1))
done

echo ""
echo "=== Phase 1.5 COMPLETE: $(date) ==="
echo "All 8 sub-agents have completed Phase 1 + Phase 1.5 exposure."
echo "Ready for: reward_model.sh → phase2_sarvadrishi.sh"
echo "Job $SLURM_JOB_ID complete on $(date)"
