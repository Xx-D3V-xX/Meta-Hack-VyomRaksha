#!/bin/bash
# VyomRaksha — Phase 1 Training: Card 2
# GPU 2: thermal → computational → structural → communications → probe_systems
# All Qwen2.5-7B, 200 steps each. Runs in parallel with phase1_card1.sh.
#
# Submit from project root:
#   sbatch training/cluster_jobs/phase1_card2.sh
#
#SBATCH --job-name=vy_phase1_c2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/phase1_card2_%j.log
#SBATCH --error=logs/phase1_card2_%j.err

set -euo pipefail

# ── Environment ──────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT_DIR="${PROJECT_DIR}/training/checkpoints"

mkdir -p "${CKPT_DIR}" "${PROJECT_DIR}/logs"

module load python/3.11.14
source "${PROJECT_DIR}/my_env/bin/activate"

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_DISABLED=true

PUSH_FLAG=""
if [ -n "${HF_TOKEN:-}" ]; then
    PUSH_FLAG="--push_to_hub"
    echo "HF_TOKEN found — checkpoints will be pushed to Hub"
else
    echo "HF_TOKEN not set — checkpoints saved locally only"
fi

cd "${PROJECT_DIR}"

echo "=== VyomRaksha Phase 1 — Card 2 — Job ${SLURM_JOB_ID} ==="
echo "Node: $(hostname)  Started: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Checkpoints: ${CKPT_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# ── Sequential training: 5 agents, all 7B, 200 steps each ───────────────────
# Order chosen so the most structurally distinct agents train consecutively,
# reducing any interference from VRAM residue between jobs.
AGENTS=("thermal" "computational" "structural" "communications" "probe_systems")
TOTAL=${#AGENTS[@]}

for i in "${!AGENTS[@]}"; do
    AGENT="${AGENTS[$i]}"
    STEP=$((i + 1))
    echo ""
    echo "$(date) --- [${STEP}/${TOTAL}] ${AGENT} (Qwen2.5-7B, 200 steps) ---"

    python training/train_sub_agent.py \
        --agent     "${AGENT}" \
        --model_size 7b \
        --steps     200 \
        --batch_size 4 \
        --output_dir "${CKPT_DIR}" \
        ${PUSH_FLAG}

    echo "$(date) ${AGENT} DONE — checkpoint: ${CKPT_DIR}/${AGENT}/"
done

echo ""
echo "=== Card 2 Phase 1 COMPLETE: $(date) ==="
echo "Agents trained: ${AGENTS[*]}"
echo "Checkpoints at: ${CKPT_DIR}/{thermal,computational,structural,communications,probe_systems}/"
