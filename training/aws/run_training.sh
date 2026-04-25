#!/bin/bash
# VyomRaksha — AWS EC2 Training Launcher
# Replaces SLURM submit_phase1.sh + submit_phase2.sh for EC2.
# Uses nohup so training survives SSH disconnection.
#
# Usage:
#   export HF_TOKEN=hf_your_token_here
#   bash training/aws/run_training.sh <ROLE> [--push_hub] [--push_github]
#
# ROLES (run in this order across your 3 instances):
#
#   INSTANCE 1 (your account, g5.2xlarge):
#     card1        Phase 1 Card 1: threat(14B,300) → power(7B,200) → fuel(7B,200)
#     phase1_5_c1  Phase 1.5 Card 1 agents: threat, power, fuel (200 steps each, skip_sft)
#     sarvadrishi  Phase 2 SarvaDrishti (run after reward model is done on Instance 3)
#
#   INSTANCE 2 (friend 1's account, g5.2xlarge):
#     card2        Phase 1 Card 2: thermal→computational→structural→communications→probe_systems
#     phase1_5_c2  Phase 1.5 Card 2 agents (200 steps each, skip_sft)
#
#   INSTANCE 3 (friend 2's account, g5.xlarge):
#     reward       Reward model: Qwen2.5-3B, 150 steps (run after phase1_5 complete on both)
#     emergency    Phase 3 emergency calibration (run after sarvadrishi complete on Instance 1)
#
# Flags:
#   --push_hub      Push checkpoints to HuggingFace Hub (D3V1601) after each agent
#   --push_github   Push code + dashboard data (NOT weights) to GitHub after training
#
# Example:
#   export HF_TOKEN=hf_...
#   bash training/aws/run_training.sh card1 --push_hub --push_github

set -euo pipefail

ROLE="${1:-}"
if [ -z "${ROLE}" ]; then
    echo "ERROR: ROLE argument required."
    echo "Usage: bash training/aws/run_training.sh <ROLE> [--push_hub] [--push_github]"
    echo "Valid roles: card1, card2, phase1_5_c1, phase1_5_c2, reward, sarvadrishi, emergency"
    exit 1
fi

# Parse flags
PUSH_HUB=false
PUSH_GITHUB=false
for arg in "${@:2}"; do
    case "${arg}" in
        --push_hub)     PUSH_HUB=true ;;
        --push_github)  PUSH_GITHUB=true ;;
        *) echo "WARNING: Unknown flag '${arg}' — ignored" ;;
    esac
done

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
JOBS_DIR="${PROJECT_DIR}/training/cluster_jobs"
LOG_DIR="${PROJECT_DIR}/logs/aws"
CKPT_DIR="${PROJECT_DIR}/training/checkpoints"

mkdir -p "${LOG_DIR}" "${CKPT_DIR}"

# ── Environment ───────────────────────────────────────────────────────────────
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
    conda activate pytorch 2>/dev/null || true
fi

export HF_HOME="${PROJECT_DIR}/.hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

PUSH_HUB_FLAG=""
if [ "${PUSH_HUB}" = true ] && [ -n "${HF_TOKEN:-}" ]; then
    PUSH_HUB_FLAG="--push_to_hub"
    echo "HF Hub push: ENABLED (D3V1601)"
else
    echo "HF Hub push: DISABLED (set HF_TOKEN and --push_hub to enable)"
fi

echo "=========================================="
echo " VyomRaksha Training Launcher"
echo " Role:    ${ROLE}"
echo " Project: ${PROJECT_DIR}"
echo " Logs:    ${LOG_DIR}"
echo " Started: $(date)"
echo "=========================================="

# ── Helper: launch a job with nohup ──────────────────────────────────────────
launch() {
    local SCRIPT="$1"
    local LABEL="$2"
    local LOG_FILE="${LOG_DIR}/${LABEL}_$(date +%Y%m%d_%H%M%S).log"
    local PID_FILE="${LOG_DIR}/${LABEL}.pid"

    echo ""
    echo "Launching: ${LABEL}"
    echo "Script:    ${SCRIPT}"
    echo "Log:       ${LOG_FILE}"

    nohup bash "${SCRIPT}" > "${LOG_FILE}" 2>&1 &
    local PID=$!
    echo "${PID}" > "${PID_FILE}"

    echo "PID:       ${PID}"
    echo "Monitor:   tail -f ${LOG_FILE}"
    echo "Status:    ps -p \$(cat ${PID_FILE})"
    echo ""
}

# ── Helper: wait for a PID file to indicate completion ───────────────────────
wait_for() {
    local LABEL="$1"
    local PID_FILE="${LOG_DIR}/${LABEL}.pid"
    if [ ! -f "${PID_FILE}" ]; then
        echo "ERROR: PID file not found for ${LABEL}: ${PID_FILE}"
        echo "Is the ${LABEL} job running on this instance?"
        exit 1
    fi
    local PID
    PID=$(cat "${PID_FILE}")
    echo "Waiting for ${LABEL} (PID ${PID}) to complete..."
    while kill -0 "${PID}" 2>/dev/null; do
        sleep 30
        echo "  Still running (PID ${PID}) — $(date)"
    done
    echo "${LABEL} complete."
}

# ── Helper: inline sequential training (no separate script) ──────────────────
launch_phase1_5() {
    local AGENTS=("$@")
    local LABEL="phase1_5_${ROLE#phase1_5_}"
    local LOG_FILE="${LOG_DIR}/${LABEL}_$(date +%Y%m%d_%H%M%S).log"
    local PID_FILE="${LOG_DIR}/${LABEL}.pid"

    echo "Launching Phase 1.5 for agents: ${AGENTS[*]}"
    echo "Log: ${LOG_FILE}"

    (
        cd "${PROJECT_DIR}"
        for AGENT in "${AGENTS[@]}"; do
            echo "$(date) --- Phase 1.5: ${AGENT} ---"
            SIZE="7b"
            BATCH=4
            if [ "${AGENT}" = "threat" ]; then SIZE="14b"; BATCH=4; fi
            python training/train_sub_agent.py \
                --agent "${AGENT}" \
                --model_size "${SIZE}" \
                --steps 200 \
                --batch_size "${BATCH}" \
                --skip_sft \
                --output_dir "${CKPT_DIR}" \
                ${PUSH_HUB_FLAG}
            echo "$(date) ${AGENT} Phase 1.5 DONE"
        done
    ) > "${LOG_FILE}" 2>&1 &

    local PID=$!
    echo "${PID}" > "${PID_FILE}"
    echo "Phase 1.5 PID: ${PID}"
    echo "Monitor: tail -f ${LOG_FILE}"
}

# ── Route by ROLE ─────────────────────────────────────────────────────────────
case "${ROLE}" in

    card1)
        # Instance 1 — Phase 1 Card 1
        # Runs: threat (14B, 300 steps) → power (7B, 200) → fuel (7B, 200)
        launch "${JOBS_DIR}/phase1_card1.sh" "phase1_card1"
        echo "Card 1 launched. When it completes, run:"
        echo "  bash training/aws/run_training.sh phase1_5_c1 ${PUSH_HUB:+--push_hub} ${PUSH_GITHUB:+--push_github}"
        ;;

    card2)
        # Instance 2 — Phase 1 Card 2
        # Runs: thermal, computational, structural, communications, probe_systems (all 7B, 200 steps)
        launch "${JOBS_DIR}/phase1_card2.sh" "phase1_card2"
        echo "Card 2 launched. When it completes, run:"
        echo "  bash training/aws/run_training.sh phase1_5_c2 ${PUSH_HUB:+--push_hub}"
        ;;

    phase1_5_c1)
        # Instance 1 — Phase 1.5 Card 1 agents
        launch_phase1_5 "threat" "power" "fuel"
        echo "Phase 1.5 Card 1 launched."
        echo "Wait for BOTH phase1_5_c1 AND phase1_5_c2 (on Instance 2) before running 'reward' on Instance 3."
        ;;

    phase1_5_c2)
        # Instance 2 — Phase 1.5 Card 2 agents
        launch_phase1_5 "thermal" "computational" "structural" "communications" "probe_systems"
        echo "Phase 1.5 Card 2 launched."
        echo "When done, notify Instance 3 to start: bash training/aws/run_training.sh reward"
        ;;

    reward)
        # Instance 3 — Reward model
        # IMPORTANT: Run only after BOTH phase1_5_c1 and phase1_5_c2 are complete
        # and their checkpoints are available (pulled from HF Hub or synced via scp).
        echo "Starting reward model training."
        echo "PREREQUISITE: Confirm phase1_5_c1 and phase1_5_c2 are both complete."
        read -r -p "Are both Phase 1.5 jobs complete? (y/N): " CONFIRM
        if [[ "${CONFIRM}" != "y" && "${CONFIRM}" != "Y" ]]; then
            echo "Aborted. Run after Phase 1.5 is complete on both Instance 1 and Instance 2."
            exit 0
        fi
        launch "${JOBS_DIR}/reward_model.sh" "reward_model"
        echo "Reward model launched."
        echo "When it completes, notify Instance 1 to run:"
        echo "  bash training/aws/run_training.sh sarvadrishi --push_hub"
        ;;

    sarvadrishi)
        # Instance 1 — Phase 2 SarvaDrishti
        # IMPORTANT: Run only after reward model checkpoint is available.
        echo "Starting SarvaDrishti Phase 2 training."
        echo "PREREQUISITE: Reward model checkpoint must exist at:"
        echo "  ${CKPT_DIR}/sarvadrishi_reward_model/"
        if [ ! -d "${CKPT_DIR}/sarvadrishi_reward_model" ]; then
            echo "ERROR: Reward model checkpoint not found."
            echo "Pull it from HF Hub or scp from Instance 3 first."
            echo "  scp -r ubuntu@<instance3-ip>:~/Meta-Hack-VyomRaksha/training/checkpoints/sarvadrishi_reward_model/ ${CKPT_DIR}/"
            exit 1
        fi
        launch "${JOBS_DIR}/phase2_sarvadrishi.sh" "phase2_sarvadrishi"
        echo "SarvaDrishti launched."
        echo "When complete, Instance 3 can run Phase 3:"
        echo "  bash training/aws/run_training.sh emergency --push_hub"
        ;;

    emergency)
        # Instance 3 — Phase 3 emergency calibration (optional — can run later)
        echo "Starting Phase 3 emergency authority calibration."
        echo "PREREQUISITE: SarvaDrishti checkpoint must exist at ${CKPT_DIR}/sarvadrishi/"
        if [ ! -d "${CKPT_DIR}/sarvadrishi" ]; then
            echo "ERROR: SarvaDrishti checkpoint not found."
            echo "Pull it from HF Hub or scp from Instance 1 first."
            exit 1
        fi
        launch "${JOBS_DIR}/phase3_emergency.sh" "phase3_emergency"
        echo "Phase 3 launched."
        ;;

    *)
        echo "ERROR: Unknown role '${ROLE}'"
        echo "Valid roles: card1, card2, phase1_5_c1, phase1_5_c2, reward, sarvadrishi, emergency"
        exit 1
        ;;
esac

# ── Optional: GitHub code push after training ─────────────────────────────────
if [ "${PUSH_GITHUB}" = true ]; then
    LABEL="${ROLE}"
    PID_FILE="${LOG_DIR}/${LABEL}.pid"

    if [ ! -f "${PID_FILE}" ]; then
        # phase1_5 roles use a different PID file name
        PID_FILE="${LOG_DIR}/phase1_5_${ROLE#phase1_5_}.pid"
    fi

    echo "GitHub push scheduled — will run after training completes."
    (
        # Wait for training PID to exit
        if [ -f "${PID_FILE}" ]; then
            TRAINING_PID=$(cat "${PID_FILE}")
            while kill -0 "${TRAINING_PID}" 2>/dev/null; do
                sleep 60
            done
        fi
        echo "Training complete — pushing code to GitHub"
        cd "${PROJECT_DIR}"
        git add training/cluster_jobs/ training/aws/ dashboard/data/ progress_r2.md --force 2>/dev/null || true
        git commit -m "chore: AWS training complete — ${ROLE} $(date +%Y-%m-%d)" 2>/dev/null || echo "Nothing to commit"
        git push origin main 2>/dev/null || echo "GitHub push failed — check GITHUB_TOKEN"
        echo "GitHub push done."
    ) &
    echo "GitHub push watcher PID: $!"
fi

echo ""
echo "=========================================="
echo " Launcher done: $(date)"
echo " All training jobs are running in background."
echo " SSH disconnect is safe — nohup protects the process."
echo ""
echo " Useful commands:"
echo "   tail -f ${LOG_DIR}/<role>_*.log    # live log"
echo "   ps aux | grep train                 # check if running"
echo "   nvidia-smi                          # GPU utilisation"
echo "=========================================="
