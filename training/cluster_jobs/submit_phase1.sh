#!/bin/bash
# VyomRaksha — Phase 1 Submission Wrapper
#
# Submits the full Phase 1 → 1.5 → reward model pipeline with proper
# SLURM job dependencies. Run this once from the project root.
#
# Usage:
#   cd /path/to/Meta\ Hack
#   bash training/cluster_jobs/submit_phase1.sh
#
# What gets submitted:
#   1. phase1_card1.sh  (GPU 1: threat→power→fuel)       → immediately
#   2. phase1_card2.sh  (GPU 2: thermal→…→probe_systems) → immediately, parallel with 1
#   3. phase1_5.sh      (all 8 agents, 200 steps)        → after 1 AND 2 complete
#   4. reward_model.sh  (Qwen2.5-3B, 150 steps)          → after 3 completes
#
# Log job IDs to progress_r2.md manually after reviewing the output below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${PROJECT_DIR}/logs"

echo "=== VyomRaksha Phase 1 Pipeline Submission ==="
echo "Project: ${PROJECT_DIR}"
echo "Submitting from: $(pwd)"
echo ""

# ── Submit Phase 1 cards in parallel ────────────────────────────────────────
echo "Submitting phase1_card1.sh (GPU 1: threat → power → fuel)..."
JOB_CARD1=$(sbatch --parsable "${SCRIPT_DIR}/phase1_card1.sh")
echo "  Card 1 job ID: ${JOB_CARD1}"

echo "Submitting phase1_card2.sh (GPU 2: thermal → computational → structural → comms → probe_systems)..."
JOB_CARD2=$(sbatch --parsable "${SCRIPT_DIR}/phase1_card2.sh")
echo "  Card 2 job ID: ${JOB_CARD2}"

# ── Submit Phase 1.5 with dependency on both Phase 1 cards ──────────────────
echo ""
echo "Submitting phase1_5.sh (dependency: afterok:${JOB_CARD1}:${JOB_CARD2})..."
JOB_PHASE15=$(sbatch --parsable \
    --dependency="afterok:${JOB_CARD1}:${JOB_CARD2}" \
    "${SCRIPT_DIR}/phase1_5.sh")
echo "  Phase 1.5 job ID: ${JOB_PHASE15}"

# ── Submit reward model with dependency on Phase 1.5 ────────────────────────
echo ""
echo "Submitting reward_model.sh (dependency: afterok:${JOB_PHASE15})..."
JOB_REWARD=$(sbatch --parsable \
    --dependency="afterok:${JOB_PHASE15}" \
    "${SCRIPT_DIR}/reward_model.sh")
echo "  Reward model job ID: ${JOB_REWARD}"

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Submission complete ==="
echo ""
echo "Job chain:"
echo "  [${JOB_CARD1}] phase1_card1  ─┐"
echo "                                ├─▶ [${JOB_PHASE15}] phase1_5 ─▶ [${JOB_REWARD}] reward_model"
echo "  [${JOB_CARD2}] phase1_card2  ─┘"
echo ""
echo "Monitor:"
echo "  squeue -l -u \$USER"
echo "  tail -f logs/phase1_card1_${JOB_CARD1}.log"
echo "  tail -f logs/phase1_card2_${JOB_CARD2}.log"
echo ""
echo "Log these job IDs in progress_r2.md:"
echo "  Phase 1 Card 1:  ${JOB_CARD1}"
echo "  Phase 1 Card 2:  ${JOB_CARD2}"
echo "  Phase 1.5:       ${JOB_PHASE15}"
echo "  Reward model:    ${JOB_REWARD}"
echo ""
echo "When Phase 1.5 and reward model complete, submit Phase 2:"
echo "  sbatch training/cluster_jobs/phase2_sarvadrishi.sh  (create in R2-8.2)"
