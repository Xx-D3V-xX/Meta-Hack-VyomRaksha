#!/bin/bash
# VyomRaksha — Submit Phase 2 → Phase 3 pipeline
# Usage: bash training/cluster_jobs/submit_phase2.sh [<REWARD_MODEL_JOB_ID>]
#
# If a reward model job ID is supplied, phase2_sarvadrishi is submitted with
# --dependency=afterok:<id>. Otherwise it starts immediately.
# Phase 3 always depends on Phase 2 completing successfully.
#
# Example:
#   bash training/cluster_jobs/submit_phase2.sh 98765

set -euo pipefail

REWARD_MODEL_JOB_ID="${1:-}"

# ── Submit Phase 2: SarvaDrishti ─────────────────────────────────────────────
if [ -n "${REWARD_MODEL_JOB_ID}" ]; then
    PHASE2_RESULT=$(sbatch \
        --dependency=afterok:"${REWARD_MODEL_JOB_ID}" \
        training/cluster_jobs/phase2_sarvadrishi.sh)
else
    PHASE2_RESULT=$(sbatch training/cluster_jobs/phase2_sarvadrishi.sh)
fi

PHASE2_JOB_ID=$(echo "${PHASE2_RESULT}" | awk '{print $NF}')
echo "Phase 2 (SarvaDrishti) submitted — Job ID: ${PHASE2_JOB_ID}"

# ── Submit Phase 3: Emergency Authority (depends on Phase 2) ─────────────────
PHASE3_RESULT=$(sbatch \
    --dependency=afterok:"${PHASE2_JOB_ID}" \
    training/cluster_jobs/phase3_emergency.sh)

PHASE3_JOB_ID=$(echo "${PHASE3_RESULT}" | awk '{print $NF}')
echo "Phase 3 (Emergency)    submitted — Job ID: ${PHASE3_JOB_ID}"

echo ""
echo "Pipeline queued:"
echo "  Phase 2 (SarvaDrishti) : ${PHASE2_JOB_ID}"
echo "  Phase 3 (Emergency)    : ${PHASE3_JOB_ID}  [depends on ${PHASE2_JOB_ID}]"
echo ""
echo "Monitor with: squeue -u \$(whoami)"
echo "Logs: logs/vy_phase2_sarva_${PHASE2_JOB_ID}.log"
echo "      logs/vy_phase3_emrg_${PHASE3_JOB_ID}.log"
