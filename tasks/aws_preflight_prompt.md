# VyomRaksha — AWS Pre-Flight Audit & Fix Prompt
# Paste this entire file into a Claude Code session opened at C:/Dev/Meta Hack/

You are performing a pre-AWS audit and fix pass on the VyomRaksha training pipeline.
Read CLAUDE.md and progress_r2.md first for full context.

Your job is to:
1. Fix every bug listed below IN THE EXISTING FILES (do not create new files unless specified)
2. Create three new files: training/aws/setup_instance.sh, training/aws/run_training.sh, training/aws/README.md
3. Verify the fixes locally by running smoke-tests on all five training scripts
4. Update progress_r2.md with a new session log entry

---

## BUGS TO FIX IN EXISTING FILES

### BUG 1 — training/cluster_jobs/phase2_sarvadrishi.sh
Line that reads:
    PUSH_FLAG="--push_to_hub True"
Fix: change to:
    PUSH_FLAG="--push_to_hub"
Reason: train_sarvadrishi.py defines --push_to_hub as action="store_true".
Passing a value "True" breaks argparse.

### BUG 2 — training/train_emergency.py
In function _run_agent_grpo(), the GRPOTrainer instantiation always passes
processing_class=tokenizer unconditionally. On TRL < 0.12 this raises TypeError.
train_sub_agent.py and train_sarvadrishi.py both do the version check correctly.

Find this block in _run_agent_grpo():
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=reward_fn, args=config, train_dataset=dataset,
    )

Replace with:
    import trl as _trl_mod
    _grpo_kwargs = (
        {"processing_class": tokenizer}
        if tuple(int(x) for x in _trl_mod.__version__.split(".")[:2]) >= (0, 12)
        else {"tokenizer": tokenizer}
    )
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    trainer = GRPOTrainer(
        model=model,
        **_grpo_kwargs,
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
    )

### BUG 3 — training/train_reward_model.py
The Unsloth branch in _load_model_and_tokenizer() loads a causal LM via
FastLanguageModel and wraps it as a causal LoRA model. But the reward model
requires a sequence classification head (num_labels=1). The HF fallback
correctly uses AutoModelForSequenceClassification. If Unsloth is installed
(which it will be on EC2), the wrong model type gets trained silently.

Fix: Remove the entire Unsloth try/except block from _load_model_and_tokenizer()
in train_reward_model.py ONLY. Keep the HF BitsAndBytes block as the first
and only path. Add this comment above the function:
    # Reward model requires a classification head (num_labels=1).
    # Unsloth FastLanguageModel is causal-only and cannot produce a classifier.
    # We use HF AutoModelForSequenceClassification directly, even when Unsloth
    # is installed. Do NOT add an Unsloth branch here.

### BUG 4 — ALL six cluster_jobs scripts
Files: phase1_card1.sh, phase1_card2.sh, phase1_5.sh,
       reward_model.sh, phase2_sarvadrishi.sh, phase3_emergency.sh

These scripts have SPIT-specific hardcoded paths and SLURM variables that
fail silently or crash on EC2. Apply ALL of the following to EACH script:

  (a) Remove every line containing "module load" — EC2 DLAMI does not use
      environment modules.

  (b) Replace the PROJECT_DIR assignment. Any script that has:
          PROJECT_DIR="${SLURM_SUBMIT_DIR}"
      Change to:
          PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
      Scripts that already use PROJECT_DIR="$(pwd)" are fine — leave them.
      phase1_card1.sh and phase1_card2.sh use SLURM_SUBMIT_DIR — fix those.
      phase1_5.sh also uses SLURM_SUBMIT_DIR — fix that too.

  (c) Replace the venv activation line. Any line that reads:
          source ~/vyom_env/bin/activate
      Change to:
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

  (d) Replace the cd line. Any line that reads:
          cd ~/Meta-Hack-VyomRaksha
      Change to:
          cd "${PROJECT_DIR}"

  (e) Replace all ${SLURM_JOB_ID} references with ${SLURM_JOB_ID:-local_$$}
      so logs show a meaningful ID on EC2 instead of empty string.

  (f) phase2_sarvadrishi.sh only:
      Change:   #SBATCH --gres=gpu:2
      To:       #SBATCH --gres=gpu:1
      Change:   #SBATCH --cpus-per-task=112
      To:       #SBATCH --cpus-per-task=8
      Reason: train_sarvadrishi.py is single-process. Requesting 2 GPUs on a
      g5.2xlarge (which has 1 GPU) will cause the SLURM job to hang waiting
      for a second GPU that never comes. On EC2 we run directly anyway, but
      the SBATCH directives should be correct for portability.

  (g) phase2_sarvadrishi.sh only: also apply Bug 1 fix (already listed above
      but apply it here too as part of the cluster_jobs pass).

### BUG 5 — .gitignore missing model weight exclusions
Open the existing .gitignore and append the following block at the end
if these patterns are not already present:

    # Model weights — push to HF Hub, not GitHub (100MB file limit)
    training/checkpoints/**/*.safetensors
    training/checkpoints/**/*.bin
    training/checkpoints/**/*.gguf
    training/checkpoints/**/*.pt
    training/checkpoints/**/*.pth
    .hf_cache/

    # AWS training logs (can be very large)
    logs/aws/*.log

    # Keep PID files and small metadata
    !logs/aws/*.pid
    !logs/aws/*.json

---

## NEW FILE 1 — training/aws/setup_instance.sh

Create this file with the following exact content:

```
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
```

---

## NEW FILE 2 — training/aws/run_training.sh

Create this file with the following exact content:

```
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
            if [ "${AGENT}" = "threat" ]; then SIZE="14b"; BATCH=2; fi
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
```

---

## NEW FILE 3 — training/aws/README.md

Create this file with the following exact content:

```
# VyomRaksha — AWS Training Guide

## Instance Setup

| Instance | Account | Type | Role | Est. Time | Est. Cost |
|---|---|---|---|---|---|
| Instance 1 | Divit (you) | g5.2xlarge | card1 + phase1_5_c1 + sarvadrishi | ~22h | ~$27 |
| Instance 2 | Friend 1 | g5.2xlarge | card2 + phase1_5_c2 | ~14h | ~$17 |
| Instance 3 | Friend 2 | g5.xlarge | reward + emergency | ~6h | ~$7 |

Total estimated cost: ~$51 across 3 accounts. Well within combined budget.

## AMI

Search in EC2 console:
  Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)

Use this exact AMI — it has CUDA, cuDNN, and conda pre-installed.

## Disk

Set EBS root volume to 200GB when launching. Cost: ~$1-2 extra.
Models + checkpoints + cache will use ~150GB.

## Region

Use us-east-1 (N. Virginia) — best g5 instance availability.
All 3 instances should be in the same region for easy scp between them.

## Step-by-Step Launch Order

### Step 1 — Launch all 3 instances (do this first, in parallel)

  All instances: g5.2xlarge (Instances 1+2) or g5.xlarge (Instance 3)
  AMI: Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)
  EBS: 200 GB gp3
  Security group: allow SSH (port 22) from your IP only
  Key pair: create or reuse one you have

### Step 2 — Setup each instance (run in parallel, all 3 at once)

  On each instance:
    ssh -i your-key.pem ubuntu@<instance-ip>
    git clone https://github.com/Xx-D3V-xX/Meta-Hack-VyomRaksha.git
    cd Meta-Hack-VyomRaksha
    export HF_TOKEN=hf_...
    export GITHUB_TOKEN=ghp_...   # optional
    bash training/aws/setup_instance.sh <role>
    # role: card1 for Instance 1, card2 for Instance 2, reward for Instance 3

  Wait for ALL THREE setup scripts to print "Setup complete" before proceeding.

### Step 3 — Start Phase 1 (Instances 1 and 2 in parallel)

  Instance 1:
    bash training/aws/run_training.sh card1 --push_hub

  Instance 2 (simultaneously):
    bash training/aws/run_training.sh card2 --push_hub

  Estimated duration: ~12h (Instance 1), ~10h (Instance 2)
  You can disconnect SSH — nohup keeps training alive.

### Step 4 — Phase 1.5 (Instances 1 and 2, after their Phase 1 completes)

  Instance 1 (after card1 done):
    bash training/aws/run_training.sh phase1_5_c1 --push_hub

  Instance 2 (after card2 done):
    bash training/aws/run_training.sh phase1_5_c2 --push_hub

  Estimated duration: ~8h each.

### Step 5 — Reward Model (Instance 3, after BOTH Phase 1.5 jobs complete)

  Wait for both Instance 1 and Instance 2 Phase 1.5 to finish.
  Then on Instance 3:
    bash training/aws/run_training.sh reward --push_hub

  Estimated duration: ~2h.

### Step 6 — Sync reward model checkpoint to Instance 1

  On Instance 1:
    scp -i your-key.pem -r ubuntu@<instance3-ip>:~/Meta-Hack-VyomRaksha/training/checkpoints/sarvadrishi_reward_model/ \
        ~/Meta-Hack-VyomRaksha/training/checkpoints/

  Alternatively if you pushed to HF Hub:
    python -c "
    from huggingface_hub import snapshot_download
    snapshot_download('D3V1601/VyomRaksha-sarvadrishi-reward-model',
                      local_dir='training/checkpoints/sarvadrishi_reward_model')
    "

### Step 7 — Phase 2 SarvaDrishti (Instance 1, after reward model is synced)

  Instance 1:
    bash training/aws/run_training.sh sarvadrishi --push_hub --push_github

  Estimated duration: ~8h.

### Step 8 — Phase 3 Emergency (Instance 3, optional, can run later)

  Instance 3 (after sarvadrishi checkpoint is synced):
    bash training/aws/run_training.sh emergency --push_hub

  Estimated duration: ~4h.

## Monitoring

  # Live log
  tail -f ~/Meta-Hack-VyomRaksha/logs/aws/<role>_*.log

  # GPU utilisation (should be near 100% during training)
  watch -n 5 nvidia-smi

  # Check if training process is alive
  ps aux | grep train_

  # Check by PID file
  ps -p $(cat ~/Meta-Hack-VyomRaksha/logs/aws/<role>.pid)

## If Training Crashes

  1. Check the log: tail -100 logs/aws/<role>_*.log
  2. Common causes:
     - OOM: reduce batch_size by half in the cluster_jobs/*.sh script
     - Download failed: re-run setup_instance.sh <role>
     - Import error: pip install --upgrade trl transformers unsloth
  3. Training scripts are resumable — re-run the same run_training.sh command.
     The training scripts save checkpoints every steps//5 steps.

## Checkpoint Locations After Training

  All sub-agents:  training/checkpoints/<agent_name>/
  Reward model:    training/checkpoints/sarvadrishi_reward_model/
  SarvaDrishti:    training/checkpoints/sarvadrishi/
  Emergency P3:    training/checkpoints/emergency_phase3/<agent_name>/

## Setting LORA Paths on HF Space (after training)

  In HuggingFace Space settings, add these environment variables:
    LORA_THREAT_PATH=training/checkpoints/threat
    LORA_POWER_PATH=training/checkpoints/power
    LORA_FUEL_PATH=training/checkpoints/fuel
    LORA_THERMAL_PATH=training/checkpoints/thermal
    LORA_COMPUTATIONAL_PATH=training/checkpoints/computational
    LORA_STRUCTURAL_PATH=training/checkpoints/structural
    LORA_COMMUNICATIONS_PATH=training/checkpoints/communications
    LORA_PROBE_SYSTEMS_PATH=training/checkpoints/probe_systems
    R2_MODE=true

  Then: git push hf main --force
```

---

## LOCAL VERIFICATION

After making all edits, run each command below from the project root (C:\Dev\Meta Hack).
All must exit with code 0. Record the exact output in the progress_r2.md session entry.

Command 1:
  python training/train_sub_agent.py --agent power --steps 3 --batch_size 1 --skip_model_load

Command 2:
  python training/train_sub_agent.py --agent threat --steps 3 --batch_size 1 --skip_model_load

Command 3:
  python training/train_reward_model.py --steps 3 --batch_size 1 --model_size tiny

Command 4:
  python training/train_sarvadrishi.py --steps 3 --batch_size 1 --model_size tiny

Command 5:
  python training/train_emergency.py --steps 3

Command 6:
  python training/eval_pipeline.py --n_eval_episodes 1 --output_dir /tmp/vyom_eval_test

Command 7:
  pytest tests/ -q --tb=short

All 7 must pass. If any fail, fix the failure before proceeding to AWS.

---

## progress_r2.md SESSION LOG ENTRY

Add a new session block at the END of the session log section in progress_r2.md.
Use today's date: 2026-04-23.
Session name: "AWS Pre-Flight Audit — 2026-04-23"

Include in the entry:
- List of all 5 bugs fixed (Bugs 1-5 as numbered above)
- Output of all 7 verification commands (exact exit codes and summary lines)
- pytest result (pass count / total)
- Files created: training/aws/setup_instance.sh, training/aws/run_training.sh, training/aws/README.md
- Files modified: train_emergency.py, train_reward_model.py, phase2_sarvadrishi.sh,
  all six cluster_jobs/*.sh scripts, .gitignore
- Status: "AWS training pipeline verified and ready to launch"
- Next session: "Launch 3 AWS EC2 instances per training/aws/README.md"
