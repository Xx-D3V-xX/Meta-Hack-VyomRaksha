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
