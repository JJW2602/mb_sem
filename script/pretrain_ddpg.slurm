#!/bin/bash
#SBATCH --job-name=URLB
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-119%50
#SBATCH --output=/scratch2/james2602/URLB/logs_baseline_pretrain/%A/%A_%a.out
#SBATCH --error=/scratch2/james2602/URLB/logs_baseline_pretrain/%A/%A_%a.err

set -euo pipefail
BASE_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$BASE_TMP"


##### Job settings #####
SEEDS=(1 2 3 4 5 6 7 8 9 10)
TASKS=("walker_stand" "walker_walk" "walker_run" "walker_flip" "quadruped_walk" "quadruped_run" "quadruped_stand" "quadruped_jump" "jaco_reach_top_left" "jaco_reach_top_right" "jaco_reach_bottom_left" "jaco_reach_bottom_right")
AGENT_LABELS=("ddpg")
AGENT_KEYS=("ddpg")
AGENT_DIRS=("ddpg")


S=${#SEEDS[@]}            # 10
A=${#AGENT_LABELS[@]}     # 1
T=${#TASKS[@]}            # 12
TOTAL=$((S * A * T))


seed_idx=$(( SLURM_ARRAY_TASK_ID % S ))
agent_idx=$(( (SLURM_ARRAY_TASK_ID / S) % A ))
task_idx=$(( SLURM_ARRAY_TASK_ID / (S * A) ))

SEED=${SEEDS[$seed_idx]}
TASK=${TASKS[$task_idx]}
DOMAIN=${TASK%%_*}
AGENT_LABEL=${AGENT_LABELS[$agent_idx]}
AGENT_KEY=${AGENT_KEYS[$agent_idx]}
AGENT_DIR=${AGENT_DIRS[$agent_idx]}


echo "JOB ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} :: agent_label='${AGENT_LABEL}' key='${AGENT_KEY}' task='${TASK}' domain='${DOMAIN}' seed=${SEED}"


##### Logging directories #####
LOG_BASE="/scratch2/james2602/URLB/outputs_pretrain/${DOMAIN}/${AGENT_DIR}/${TASK}/seed_${SEED}"
LOG_DIR="${LOG_BASE}/logs"
mkdir -p "$LOG_DIR"

OUT="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.out"
ERR="${LOG_DIR}/job${SLURM_JOB_ID}_task${SLURM_ARRAY_TASK_ID}.err"
exec > >(tee -a "$OUT") 2> >(tee -a "$ERR" >&2)


RUN_ID=${SLURM_ARRAY_TASK_ID}
RUN_TAG="${SLURM_JOB_ID}_${RUN_ID}"


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export WANDB_START_METHOD=thread
export WANDB__SERVICE_WAIT=300
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

OBS_TYPE=states
set -eo pipefail

export OMP_NUM_THREADS=${OMP_NUM_THREADS}
export MKL_NUM_THREADS=${MKL_NUM_THREADS}
export MUJOCO_GL=egl
export MKL_SERVICE_FORCE_INTEL=1
export WANDB_START_METHOD=${WANDB_START_METHOD}
export WANDB__SERVICE_WAIT=${WANDB__SERVICE_WAIT}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED}

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null \
  || source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  conda activate urlb || true
fi

echo "HOME=[$HOME]"
ls -ld "$HOME" "$HOME/URLB" "$HOME/URLB/url_benchmark" || true
cd "$HOME/URLB/url_benchmark"


python pretrain.py \
  seed=${SEED} \
  domain=${DOMAIN} \
  task=${TASK} \
  agent=${AGENT_KEY} \
  obs_type=${OBS_TYPE} \
  action_repeat=1 \
  use_wandb=True \
  snapshot_dir="snapshot" \
  wandb_project="urlb_pretraining_ddpg" \
  hydra.run.dir="${LOG_BASE}"

echo "=========== seff & dmesg ==========="
seff $SLURM_JOB_ID 2>/dev/null || true
dmesg | egrep -i "oom|Out of memory|BUS error|bad page" | tail -n 100 || true
echo "====================================" 