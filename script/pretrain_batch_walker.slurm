#!/bin/bash
#SBATCH --job-name=URLB
#SBATCH --partition=big_suma_rtx3090
#SBATCH --qos=big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-49%50
#SBATCH --output=/scratch2/james2602/URLB/logs_baseline_pretrain/%A/%A_%a.out
#SBATCH --error=/scratch2/james2602/URLB/logs_baseline_pretrain/%A/%A_%a.err

set -euo pipefail
BASE_TMP=${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}
mkdir -p "$BASE_TMP"


##### Job settings #####
SEEDS=(1 2 3 4 5 6 7 8 9 10)
DOMAINS=("walker")
AGENT_LABELS=("icm" "proto"  "diayn" "icm_apt"  "ind_apt"  "aps" "smm" "rnd" "disagreement" "cic")
AGENT_KEYS=(   "icm" "proto"  "diayn" "icm_apt"  "ind_apt"  "aps" "smm" "rnd" "disagreement" "cic")
AGENT_DIRS=(   "icm" "proto"  "diayn" "icm_apt"  "ind_apt"  "aps" "smm" "rnd" "disagreement" "cic")


S=${#SEEDS[@]}            # 10
A=${#AGENT_LABELS[@]}     # 10
D=${#DOMAINS[@]}          # 1
TOTAL=$((S * A * D))


IDX1=$(( SLURM_ARRAY_TASK_ID * 2 ))
IDX2=$(( IDX1 + 1 ))

calc_params () {
  local idx=$1
  seed_idx=$(( idx % S ))
  agent_idx=$(( (idx / S) % A ))
  domain_idx=$(( idx / (S * A) ))
  SEED=${SEEDS[$seed_idx]}
  DOMAIN=${DOMAINS[$domain_idx]}
  AGENT_LABEL=${AGENT_LABELS[$agent_idx]}
  AGENT_KEY=${AGENT_KEYS[$agent_idx]}
  AGENT_DIR=${AGENT_DIRS[$agent_idx]}
}


run_one () {
  local idx=$1
  calc_params "$idx"

  echo "[RUN] idx=${idx} agent='${AGENT_LABEL}' key='${AGENT_KEY}' domain='${DOMAIN}' seed=${SEED}"

  # 각 Run별 로그 디렉토리 (공백 버그 제거됨)
  LOG_BASE="/scratch2/james2602/URLB/outputs_pretrain/${DOMAIN}/${AGENT_DIR}/seed_${SEED}"
  LOG_DIR="${LOG_BASE}/logs"
  mkdir -p "$LOG_DIR"

  OUT="${LOG_DIR}/job${SLURM_JOB_ID}_array${SLURM_ARRAY_TASK_ID}.out"
  ERR="${LOG_DIR}/job${SLURM_JOB_ID}_array${SLURM_ARRAY_TASK_ID}.err"

  # 각 작업에 2코어 할당 요청(+코어 바인딩), 같은 GPU 공유
  srun -c 2 --overlap --cpu-bind=threads bash -lc "
    set -euo pipefail
    set +u
    if command -v conda >/dev/null 2>&1; then
      source \"\$(conda info --base)/etc/profile.d/conda.sh\" 2>/dev/null \
      || source \"\$HOME/miniconda3/etc/profile.d/conda.sh\" 2>/dev/null
      conda activate urlb || true
    fi
    set -u

    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export MUJOCO_GL=egl
    export MKL_SERVICE_FORCE_INTEL=1
    export WANDB_START_METHOD=thread
    export WANDB__SERVICE_WAIT=300
    export HYDRA_FULL_ERROR=1
    export PYTHONUNBUFFERED=1

   
    cd \"\$HOME/URLB/url_benchmark\"
    python pretrain.py \
      seed=${SEED} \
      domain=${DOMAIN} \
      agent=${AGENT_KEY} \
      obs_type="states" \
      action_repeat=1 \
      use_wandb=True \
      snapshot_dir="snapshot" \
      wandb_project="urlb_pretraining_walker" \
      hydra.run.dir="${LOG_BASE}"
  " >"$OUT" 2>"$ERR" &
}

echo "HOME=[$HOME]"
ls -ld "$HOME" "$HOME/URLB" "$HOME/URLB/url_benchmark" || true


if (( IDX1 < TOTAL )); then
  run_one "$IDX1"
fi

if (( IDX2 < TOTAL )); then
  run_one "$IDX2"
fi

wait

echo "=========== seff & dmesg ==========="
seff "$SLURM_JOB_ID" 2>/dev/null || true
dmesg | egrep -i "oom|Out of memory|BUS error|bad page" | tail -n 100 || true
echo "===================================="