#!/bin/bash
#SBATCH --job-name=beacon-target2000
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu@ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/home/i/iatif/REAL_BEACON/bayesian/batch_study/shared/logs/target-2000-memory-safe-%j.out

set -euo pipefail
BAYESIAN=/sdf/home/i/iatif/REAL_BEACON/bayesian
UPSTREAM=/sdf/home/i/iatif/larnd-sim-jax
CONTAINER=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
MONITOR="$BAYESIAN/batch_study/shared/logs/target-2000-memory-safe-${SLURM_JOB_ID}-nvidia.csv"

export LARNDSIM_MEMORY_EFFICIENT_SIGNALS=1
export PYTHONPATH="$BAYESIAN/workflows:$BAYESIAN/workflows/six_d:$UPSTREAM:$UPSTREAM/src"
cd "$UPSTREAM"

(
  while true; do
    date +%s
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 1
  done
) > "$MONITOR" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

singularity exec --nv --bind /sdf:/sdf "$CONTAINER" \
  python3 "$BAYESIAN/batch_study/prepare_target.py" \
    --length-cm 2000 \
    --output "$BAYESIAN/batch_study/shared/target_2000cm.npz"
