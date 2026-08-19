#!/bin/bash
#SBATCH --job-name=beacon-eval2000
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu@ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/home/i/iatif/REAL_BEACON/bayesian/batch_study/2000cm/logs/objective-nominal-%j.out

set -euo pipefail
BAYESIAN=/sdf/home/i/iatif/REAL_BEACON/bayesian
UPSTREAM=/sdf/home/i/iatif/larnd-sim-jax
CONTAINER=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
MONITOR="$BAYESIAN/batch_study/2000cm/logs/objective-nominal-${SLURM_JOB_ID}-nvidia.csv"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LARNDSIM_MEMORY_EFFICIENT_SIGNALS=1
export LARNDSIM_SIGNAL_CONTRIB_BLOCK_SIZE=500000
export PYTHONPATH="$BAYESIAN/workflows:$BAYESIAN/workflows/six_d:$UPSTREAM:$UPSTREAM/src"
cd "$UPSTREAM"
(
  while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total \
      --format=csv,noheader,nounits
    sleep 1
  done
) > "$MONITOR" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
singularity exec --nv --bind /sdf:/sdf "$CONTAINER" \
  python3 "$BAYESIAN/batch_study/evaluate_2000_nominal.py"
