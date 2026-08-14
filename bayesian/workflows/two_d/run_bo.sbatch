#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=twod_bo
#SBATCH --output=/sdf/home/i/iatif/larnd-sim-jax/optimize/bayesian/.local/two_d/%x-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=02:00:00
set -euo pipefail
REPO=/sdf/home/i/iatif/larnd-sim-jax
BAY=$REPO/optimize/bayesian
CONFIG=${CONFIG:-$BAY/workflows/two_d/configs/eField_lifetime.yaml}
cd "$REPO"
apptainer exec --nv -B /sdf,/fs /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \
  env PYTHONPATH="$BAY/.local/shared/site-packages:$REPO:$REPO/src" \
  python3 -u "$BAY/workflows/two_d/run_bo.py" --config "$CONFIG"
