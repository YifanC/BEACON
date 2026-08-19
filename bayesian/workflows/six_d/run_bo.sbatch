#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=sixd_bo
#SBATCH --output=/sdf/home/i/iatif/REAL_BEACON/bayesian/.local/six_d/current/raw/logs/bo-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=06:00:00
set -euo pipefail
REPO=/sdf/home/i/iatif/larnd-sim-jax
BAY=$REPO/optimize/bayesian
mkdir -p "$BAY/.local/six_d/current/raw/logs"
cd "$REPO"
apptainer exec --nv -B /sdf,/fs /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \
  env PYTHONPATH="$BAY/.local/shared/site-packages:$REPO:$REPO/src:$BAY/workflows/six_d" \
  python3 -u "$BAY/workflows/six_d/run_bo.py"
