#!/bin/bash
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu
#SBATCH --qos=normal
#SBATCH --job-name=sixd_botr_nu
#SBATCH --output=/sdf/home/i/iatif/REAL_BEACON/bayesian/.local/six_d/current/raw/logs/botr-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=07:00:00
set -euo pipefail
REPO=/sdf/home/i/iatif/larnd-sim-jax
BAY=$REPO/optimize/bayesian
mkdir -p "$BAY/.local/six_d/current/raw/logs"
export MPLBACKEND=Agg
export TMPDIR=$BAY/.local/six_d/current/raw/logs
export XDG_CACHE_HOME=$BAY/.local/six_d/current/raw/logs/cache
export MPLCONFIGDIR=$BAY/.local/six_d/current/raw/logs/mpl
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
MODE="${MODE:-trust_region}"
cd "$REPO"
apptainer exec --nv -B /sdf,/fs /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \
  env MPLBACKEND=Agg PYTHONPATH="$REPO:$REPO/src:$BAY/workflows:$BAY/workflows/six_d" \
  python3 -u "$BAY/workflows/six_d/continue_bo.py" --mode "$MODE"
