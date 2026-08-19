#!/bin/bash
#SBATCH --partition=milano
#SBATCH --account=neutrino:cider-nu
#SBATCH --job-name=scan1d
#SBATCH --output=/sdf/home/i/iatif/REAL_BEACON/bayesian/.local/one_d/scan-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
set -euo pipefail
REPO=/sdf/home/i/iatif/larnd-sim-jax
BAY=$REPO/optimize/bayesian
cd "$REPO"
apptainer exec /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \
  env PYTHONPATH="$BAY/.local/shared/site-packages:$REPO:$REPO/src" \
  python3 -u "$BAY/workflows/one_d/run_scan_analysis.py" \
      --config "$BAY/workflows/one_d/config.yaml"
