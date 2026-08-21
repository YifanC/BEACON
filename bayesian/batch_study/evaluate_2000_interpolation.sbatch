#!/bin/bash
#SBATCH --job-name=beacon-interp2000
#SBATCH --partition=ampere
#SBATCH --account=neutrino:default@ampere
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/sdf/home/i/iatif/REAL_BEACON/bayesian/batch_study/2000cm/provenance/nominal-best-interpolation-%j.out

set -euo pipefail
BAYESIAN=/sdf/home/i/iatif/REAL_BEACON/bayesian
UPSTREAM=/sdf/home/i/iatif/larnd-sim-jax
CONTAINER=/sdf/group/neutrino/pgranger/larnd-sim-jax.sif

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export LARNDSIM_MEMORY_EFFICIENT_SIGNALS=1
export LARNDSIM_SIGNAL_CONTRIB_BLOCK_SIZE=500000
export LARNDSIM_PROBABILISTIC_BLOCK_SIZE=128
export PYTHONPATH="$BAYESIAN/workflows:$BAYESIAN/workflows/six_d:$UPSTREAM:$UPSTREAM/src"

cd "$UPSTREAM"
singularity exec --nv --bind /sdf:/sdf "$CONTAINER" \
  python3 "$BAYESIAN/batch_study/evaluate_2000_interpolation.py"
