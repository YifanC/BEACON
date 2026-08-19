#!/bin/bash
#SBATCH --job-name=beacon-memeq
#SBATCH --partition=ampere
#SBATCH --account=neutrino:cider-nu@ampere
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/sdf/home/i/iatif/REAL_BEACON/bayesian/batch_study/shared/logs/memory-equivalence-%j.out

set -euo pipefail
cd /sdf/home/i/iatif/larnd-sim-jax
export PYTHONPATH=/sdf/home/i/iatif/REAL_BEACON/bayesian/workflows:/sdf/home/i/iatif/REAL_BEACON/bayesian/workflows/six_d:/sdf/home/i/iatif/larnd-sim-jax:/sdf/home/i/iatif/larnd-sim-jax/src
singularity exec --nv --bind /sdf:/sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \
  python3 /sdf/home/i/iatif/REAL_BEACON/bayesian/batch_study/memory_equivalence.py
