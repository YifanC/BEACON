#!/bin/bash
#SBATCH --partition=ampere
##SBATCH --account=neutrino:cider-nu
#SBATCH --account=neutrino:dune-ml
##SBATCH --account=mli:nu-ml-dev
#SBATCH --job-name=bo_detector
#SBATCH --output=logs/bo-%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32g
#SBATCH --gpus-per-node=a100:1
#SBATCH --time=36:00:00

set -euo pipefail

BEACON_DIR=${BEACON_DIR:-/sdf/group/neutrino/cyifan/BEACON}
SIF_FILE=${SIF_FILE:-/sdf/group/neutrino/pgranger/larnd-sim-jax.sif}
CONFIG=${CONFIG:-${BEACON_DIR}/configs/bo_detector.yaml}

# Default LARNDSIM_REPO to the larndsim_repo value in the YAML (single source of
# truth); fall back to the user's own checkout if the key is missing.
LARNDSIM_REPO=${LARNDSIM_REPO:-$(grep -E '^larndsim_repo:' "$CONFIG" \
  | head -n1 | sed -E "s/^larndsim_repo:[[:space:]]*//; s/[[:space:]]*#.*//; s/^['\"]//; s/['\"]$//")}
LARNDSIM_REPO=${LARNDSIM_REPO:-/sdf/group/neutrino/cyifan/BEACON/larnd-sim-jax}

mkdir -p logs

apptainer exec --nv \
  -B /sdf,/fs,/sdf/scratch,/lscratch \
  ${SIF_FILE} /bin/bash -c '
    set -e
    # NOTE: deliberately no `pip install .` for larnd-sim-jax — repeated installs
    # re-resolve transitive deps and have downgraded nvidia-cudnn-cu12, breaking
    # jaxlib`s CuDNN ABI. forward._ensure_larndsim_importable adds LARNDSIM_REPO
    # to sys.path at runtime, so the package imports straight from the source tree.

    # The SIF bundles its own jax 0.5.3 + nvidia-cudnn-cu12 9.1.0 under
    # /usr/local/lib/python3.10/dist-packages/. The dynamic linker resolves
    # libcudnn.so.9 from there via ld.so.cache, even though Python imports
    # jaxlib 0.6.2 from $HOME/.local (which expects CuDNN 9.8+). Prepend the
    # user-site nvidia/*/lib dirs to LD_LIBRARY_PATH so the matching 9.23
    # CuDNN wins.
    USER_SP=$HOME/.local/lib/python3.10/site-packages
    USER_NV_LIBS=$(printf "%s:" $USER_SP/nvidia/*/lib)
    export LD_LIBRARY_PATH="$USER_NV_LIBS$LD_LIBRARY_PATH"

    pip3 install --quiet --no-deps botorch gpytorch pyyaml
    cd '"${BEACON_DIR}"'
    python3 -m detector_calibration.bo_detector --config '"${CONFIG}"'
  '
